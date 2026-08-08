
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# HYPERPARAMETERS
# ============================================================
DATA_DIR              = "./episodes"        # folder with episode_*.npz
CHECKPOINT_DIR         = "./checkpoints"
CHECKPOINT_PATH        = f"{CHECKPOINT_DIR}/diffusion_policy.pt"

CHUNK_LEN              = 16          # action chunk length
N_EXEC                 = 8           # steps executed before replanning (inference only)
N_LIDAR                = 30
LIDAR_ANGLE_STEP_DEG   = 360 / N_LIDAR
LIDAR_MAX_RANGE        = 2.0         # meters; -1 (no hit) is clipped to this
ACTION_DIM             = 2           # [left_ctrl, right_ctrl]
OBS_DIM                = N_LIDAR + 2 + 2   # lidar(30) + goal_rel(2) + start_rel(2) = 34

WHEEL_SEPARATION       = 0.105
RECORD_INTERVAL        = 0.25        # seconds between recorded steps (used as dt in rollout)
SAFETY_MARGIN          = 0.05        # meters, for collision penalty
COLLISION_LOSS_WEIGHT  = 0.1         # weight of collision penalty vs diffusion loss

# Model architecture
TIME_EMB_DIM           = 32
OBS_EMB_DIM            = 64
UNET_CHANNELS          = (32, 64, 128)

# Diffusion
T_DIFFUSION            = 100
BETA_START             = 1e-4
BETA_END               = 0.02

# Training
BATCH_SIZE             = 64
LR                     = 1e-4
EPOCHS                 = 300
SAVE_EVERY             = 20

DEVICE                 = "cuda" if __import__("torch").cuda.is_available() else "cpu"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device(DEVICE)
print("Using device:", device)

def load_episode_windows(data_dir, chunk_len=CHUNK_LEN):
    """Loads all episode_*.npz files and slices into sliding-window samples."""
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    print(f"Found {len(files)} episode files.")

    samples = []  # each: dict(lidar, goal_rel, start_rel, action_chunk)

    for f in files:
        d = np.load(f)
        lidar = d["lidar"].astype(np.float32)     # [T, 30]
        action = d["action"].astype(np.float32)   # [T, 2]
        pose = d["pose"].astype(np.float32)        # [T, 3] -> x,y,yaw
        goal = d["goal"].astype(np.float32)         # [2]

        T = action.shape[0]
        if T < chunk_len:
            continue

        # clip -1 (no-hit) lidar readings to max range
        lidar = np.where(lidar < 0, LIDAR_MAX_RANGE, lidar)
        lidar = np.clip(lidar, 0, LIDAR_MAX_RANGE)

        start_xy = pose[0, :2]

        for t in range(0, T - chunk_len + 1):
            x, y, yaw = pose[t]
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)

            dgx, dgy = goal[0] - x, goal[1] - y
            goal_rel = np.array([cos_y * dgx + sin_y * dgy,
                                  -sin_y * dgx + cos_y * dgy], dtype=np.float32)

            dsx, dsy = start_xy[0] - x, start_xy[1] - y
            start_rel = np.array([cos_y * dsx + sin_y * dsy,
                                   -sin_y * dsx + cos_y * dsy], dtype=np.float32)

            samples.append({
                "lidar": lidar[t],                          # [30]
                "goal_rel": goal_rel,                        # [2]
                "start_rel": start_rel,                      # [2]
                "action_chunk": action[t:t + chunk_len],      # [chunk_len, 2]
            })

    return samples


class Normalizer:
    """Computes and applies simple normalization stats over the dataset."""
    def __init__(self, samples):
        actions = np.concatenate([s["action_chunk"] for s in samples], axis=0)
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-6

    def normalize_action(self, a):
        return (a - self.action_mean) / self.action_std

    def denormalize_action(self, a):
        return a * self.action_std + self.action_mean

    def normalize_lidar(self, lidar):
        return lidar / LIDAR_MAX_RANGE

    def normalize_xy(self, xy):
        return xy / (LIDAR_MAX_RANGE)  # rough meter-scale normalization

    def state_dict(self):
        return {"action_mean": self.action_mean, "action_std": self.action_std}

    def load_state_dict(self, sd):
        self.action_mean = sd["action_mean"]
        self.action_std = sd["action_std"]


class DiffusionPolicyDataset(Dataset):
    def __init__(self, samples, normalizer):
        self.samples = samples
        self.norm = normalizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        lidar = self.norm.normalize_lidar(s["lidar"])
        goal_rel = self.norm.normalize_xy(s["goal_rel"])
        start_rel = self.norm.normalize_xy(s["start_rel"])
        obs = np.concatenate([lidar, goal_rel, start_rel]).astype(np.float32)  # [34]

        action = self.norm.normalize_action(s["action_chunk"]).astype(np.float32)  # [16,2]

        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
            "raw_lidar": torch.from_numpy(s["lidar"].astype(np.float32)),  # for collision penalty (meters)
        }

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.Mish()
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        scale, bias = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)   # FiLM modulation
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim=ACTION_DIM, obs_dim=OBS_DIM,
                 channels=UNET_CHANNELS, time_emb_dim=TIME_EMB_DIM, obs_emb_dim=OBS_EMB_DIM):
        super().__init__()
        cond_dim = time_emb_dim + obs_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.Mish()
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Mish(),
            nn.Linear(128, obs_emb_dim), nn.Mish()
        )

        self.in_conv = nn.Conv1d(action_dim, channels[0], 3, padding=1)

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(FiLMResidualBlock1D(channels[i], channels[i + 1], cond_dim))
            self.pools.append(nn.Conv1d(channels[i + 1], channels[i + 1], 3, stride=2, padding=1))

        self.mid = FiLMResidualBlock1D(channels[-1], channels[-1], cond_dim)

        self.ups = nn.ModuleList()
        self.unpools = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            self.unpools.append(nn.ConvTranspose1d(rev_channels[i], rev_channels[i], 4, stride=2, padding=1))
            self.ups.append(FiLMResidualBlock1D(rev_channels[i] * 2, rev_channels[i + 1], cond_dim))

        self.out_conv = nn.Conv1d(channels[0], action_dim, 3, padding=1)

    def forward(self, x, timestep, obs):
        # x: [B, chunk_len, action_dim] -> [B, action_dim, chunk_len]
        x = x.permute(0, 2, 1)
        t_emb = self.time_mlp(timestep)
        o_emb = self.obs_encoder(obs)
        cond = torch.cat([t_emb, o_emb], dim=-1)

        h = self.in_conv(x)
        skips = []
        for block, pool in zip(self.downs, self.pools):
            h = block(h, cond)
            skips.append(h)
            h = pool(h)

        h = self.mid(h, cond)

        for block, unpool in zip(self.ups, self.unpools):
            h = unpool(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        out = self.out_conv(h)          # [B, action_dim, chunk_len]
        return out.permute(0, 2, 1)      # [B, chunk_len, action_dim]

betas = torch.linspace(BETA_START, BETA_END, T_DIFFUSION).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def get_index(vals, t, shape):
    out = vals.gather(-1, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

def forward_diffusion_sample(x0, t):
    noise = torch.randn_like(x0)
    sac = get_index(sqrt_alphas_cumprod, t, x0.shape)
    somac = get_index(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    x_t = sac * x0 + somac * noise
    return x_t, noise

def predict_x0_from_noise(x_t, t, noise_pred):
    sac = get_index(sqrt_alphas_cumprod, t, x_t.shape)
    somac = get_index(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    return (x_t - somac * noise_pred) / sac

def collision_penalty(action_chunk_denorm, raw_lidar, normalizer):
    """
    action_chunk_denorm: [B, chunk_len, 2] real-scale (left_ctrl, right_ctrl)
    raw_lidar: [B, 30] real meters (already clipped to max range)
    Rolls the chunk forward as an approximate unicycle path in the robot's
    local frame at t=0, and penalizes points that approach/exceed the
    LiDAR-sensed free distance in that direction.
    """
    B, L, _ = action_chunk_denorm.shape
    v = (action_chunk_denorm[..., 0] + action_chunk_denorm[..., 1]) / 2       # [B, L]
    omega = (action_chunk_denorm[..., 1] - action_chunk_denorm[..., 0]) / WHEEL_SEPARATION

    x = torch.zeros(B, device=action_chunk_denorm.device)
    y = torch.zeros(B, device=action_chunk_denorm.device)
    theta = torch.zeros(B, device=action_chunk_denorm.device)

    total_penalty = torch.zeros(B, device=action_chunk_denorm.device)

    for i in range(L):
        x = x + v[:, i] * RECORD_INTERVAL * torch.cos(theta)
        y = y + v[:, i] * RECORD_INTERVAL * torch.sin(theta)
        theta = theta + omega[:, i] * RECORD_INTERVAL

        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        angle_deg = (torch.atan2(y, x) * 180.0 / math.pi) % 360.0
        idx = torch.round(angle_deg / LIDAR_ANGLE_STEP_DEG).long() % N_LIDAR

        lidar_dist = raw_lidar.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B]
        penalty = F.relu(r - (lidar_dist - SAFETY_MARGIN))
        total_penalty = total_penalty + penalty

    return total_penalty.mean()

def save_checkpoint(path, model, optimizer, epoch, normalizer):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path} (epoch {epoch})")

def load_checkpoint(path, model, optimizer=None, normalizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if normalizer is not None:
        normalizer.load_state_dict(ckpt["normalizer"])
    print(f"Checkpoint loaded: {path} (resuming from epoch {ckpt['epoch']})")
    return ckpt["epoch"]

def train():
    samples = load_episode_windows(DATA_DIR)
    normalizer = Normalizer(samples)
    dataset = DiffusionPolicyDataset(samples, normalizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = ConditionalUnet1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer, normalizer) + 1

    for epoch in range(start_epoch, EPOCHS):
        epoch_diff_loss, epoch_coll_loss = 0.0, 0.0

        for batch in dataloader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            raw_lidar = batch["raw_lidar"].to(device)

            t = torch.randint(0, T_DIFFUSION, (action.shape[0],), device=device).long()
            x_noisy, noise = forward_diffusion_sample(action, t)
            noise_pred = model(x_noisy, t, obs)

            diff_loss = F.mse_loss(noise_pred, noise)

            x0_pred = predict_x0_from_noise(x_noisy, t, noise_pred)
            x0_pred_denorm = torch.from_numpy(normalizer.action_std).to(device) * x0_pred + \
                              torch.from_numpy(normalizer.action_mean).to(device)
            coll_loss = collision_penalty(x0_pred_denorm, raw_lidar, normalizer)

            loss = diff_loss + COLLISION_LOSS_WEIGHT * coll_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_diff_loss += diff_loss.item()
            epoch_coll_loss += coll_loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | diff_loss={epoch_diff_loss/len(dataloader):.4f} "
                  f"| collision_loss={epoch_coll_loss/len(dataloader):.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            save_checkpoint(CHECKPOINT_PATH, model, optimizer, epoch, normalizer)

    return model, normalizer

if __name__ == "__main__":
    train()