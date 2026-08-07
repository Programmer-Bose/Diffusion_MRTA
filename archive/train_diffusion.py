"""
Train a conditional diffusion model on multi-robot trajectories.

Input: outputs/preprocessed/dataset.npz  (from preprocess_dataset.py)
       outputs/preprocessed/obstacles.json

Trajectory tensor: (N, H, 4) normalized to [-1, 1] -> channels [x, y, vx, vy]
Context: start (2,), goal (2,), obstacles (padded fixed-size vector)

Run: python train_diffusion.py
Outputs: checkpoints/model.pt, loss curve plot, and a sample-reconstruction test.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------- CONFIG ----------------------
DATASET_PATH = "outputs/preprocessed/dataset.npz"
OBSTACLES_PATH = "outputs/preprocessed/obstacles.json"
CKPT_DIR = "outputs/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

H = 64                  # horizon (waypoints)
C = 4                   # channels: x, y, vx, vy
MAX_OBSTACLES = 30      # pad/truncate obstacle list to this many
OBS_FEAT_DIM = 4        # [type(0=circle,1=rect), pos_x, pos_y, size] per obstacle -> flattened

N_DIFFUSION_STEPS = 25          # fewer steps: more stable for small datasets
BETA_START, BETA_END = 1e-4, 0.02

BATCH_SIZE = 8
EPOCHS = 5000
LR = 1e-4                        # lowered to reduce loss oscillation
LR_DECAY_STEP = 1500
LR_DECAY_GAMMA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- OBSTACLE ENCODING ----------------------
def encode_obstacles(obs_list, max_obstacles=MAX_OBSTACLES):
    """Turn a variable-length obstacle list into a fixed-size flat vector."""
    feats = np.zeros((max_obstacles, OBS_FEAT_DIM), dtype=np.float32)
    for i, obs in enumerate(obs_list[:max_obstacles]):
        obs_type = 0.0 if obs["type"] == "circle" else 1.0
        px, py = obs["position"]
        size = obs.get("radius", 0.0) if obs["type"] == "circle" else max(
            obs.get("width", 0.0), obs.get("height", 0.0)
        )
        feats[i] = [obs_type, px, py, size]
    return feats.flatten()  # (max_obstacles * OBS_FEAT_DIM,)


# ---------------------- DATASET ----------------------
class TrajectoryDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, obstacles_path=OBSTACLES_PATH):
        data = np.load(dataset_path, allow_pickle=True)
        self.traj = data["traj_norm"].astype(np.float32)     # (N, H, 4)
        self.starts = data["starts"].astype(np.float32)      # (N, 2) raw units
        self.goals = data["goals"].astype(np.float32)        # (N, 2) raw units
        self.map_ids = data.get("map_ids", data.get("map_numbers"))
        self.robot_ids = data["robot_ids"]

        with open(obstacles_path) as f:
            obstacles_by_sample = json.load(f)

        # Normalize start/goal same range as x,y channels (use norm stats from traj)
        self.norm_min = data["norm_min"]
        self.norm_max = data["norm_max"]
        self.norm_range = data["norm_range"]

        self.obs_vecs = []
        for m, r in zip(self.map_ids, self.robot_ids):
            key = f"{m}_{r}"
            obs_list = obstacles_by_sample.get(key, [])
            self.obs_vecs.append(encode_obstacles(obs_list))
        self.obs_vecs = np.stack(self.obs_vecs).astype(np.float32)  # (N, max_obs*feat)

        # normalize start/goal using the x,y portion of norm stats (channels 0,1)
        xy_min, xy_range = self.norm_min[:2], self.norm_range[:2]
        self.starts_norm = (2 * (self.starts - xy_min) / xy_range - 1).astype(np.float32)
        self.goals_norm = (2 * (self.goals - xy_min) / xy_range - 1).astype(np.float32)

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        context = np.concatenate([
            self.starts_norm[idx], self.goals_norm[idx], self.obs_vecs[idx]
        ])  # (2+2+max_obs*feat,)
        return {
            "traj": torch.from_numpy(self.traj[idx]).permute(1, 0),  # (C, H)
            "context": torch.from_numpy(context),
        }


# ---------------------- DIFFUSION SCHEDULE ----------------------
def make_schedule(n_steps=N_DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END):
    betas = torch.linspace(beta_start, beta_end, n_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


# ---------------------- MODEL: small 1D temporal U-Net ----------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.mish(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = F.mish(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TemporalUNet1D(nn.Module):
    def __init__(self, channels=C, context_dim=None, base_ch=32, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Mish(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim * 2),  # wider: stronger context signal
        )
        emb_dim = time_emb_dim + time_emb_dim * 2  # concatenated, not summed -> context can't be washed out

        self.down1 = ResBlock1D(channels, base_ch, emb_dim)
        self.down2 = ResBlock1D(base_ch, base_ch * 2, emb_dim)
        self.pool = nn.AvgPool1d(2)

        self.mid = ResBlock1D(base_ch * 2, base_ch * 2, emb_dim)

        self.up_conv = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        # Up-sample concatenates the upsampled features (base_ch*2 channels) with the skip
        # connection from the first down block (base_ch channels).
        self.up1 = ResBlock1D(base_ch * 2 + base_ch, base_ch, emb_dim)
        self.out_conv = nn.Conv1d(base_ch, channels, 3, padding=1)

    def forward(self, x, t, context):
        # x: (B, C, H)
        t_emb = self.time_mlp(t)                  # (B, time_emb_dim)
        c_emb = self.context_mlp(context)          # (B, time_emb_dim*2)
        emb = torch.cat([t_emb, c_emb], dim=-1)     # (B, emb_dim) concatenated, context dominates

        h1 = self.down1(x, emb)                    # (B, base_ch, H)
        h1p = self.pool(h1)                         # (B, base_ch, H/2)
        h2 = self.down2(h1p, emb)                    # (B, base_ch*2, H/2)

        m = self.mid(h2, emb)                         # (B, base_ch*2, H/2)

        u = self.up_conv(m)                            # (B, base_ch*2, H)
        if u.shape[-1] != h1.shape[-1]:
            u = F.interpolate(u, size=h1.shape[-1])
        u = torch.cat([u, h1], dim=1)
        out = self.up1(u, emb)
        return self.out_conv(out)


# ---------------------- TRAINING ----------------------
def train():
    ds = TrajectoryDataset()
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, len(ds)), shuffle=True, drop_last=False)

    context_dim = 2 + 2 + MAX_OBSTACLES * OBS_FEAT_DIM
    model = TemporalUNet1D(channels=C, context_dim=context_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    betas, alphas, alphas_cumprod = make_schedule()
    betas, alphas, alphas_cumprod = betas.to(DEVICE), alphas.to(DEVICE), alphas_cumprod.to(DEVICE)

    losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in dl:
            traj = batch["traj"].to(DEVICE)          # (B, C, H)
            context = batch["context"].to(DEVICE)     # (B, context_dim)
            B = traj.shape[0]

            t = torch.randint(0, N_DIFFUSION_STEPS, (B,), device=DEVICE)
            noise = torch.randn_like(traj)

            sqrt_ac = alphas_cumprod[t].sqrt()[:, None, None]
            sqrt_1m_ac = (1 - alphas_cumprod[t]).sqrt()[:, None, None]
            traj_t = sqrt_ac * traj + sqrt_1m_ac * noise

            pred_noise = model(traj_t, t, context)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * B

        epoch_loss /= len(ds)
        losses.append(epoch_loss)
        scheduler.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:4d} | loss {epoch_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

    torch.save({
        "model_state": model.state_dict(),
        "context_dim": context_dim,
        "norm_min": ds.norm_min,
        "norm_max": ds.norm_max,
        "norm_range": ds.norm_range,
    }, os.path.join(CKPT_DIR, "model.pt"))

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE noise-prediction loss")
    plt.title("Training loss")
    plt.yscale("log")
    plt.savefig(os.path.join(CKPT_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    print(f"Saved model to {CKPT_DIR}/model.pt")
    print(f"Saved loss curve to {CKPT_DIR}/loss_curve.png")

    # ---- quick reconstruction sanity check on one training sample ----
    sanity_check_reconstruction(model, ds, betas, alphas, alphas_cumprod)


@torch.no_grad()
def sanity_check_reconstruction(model, ds, betas, alphas, alphas_cumprod, sample_idx=0):
    """Sample from pure noise conditioned on a known training sample's context,
    then compare against the ground-truth trajectory."""
    model.eval()
    context = torch.from_numpy(
        np.concatenate([ds.starts_norm[sample_idx], ds.goals_norm[sample_idx], ds.obs_vecs[sample_idx]])
    ).unsqueeze(0).float().to(DEVICE)

    x = torch.randn(1, C, H, device=DEVICE)
    for t in reversed(range(N_DIFFUSION_STEPS)):
        t_batch = torch.full((1,), t, device=DEVICE, dtype=torch.long)
        pred_noise = model(x, t_batch, context)

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_cumprod).sqrt()) * pred_noise)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = beta.sqrt()
            x = mean + sigma * noise
        else:
            x = mean

    generated = x.squeeze(0).permute(1, 0).cpu().numpy()  # (H, C) normalized
    ground_truth = ds.traj[sample_idx]  # (H, C) normalized

    mse = np.mean((generated - ground_truth) ** 2)
    print(f"\nSanity check: reconstruction MSE (normalized space) vs training sample {sample_idx} = {mse:.5f}")

    plt.figure(figsize=(6, 6))
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], label="ground truth", linewidth=2)
    plt.plot(generated[:, 0], generated[:, 1], label="generated", linestyle="--")
    plt.legend()
    plt.title(f"Reconstruction sanity check (sample {sample_idx})")
    plt.savefig(os.path.join(CKPT_DIR, "sanity_check_reconstruction.png"), dpi=150)
    plt.close()
    print(f"Saved reconstruction plot to {CKPT_DIR}/sanity_check_reconstruction.png")


if __name__ == "__main__":
    train()