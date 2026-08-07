# ============================================================
# HYPERPARAMETERS (must match training config)
# ============================================================
CHECKPOINT_PATH        = "./checkpoints/diffusion_policy.pt"

CHUNK_LEN              = 16
N_EXEC                 = 8
N_LIDAR                = 30
LIDAR_ANGLE_STEP_DEG   = 360 / N_LIDAR
LIDAR_MAX_RANGE        = 2.0
ACTION_DIM             = 2
OBS_DIM                = N_LIDAR + 2 + 2

WHEEL_SEPARATION       = 0.105
RECORD_INTERVAL        = 0.25
steps_per_frame        = 16          # physics steps per RECORD_INTERVAL (matches recording rate)

TIME_EMB_DIM           = 32
OBS_EMB_DIM            = 64
UNET_CHANNELS          = (32, 64, 128)

T_DIFFUSION            = 100
BETA_START             = 1e-4
BETA_END               = 0.02

# Episode / arena spawn settings (same as recording script)
N_OBSTACLES            = 10
ARENA_HALF_SIZE        = 0.85
ROBOT_SPAWN_HALF_SIZE   = 0.5
OBSTACLE_RADIUS        = 0.03
MIN_OBSTACLE_DIST      = 1.0
MAX_EXTRA_DIST         = 0.5
GOAL_MIN_CLEARANCE     = 0.2
GOAL_REACH_THRESHOLD   = 0.1

DEVICE = "cuda"   # will fall back to cpu automatically below

# ============================================================
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import cv2

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Model definitions (must match training file) ----------------
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
        h = h * (1 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
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

        out = self.out_conv(h)
        return out.permute(0, 2, 1)


class Normalizer:
    def __init__(self):
        self.action_mean = None
        self.action_std = None

    def normalize_lidar(self, lidar):
        return lidar / LIDAR_MAX_RANGE

    def normalize_xy(self, xy):
        return xy / LIDAR_MAX_RANGE

    def denormalize_action(self, a):
        return a * self.action_std + self.action_mean

    def load_state_dict(self, sd):
        self.action_mean = sd["action_mean"]
        self.action_std = sd["action_std"]


# ---------------- Diffusion utilities ----------------
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

@torch.no_grad()
def sample_action_chunk(model, normalizer, lidar_raw, robot_pose, goal_xy, start_xy):
    x, y, yaw = robot_pose
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    dgx, dgy = goal_xy[0] - x, goal_xy[1] - y
    goal_rel = np.array([cos_y*dgx + sin_y*dgy, -sin_y*dgx + cos_y*dgy], dtype=np.float32)

    dsx, dsy = start_xy[0] - x, start_xy[1] - y
    start_rel = np.array([cos_y*dsx + sin_y*dsy, -sin_y*dsx + cos_y*dsy], dtype=np.float32)

    lidar_norm = normalizer.normalize_lidar(lidar_raw)
    goal_norm = normalizer.normalize_xy(goal_rel)
    start_norm = normalizer.normalize_xy(start_rel)
    obs = np.concatenate([lidar_norm, goal_norm, start_norm]).astype(np.float32)
    obs = torch.from_numpy(obs).unsqueeze(0).to(device)

    x_t = torch.randn((1, CHUNK_LEN, ACTION_DIM), device=device)

    for i in reversed(range(T_DIFFUSION)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        noise_pred = model(x_t, t, obs)

        beta_t = get_index(betas, t, x_t.shape)
        somac_t = get_index(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha_t = get_index(torch.sqrt(1.0 / alphas), t, x_t.shape)

        mean = sqrt_recip_alpha_t * (x_t - beta_t * noise_pred / somac_t)

        if i == 0:
            x_t = mean
        else:
            var_t = get_index(posterior_variance, t, x_t.shape)
            x_t = mean + torch.sqrt(var_t) * torch.randn_like(x_t)

    action_chunk = x_t.squeeze(0).cpu().numpy()
    return normalizer.denormalize_action(action_chunk)


def load_checkpoint_for_inference(path, model, normalizer):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    normalizer.load_state_dict(ckpt["normalizer"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")


# ---------------- Arena spawn (same logic as data collection) ----------------
def clip_to_arena(p, margin=OBSTACLE_RADIUS):
    return np.clip(p, -ARENA_HALF_SIZE + margin, ARENA_HALF_SIZE - margin)

def reset_episode(model, data, n_obstacles=N_OBSTACLES):
    robot_xy = np.random.uniform(-ROBOT_SPAWN_HALF_SIZE, ROBOT_SPAWN_HALF_SIZE, size=2)
    robot_yaw = np.random.uniform(-np.pi, np.pi)

    qpos_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    data.qpos[qpos_addr:qpos_addr+3] = [robot_xy[0], robot_xy[1], 0.031]
    quat = np.array([np.cos(robot_yaw/2), 0, 0, np.sin(robot_yaw/2)])
    data.qpos[qpos_addr+3:qpos_addr+7] = quat

    n_lidar_sites = 30
    lidar_angle_step = 360 / n_lidar_sites
    candidate_indices = list(range(0, n_lidar_sites, 2))
    chosen_indices = random.sample(candidate_indices, n_obstacles)
    obstacle_points = []

    for i, idx in enumerate(chosen_indices):
        angle = robot_yaw + np.radians(idx * lidar_angle_step)
        dist = np.random.uniform(MIN_OBSTACLE_DIST, MIN_OBSTACLE_DIST + MAX_EXTRA_DIST)
        obs_xy = clip_to_arena(robot_xy + dist * np.array([np.cos(angle), np.sin(angle)]))
        obstacle_points.append(obs_xy)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        model.body_pos[body_id][:2] = obs_xy
        model.body_pos[body_id][2] = 0.05

    goal_xy = None
    for _ in range(200):
        candidate = np.random.uniform(-ARENA_HALF_SIZE + 0.05, ARENA_HALF_SIZE - 0.05, size=2)
        if all(np.linalg.norm(candidate - o) >= (OBSTACLE_RADIUS + GOAL_MIN_CLEARANCE) for o in obstacle_points):
            goal_xy = candidate
            break
    if goal_xy is None:
        goal_xy = candidate

    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
    model.body_pos[goal_id][:2] = goal_xy
    model.body_pos[goal_id][2] = 0.05

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return robot_xy, robot_yaw, obstacle_points, goal_xy


# ---------------- Main inference loop ----------------
if __name__ == "__main__":
    mj_model = mujoco.MjModel.from_xml_path("mrs-world.xml")
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=800, width=800)

    left_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
    right_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
    root_jnt = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qpos_addr = mj_model.jnt_qposadr[root_jnt]

    policy = ConditionalUnet1D().to(device)
    normalizer = Normalizer()
    load_checkpoint_for_inference(CHECKPOINT_PATH, policy, normalizer)
    policy.eval()

    print("Q/ESC: quit | Diffusion policy driving the robot toward the goal.")

    episode_num = 1
    MAX_EPISODE_TIME = 60.0   # seconds, safety timeout

    while True:
        robot_start, robot_yaw0, obstacles, goal_xy = reset_episode(mj_model, mj_data)
        start_xy = robot_start.copy()
        print(f"\nEpisode {episode_num} start: robot={robot_start}, goal={goal_xy}")

        current_chunk = None
        chunk_step = 0
        episode_start_time = mj_data.time
        quit_requested = False

        while True:
            robot_x, robot_y = mj_data.qpos[qpos_addr], mj_data.qpos[qpos_addr+1]
            quat = mj_data.qpos[qpos_addr+3:qpos_addr+7]
            yaw = np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),
                              1-2*(quat[2]**2+quat[3]**2))

            # Replan when we've used up N_EXEC steps of the current chunk
            if current_chunk is None or chunk_step >= N_EXEC:
                lidar_raw = mj_data.sensordata.copy()
                lidar_raw = np.where(lidar_raw < 0, LIDAR_MAX_RANGE, lidar_raw)
                lidar_raw = np.clip(lidar_raw, 0, LIDAR_MAX_RANGE).astype(np.float32)

                current_chunk = sample_action_chunk(
                    policy, normalizer, lidar_raw,
                    (robot_x, robot_y, yaw), goal_xy, start_xy
                )
                chunk_step = 0

            left_ctrl, right_ctrl = current_chunk[chunk_step]
            mj_data.ctrl[left_id] = left_ctrl
            mj_data.ctrl[right_id] = right_ctrl

            for _ in range(steps_per_frame):
                mujoco.mj_step(mj_model, mj_data)
            chunk_step += 1

            dist_to_goal = np.linalg.norm([robot_x - goal_xy[0], robot_y - goal_xy[1]])

            renderer.update_scene(mj_data, camera="overhead_cam")
            pixels = renderer.render()
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"Ep {episode_num} | L:{left_ctrl:.3f} R:{right_ctrl:.3f} | dist:{dist_to_goal:.2f}",
                        (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            cv2.imshow("Diffusion Policy Inference", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                quit_requested = True
                break

            if dist_to_goal < GOAL_REACH_THRESHOLD:
                print("Goal reached!")
                break

            if mj_data.time - episode_start_time > MAX_EPISODE_TIME:
                print("Timeout — episode did not reach goal.")
                break

        episode_num += 1
        if quit_requested:
            break

    cv2.destroyAllWindows()