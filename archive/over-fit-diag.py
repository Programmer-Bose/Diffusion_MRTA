import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Loaders
# ============================================================
def load_scenario(filepath="scenario.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def build_ground_truth_schedule(scenario, seed=0):
    rng = np.random.default_rng(seed)
    R, T = scenario["R"], scenario["T"]
    capacity = np.array(scenario["robot_max_payload"])
    weight = np.array(scenario["task_weight"])
    task_order = rng.permutation(T)
    remaining_capacity = capacity.copy()
    schedule = np.zeros((R, T), dtype=int)
    robot_next_slot = np.ones(R, dtype=int)
    for t in task_order:
        candidates = np.where(remaining_capacity >= weight[t])[0]
        if len(candidates) == 0:
            continue
        r = rng.choice(candidates)
        schedule[r, t] = robot_next_slot[r]
        robot_next_slot[r] += 1
        remaining_capacity[r] -= weight[t]
    return schedule

def forward_diffuse(schedule, timestep, num_steps, num_classes, rng):
    corrupt_prob = timestep / num_steps
    noisy = schedule.copy()
    mask = rng.random(schedule.shape) < corrupt_prob
    random_vals = rng.integers(0, num_classes, size=schedule.shape)
    noisy[mask] = random_vals[mask]
    return noisy


# ============================================================
# NORMALIZED feature builders
# ============================================================
def zscore(x, axis=0, eps=1e-8):
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True) + eps
    return (x - mean) / std

def build_robot_features(scenario):
    pos = np.array(scenario["robot_pos"])                       # already in [0,1]^3, keep as-is
    battery = np.array(scenario["robot_battery"]).reshape(-1, 1)  # 85-100 -> normalize
    capacity = np.array(scenario["robot_max_payload"]).reshape(-1, 1)  # 10-25 -> normalize
    own_weight = np.array(scenario["robot_own_weight"]).reshape(-1, 1) # 5-12.5 -> normalize

    battery_norm = zscore(battery)
    capacity_norm = zscore(capacity)
    own_weight_norm = zscore(own_weight)

    feats = np.hstack([pos, battery_norm, capacity_norm, own_weight_norm])  # (R, 6)
    return feats

def build_task_features(scenario):
    pos = np.array(scenario["task_pos"])                          # already in [0,1]^3, keep as-is
    weight = np.array(scenario["task_weight"]).reshape(-1, 1)      # 1-6 -> normalize
    service_time = np.array(scenario["task_service_time"]).reshape(-1, 1)  # 4-10 -> normalize

    weight_norm = zscore(weight)
    service_time_norm = zscore(service_time)

    feats = np.hstack([pos, weight_norm, service_time_norm])  # (T, 5)
    return feats


# ============================================================
# Model (unchanged)
# ============================================================
class SimpleDenoiser(nn.Module):
    def __init__(self, robot_feat_dim, task_feat_dim, num_classes,
                 d_model=64, nhead=4, num_layers=2, pref_dim=3):
        super().__init__()
        self.num_classes = num_classes
        self.robot_embed = nn.Linear(robot_feat_dim, d_model)
        self.task_embed = nn.Linear(task_feat_dim, d_model)
        self.schedule_embed = nn.Embedding(num_classes, d_model)
        self.time_embed = nn.Linear(1, d_model)
        self.pref_embed = nn.Linear(pref_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.out_head = nn.Linear(d_model, num_classes)

    def forward(self, robot_feats, task_feats, noisy_schedule, timestep, pref, verbose=False):
        r_emb = self.robot_embed(robot_feats)
        t_emb = self.task_embed(task_feats)
        sched_emb = self.schedule_embed(noisy_schedule)

        cell_ctx = r_emb.unsqueeze(2) + t_emb.unsqueeze(1)
        cell_emb = sched_emb + cell_ctx

        time_emb = self.time_embed(timestep).unsqueeze(1).unsqueeze(1)
        cell_emb = cell_emb + time_emb

        if verbose:
            print("\n--- DIAGNOSTIC: embedding stats before encoder ---")
            print(f"robot_embed  mean={r_emb.mean().item():.4f} std={r_emb.std().item():.4f}")
            print(f"task_embed   mean={t_emb.mean().item():.4f} std={t_emb.std().item():.4f}")
            print(f"sched_embed  mean={sched_emb.mean().item():.4f} std={sched_emb.std().item():.4f}")
            print(f"time_embed   mean={time_emb.mean().item():.4f} std={time_emb.std().item():.4f}")
            print(f"cell_emb (pre-encoder) mean={cell_emb.mean().item():.4f} std={cell_emb.std().item():.4f}")
            flat = cell_emb.view(-1, cell_emb.shape[-1])
            pairwise_std = flat.std(dim=0).mean().item()
            print(f"Per-dimension std across all {flat.shape[0]} cells (pre-encoder): {pairwise_std:.6f}")

        B, R_, T_, D = cell_emb.shape
        x = cell_emb.view(B, R_ * T_, D)
        x = self.encoder(x)

        if verbose:
            flat_post = x.view(-1, x.shape[-1])
            pairwise_std_post = flat_post.std(dim=0).mean().item()
            print(f"Per-dimension std across all cells (post-encoder): {pairwise_std_post:.6f}")

        pref_emb = self.pref_embed(pref).unsqueeze(1)
        x, _ = self.cross_attn(x, pref_emb, pref_emb)

        logits = self.out_head(x)
        logits = logits.view(B, R_, T_, self.num_classes)

        if verbose:
            print(f"\nlogits stats: mean={logits.mean().item():.4f} std={logits.std().item():.4f}")
            probs = F.softmax(logits, dim=-1)
            flat_probs = probs.view(-1, self.num_classes)
            probs_std_across_cells = flat_probs.std(dim=0).mean().item()
            print(f"Softmax prob std across cells (higher = more distinguishable): {probs_std_across_cells:.6f}")
            print(f"Argmax prediction per cell (robot 0):\n{torch.argmax(logits[0,0], dim=-1)}")
            print(f"Argmax prediction per cell (robot 1):\n{torch.argmax(logits[0,1], dim=-1)}")

        return logits


# ============================================================
# DIAGNOSTIC RUN
# ============================================================
def run_diagnostic(json_path="scenario.json", num_steps=50, seed=0):
    scenario = load_scenario(json_path)
    R, T = scenario["R"], scenario["T"]
    num_classes = T + 1

    gt_schedule = build_ground_truth_schedule(scenario, seed=seed)
    robot_feats_np = build_robot_features(scenario)
    task_feats_np = build_task_features(scenario)

    robot_feats = torch.tensor(robot_feats_np, dtype=torch.float32).unsqueeze(0)
    task_feats = torch.tensor(task_feats_np, dtype=torch.float32).unsqueeze(0)
    pref = torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float32)

    rng = np.random.default_rng(seed)
    t = num_steps
    noisy_np = forward_diffuse(gt_schedule, t, num_steps, num_classes, rng)
    noisy = torch.tensor(noisy_np, dtype=torch.long).unsqueeze(0)
    timestep = torch.tensor([[t / num_steps]], dtype=torch.float32)

    torch.manual_seed(seed)
    model = SimpleDenoiser(robot_feats_np.shape[1], task_feats_np.shape[1], num_classes)
    model.eval()

    print(f"Normalized robot features:\n{robot_feats_np}")
    print(f"\nNormalized task features (first 5 tasks):\n{task_feats_np[:5]}")

    with torch.no_grad():
        logits = model(robot_feats, task_feats, noisy, timestep, pref, verbose=True)

    return model, logits



if __name__ == "__main__":
    run_diagnostic("scenarios/scenario_3_15_1001.json", num_steps=50, seed=0)