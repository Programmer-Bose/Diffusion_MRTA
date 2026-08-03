import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Loaders (unchanged)
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

def zscore(x, axis=0, eps=1e-8):
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True) + eps
    return (x - mean) / std

def build_robot_features(scenario):
    pos = np.array(scenario["robot_pos"])
    battery = zscore(np.array(scenario["robot_battery"]).reshape(-1, 1))
    capacity = zscore(np.array(scenario["robot_max_payload"]).reshape(-1, 1))
    own_weight = zscore(np.array(scenario["robot_own_weight"]).reshape(-1, 1))
    return np.hstack([pos, battery, capacity, own_weight])

def build_task_features(scenario):
    pos = np.array(scenario["task_pos"])
    weight = zscore(np.array(scenario["task_weight"]).reshape(-1, 1))
    service_time = zscore(np.array(scenario["task_service_time"]).reshape(-1, 1))
    return np.hstack([pos, weight, service_time])


# ============================================================
# SIMPLIFIED model: NO self-attention encoder, just per-cell MLP
# (isolates whether nn.TransformerEncoder is the problem)
# ============================================================
class NoAttentionDenoiser(nn.Module):
    def __init__(self, robot_feat_dim, task_feat_dim, num_classes,
                 max_R=20, max_T=50, d_model=64, pref_dim=3):
        super().__init__()
        self.num_classes = num_classes
        self.robot_embed = nn.Linear(robot_feat_dim, d_model)
        self.task_embed = nn.Linear(task_feat_dim, d_model)
        self.schedule_embed = nn.Embedding(num_classes, d_model)
        self.time_embed = nn.Linear(1, d_model)
        self.pref_embed = nn.Linear(pref_dim, d_model)
        self.row_pos_embed = nn.Embedding(max_R, d_model)
        self.col_pos_embed = nn.Embedding(max_T, d_model)

        # per-cell MLP instead of Transformer encoder
        self.cell_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
        )
        self.out_head = nn.Linear(d_model, num_classes)

    def forward(self, robot_feats, task_feats, noisy_schedule, timestep, pref):
        B, R_, T_ = noisy_schedule.shape

        r_emb = self.robot_embed(robot_feats)
        t_emb = self.task_embed(task_feats)
        sched_emb = self.schedule_embed(noisy_schedule)

        cell_ctx = r_emb.unsqueeze(2) + t_emb.unsqueeze(1)
        cell_emb = sched_emb + cell_ctx

        time_emb = self.time_embed(timestep).unsqueeze(1).unsqueeze(1)
        cell_emb = cell_emb + time_emb

        device = cell_emb.device
        row_pe = self.row_pos_embed(torch.arange(R_, device=device)).unsqueeze(1)
        col_pe = self.col_pos_embed(torch.arange(T_, device=device)).unsqueeze(0)
        cell_emb = cell_emb + (row_pe + col_pe).unsqueeze(0)

        # preference injected additively (no cross-attn needed for this ablation)
        pref_emb = self.pref_embed(pref).unsqueeze(1).unsqueeze(1)  # (B,1,1,d)
        cell_emb = cell_emb + pref_emb

        x = self.cell_mlp(cell_emb)          # per-cell independent MLP
        logits = self.out_head(x)             # (B,R,T,num_classes)
        return logits


def compute_class_weights(gt_schedule, num_classes):
    counts = np.bincount(gt_schedule.flatten(), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def run_overfit_test(json_path="scenario.json", num_steps=50, epochs=1500, seed=0, lr=1e-3):
    scenario = load_scenario(json_path)
    R, T = scenario["R"], scenario["T"]
    num_classes = T + 1

    gt_schedule = build_ground_truth_schedule(scenario, seed=seed)
    robot_feats_np = build_robot_features(scenario)
    task_feats_np = build_task_features(scenario)

    robot_feats = torch.tensor(robot_feats_np, dtype=torch.float32).unsqueeze(0)
    task_feats = torch.tensor(task_feats_np, dtype=torch.float32).unsqueeze(0)
    gt_tensor = torch.tensor(gt_schedule, dtype=torch.long).unsqueeze(0)
    pref = torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float32)

    class_weights = compute_class_weights(gt_schedule, num_classes)
    print(f"Class weights: {class_weights.numpy().round(3)}")
    print(f"\nGround truth schedule:\n{gt_schedule}\n")

    torch.manual_seed(seed)
    model = NoAttentionDenoiser(robot_feats_np.shape[1], task_feats_np.shape[1], num_classes,
                                 max_R=max(20, R), max_T=max(50, T))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        t = rng.integers(1, num_steps + 1)
        noisy_np = forward_diffuse(gt_schedule, t, num_steps, num_classes, rng)
        noisy = torch.tensor(noisy_np, dtype=torch.long).unsqueeze(0)
        timestep = torch.tensor([[t / num_steps]], dtype=torch.float32)

        logits = model(robot_feats, task_feats, noisy, timestep, pref)
        loss = F.cross_entropy(logits.view(-1, num_classes), gt_tensor.view(-1), weight=class_weights)

        optimizer.zero_grad()
        loss.backward()

        # gradient norm diagnostic
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == gt_tensor).float().mean().item()
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}% | GradNorm: {total_norm:.6f}")

    return model, scenario, gt_schedule, robot_feats, task_feats, pref, num_steps, num_classes


def evaluate_reverse_sampling(model, robot_feats, task_feats, pref,
                               num_steps, num_classes, gt_schedule, seed=1, greedy_from_step=5):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    R, T = gt_schedule.shape
    x = torch.tensor(rng.integers(0, num_classes, size=(R, T)), dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for t in reversed(range(1, num_steps + 1)):
            timestep = torch.tensor([[t / num_steps]], dtype=torch.float32)
            logits = model(robot_feats, task_feats, x, timestep, pref)
            probs = F.softmax(logits, dim=-1)
            if t <= greedy_from_step:
                x = torch.argmax(probs, dim=-1)
            else:
                B, R_, T_, C = probs.shape
                sampled = torch.multinomial(probs.view(-1, C), 1).squeeze(-1)
                x = sampled.view(B, R_, T_)

    predicted = x.squeeze(0).numpy()
    print(f"\nPredicted schedule:\n{predicted}")
    print(f"Ground truth schedule:\n{gt_schedule}")
    acc = np.mean(predicted == gt_schedule)
    nz = gt_schedule > 0
    nz_acc = np.mean(predicted[nz] == gt_schedule[nz]) if nz.sum() > 0 else 0.0
    print(f"Overall accuracy: {acc*100:.2f}% | Non-zero accuracy: {nz_acc*100:.2f}%")
    return predicted, acc, nz_acc


if __name__ == "__main__":
    model, scenario, gt_schedule, robot_feats, task_feats, pref, num_steps, num_classes = \
        run_overfit_test(json_path="scenarios/scenario_3_15_2001.json", num_steps=50, epochs=1500, lr=1e-3)
    evaluate_reverse_sampling(model, robot_feats, task_feats, pref, num_steps, num_classes, gt_schedule)



