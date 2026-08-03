"""
model.py
--------
Core model components for the multi-UAV task assignment/sequencing
diffusion framework:

  1. D3PM-style categorical diffusion transition matrices (Markov chain)
  2. Sinkhorn column-normalization (assignment head post-processing)
  3. ThreeHeadedDenoiser network:
       - Assignment head  -> which robot gets which task
       - Rank head        -> execution order within a robot's route
       - Velocity head    -> Beta-distributed travel velocity per leg
     All conditioned on a user reference-point (preference) vector
     via a zero-init-gated cross-attention pathway.

UPDATE: added decode_schedule_capacity_aware(), a capacity-respecting
alternative to decode_schedule(). The original decode_schedule() does a
per-task-column argmax with NO notion of payload capacity, so even a
well-trained model can emit a schedule that overloads a robot -- which
is exactly what caused unrealistic SOC-drop values downstream in
objectives_and_reward.py / f450_energy2.py. decode_schedule() is kept
unchanged for backward compatibility; use the capacity-aware version
whenever task_weight / robot_max_payload are available (i.e. always,
in this project).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. D3PM-style Markov diffusion (categorical transition matrices)
# ============================================================
def make_transition_matrix(num_classes, beta_t):
    """Single-step transition matrix Q_t (num_classes x num_classes).
    With prob (1-beta_t) stay the same, with prob beta_t jump uniformly."""
    Q = np.full((num_classes, num_classes), beta_t / num_classes)
    np.fill_diagonal(Q, (1 - beta_t) + beta_t / num_classes)
    return Q


def make_beta_schedule(num_steps, beta_start=0.001, beta_end=0.2):
    return np.linspace(beta_start, beta_end, num_steps)


def cumulative_transition_matrices(num_classes, betas):
    """Returns (Q_list, Qbar_list): single-step and cumulative-product
    transition matrices for each diffusion step."""
    Q_list = [make_transition_matrix(num_classes, b) for b in betas]
    Qbar_list = []
    Qbar = np.eye(num_classes)
    for Q in Q_list:
        Qbar = Qbar @ Q
        Qbar_list.append(Qbar)
    return Q_list, Qbar_list


def compute_posterior(x_t, x_0, Q_t, Qbar_tm1, eps=1e-8):
    """q(x_{t-1} | x_t, x_0) via Bayes rule, normalized to a valid distribution."""
    term1 = Q_t[:, x_t]
    term2 = Qbar_tm1[x_0, :]
    unnorm = term1 * term2
    total = unnorm.sum() + eps
    return unnorm / total


# ============================================================
# 2. Sinkhorn column-normalization (assignment head)
#    Only column-normalized (each task -> exactly one robot);
#    a robot may take zero or many tasks, so no row constraint.
# ============================================================
def sinkhorn_column_normalize(logits, num_iters=20, temperature=0.5):
    log_alpha = logits / temperature
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return torch.exp(log_alpha)


# ============================================================
# 3. Three-Headed Denoiser
# ============================================================
class ThreeHeadedDenoiser(nn.Module):
    def __init__(self, robot_feat_dim, task_feat_dim,
                 d_model=64, nhead=4, num_layers=2, pref_dim=3,
                 min_vel=1.0, max_vel=10.0,
                 max_R=20, max_T=60):
        super().__init__()
        self.d_model = d_model
        self.min_vel = min_vel
        self.max_vel = max_vel

        # Input embeddings
        self.robot_embed = nn.Linear(robot_feat_dim, d_model)
        self.task_embed = nn.Linear(task_feat_dim, d_model)
        self.pref_embed = nn.Linear(pref_dim, d_model)

        # Row/col positional embeddings (identity survives self-attention)
        self.row_pos_embed = nn.Embedding(max_R, d_model)
        self.col_pos_embed = nn.Embedding(max_T, d_model)

        # Pre-LN Transformer encoder (self-attention over R*T cells)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Preference cross-attention, zero-init gated (LayerScale-style)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_gate = nn.Parameter(torch.zeros(d_model))

        # Head 1: assignment logits (pre-Sinkhorn), per (robot,task) cell
        self.assign_head = nn.Linear(d_model, 1)

        # Head 2: rank/order (continuous scalar per cell)
        self.rank_head = nn.Linear(d_model, 1)

        # Head 3: velocity via Beta distribution params (alpha, beta) per cell
        self.velocity_head = nn.Linear(d_model, 2)

    def forward(self, robot_feats, task_feats, pref, sample_velocity=True):
        """
        robot_feats: (B,R,robot_feat_dim)
        task_feats:  (B,T,task_feat_dim)
        pref:        (B,pref_dim)  -- reference-point / preference vector

        Returns:
          assign_logits: (B,R,T)  raw logits, pass through sinkhorn_column_normalize
          rank_raw:      (B,R,T)  continuous rank scores
          velocity:      (B,R,T)  sampled (or mean) velocity, bounded [min_vel,max_vel]
          alpha, beta:   (B,R,T)  Beta distribution parameters (for inspection/logging)
        """
        B, R, _ = robot_feats.shape
        _, T, _ = task_feats.shape

        r_emb = self.robot_embed(robot_feats)   # (B,R,d)
        t_emb = self.task_embed(task_feats)      # (B,T,d)

        cell_ctx = r_emb.unsqueeze(2) + t_emb.unsqueeze(1)  # (B,R,T,d)

        device = cell_ctx.device
        row_pe = self.row_pos_embed(torch.arange(R, device=device)).unsqueeze(1)  # (R,1,d)
        col_pe = self.col_pos_embed(torch.arange(T, device=device)).unsqueeze(0)   # (1,T,d)
        cell_emb = cell_ctx + (row_pe + col_pe).unsqueeze(0)                        # (B,R,T,d)

        x = cell_emb.view(B, R * T, self.d_model)
        x = self.encoder(x)

        pref_emb = self.pref_embed(pref).unsqueeze(1)   # (B,1,d)
        cross_out, _ = self.cross_attn(x, pref_emb, pref_emb)
        x = x + self.cross_gate * cross_out              # zero-init gated

        x = x.view(B, R, T, self.d_model)

        assign_logits = self.assign_head(x).squeeze(-1)   # (B,R,T)
        rank_raw = self.rank_head(x).squeeze(-1)           # (B,R,T)

        vel_params = self.velocity_head(x)                  # (B,R,T,2)
        alpha = F.softplus(vel_params[..., 0]) + 1e-3
        beta = F.softplus(vel_params[..., 1]) + 1e-3

        beta_dist = torch.distributions.Beta(alpha, beta)
        if sample_velocity:
            vel_unit = beta_dist.rsample()          # reparameterized sample, in (0,1)
        else:
            vel_unit = beta_dist.mean                # deterministic mean, for eval

        velocity = self.min_vel + (self.max_vel - self.min_vel) * vel_unit

        return assign_logits, rank_raw, velocity, alpha, beta


# ============================================================
# Decode (ORIGINAL): assignment (Sinkhorn -> hard argmax per task/column)
# + rank (only among cells assigned to that robot) -> valid order
# + velocity read off for each assigned leg
#
# NOTE: no payload-capacity awareness -- kept for backward compatibility
# and for callers that don't have weight/capacity data. Prefer
# decode_schedule_capacity_aware() below when those are available.
# ============================================================
def decode_schedule(assign_logits, rank_raw, velocity, temperature=0.5):
    """
    Expects batch size 1 (single scenario). Returns:
      schedule:     (R,T) int array, 0 = unassigned, 1..k = execution order
      velocity_out: (R,T) float array, velocity for assigned legs (0 elsewhere)
      hard_assign:  (T,) int array, which robot each task is assigned to
    """
    B, R, T = assign_logits.shape
    assert B == 1, "decode_schedule expects batch size 1"

    assign_logits = assign_logits.squeeze(0)
    rank_raw = rank_raw.squeeze(0)
    velocity = velocity.squeeze(0)

    P = sinkhorn_column_normalize(assign_logits, temperature=temperature)  # (R,T)
    hard_assign = torch.argmax(P, dim=0)  # (T,)

    schedule = np.zeros((R, T), dtype=int)
    velocity_out = np.zeros((R, T), dtype=float)

    for r in range(R):
        task_ids = [t for t in range(T) if hard_assign[t].item() == r]
        if len(task_ids) == 0:
            continue
        ranks = [rank_raw[r, t].item() for t in task_ids]
        order = np.argsort(ranks)
        for pos, idx in enumerate(order):
            t = task_ids[idx]
            schedule[r, t] = pos + 1
            velocity_out[r, t] = velocity[r, t].item()

    return schedule, velocity_out, hard_assign.cpu().numpy()


# ============================================================
# Decode (CAPACITY-AWARE): same Sinkhorn + rank machinery, but the hard
# assignment step now respects each robot's payload capacity instead of
# a naive per-column argmax.
#
# Algorithm: take the column-normalized assignment probabilities P (R,T).
# Process tasks in order of DECREASING max-probability (most confident
# assignments first -- mirrors how Sinkhorn/argmax already favors high-
# confidence cells). For each task, try robots in order of their P-score
# for that task (best first); assign to the first robot with enough
# REMAINING capacity. If no robot can feasibly take it, leave the task
# unassigned (schedule value 0) rather than silently overloading a robot
# -- an explicitly unassigned task is a visible, penalizable outcome
# (downstream reward/logging can flag unassigned tasks), whereas a
# silent capacity violation was invisible until the SOC blew up.
# ============================================================
def decode_schedule_capacity_aware(
    assign_logits, rank_raw, velocity,
    task_weight, robot_max_payload,
    temperature=0.5,
):
    """
    Expects batch size 1 (single scenario).

    task_weight       : (T,) array/tensor, kg per task
    robot_max_payload  : (R,) array/tensor, kg max liftable payload per robot

    Returns:
      schedule:      (R,T) int array, 0 = unassigned, 1..k = execution order
      velocity_out:  (R,T) float array, velocity for assigned legs (0 elsewhere)
      hard_assign:   (T,) int array, robot index per task, or -1 if the
                     task could not be feasibly assigned to any robot
      unassigned:    list of task indices left unassigned due to capacity
    """
    B, R, T = assign_logits.shape
    assert B == 1, "decode_schedule_capacity_aware expects batch size 1"

    assign_logits = assign_logits.squeeze(0)
    rank_raw = rank_raw.squeeze(0)
    velocity = velocity.squeeze(0)

    task_weight = np.asarray(
        task_weight.detach().cpu().numpy() if torch.is_tensor(task_weight) else task_weight,
        dtype=np.float64,
    )
    remaining_capacity = np.asarray(
        robot_max_payload.detach().cpu().numpy()
        if torch.is_tensor(robot_max_payload) else robot_max_payload,
        dtype=np.float64,
    ).copy()

    P = sinkhorn_column_normalize(assign_logits, temperature=temperature)  # (R,T)
    P_np = P.detach().cpu().numpy()

    # Process tasks most-confident-assignment-first.
    task_order = np.argsort(-P_np.max(axis=0))  # (T,) task indices, descending confidence
    hard_assign = np.full(T, -1, dtype=int)
    unassigned = []

    for t in task_order:
        # Robots ranked best-to-worst for this task by assignment probability.
        robot_pref_order = np.argsort(-P_np[:, t])
        w = task_weight[t]
        placed = False
        for r in robot_pref_order:
            if remaining_capacity[r] >= w:
                hard_assign[t] = r
                remaining_capacity[r] -= w
                placed = True
                break
        if not placed:
            unassigned.append(int(t))

    schedule = np.zeros((R, T), dtype=int)
    velocity_out = np.zeros((R, T), dtype=float)

    for r in range(R):
        task_ids = [t for t in range(T) if hard_assign[t] == r]
        if len(task_ids) == 0:
            continue
        ranks = [rank_raw[r, t].item() for t in task_ids]
        order = np.argsort(ranks)
        for pos, idx in enumerate(order):
            t = task_ids[idx]
            schedule[r, t] = pos + 1
            velocity_out[r, t] = velocity[r, t].item()

    return schedule, velocity_out, hard_assign, unassigned


# ============================================================
# Quick self-test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    R, T = 3, 15
    robot_feat_dim, task_feat_dim, pref_dim = 6, 5, 3

    model = ThreeHeadedDenoiser(robot_feat_dim, task_feat_dim,
                                 max_R=max(20, R), max_T=max(60, T))
    model.eval()

    robot_feats = torch.randn(1, R, robot_feat_dim)
    task_feats = torch.randn(1, T, task_feat_dim)
    pref = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.float32)

    with torch.no_grad():
        assign_logits, rank_raw, velocity, alpha, beta = model(robot_feats, task_feats, pref)

    schedule, velocity_out, hard_assign = decode_schedule(assign_logits, rank_raw, velocity)

    print(f"Schedule:\n{schedule}")
    print(f"Task->Robot assignment: {hard_assign}")
    print(f"Velocities (assigned legs): {velocity_out[schedule > 0].round(3)}")
    print(f"Velocity std: {velocity_out[schedule > 0].std():.4f}")

    # --- Capacity-aware decode demo ---
    task_weight = np.random.uniform(0.1, 1.0, size=T)   # kg
    robot_max_payload = np.array([1.5, 1.5, 1.5])        # kg, small robots

    schedule_c, velocity_out_c, hard_assign_c, unassigned = decode_schedule_capacity_aware(
        assign_logits, rank_raw, velocity, task_weight, robot_max_payload
    )
    print(f"\n[Capacity-aware] Schedule:\n{schedule_c}")
    print(f"[Capacity-aware] Task->Robot assignment: {hard_assign_c}")
    print(f"[Capacity-aware] Unassigned tasks (infeasible): {unassigned}")
    loads = [task_weight[schedule_c[r] > 0].sum() for r in range(R)]
    print(f"[Capacity-aware] Per-robot load vs capacity: {np.round(loads, 3)} <= {robot_max_payload}")