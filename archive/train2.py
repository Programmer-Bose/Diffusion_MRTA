"""
train2.py (FIXED)
------------------
Two-phase training for ThreeHeadedDenoiser (model.py).

FIXES in this version:
  1. RUNNING reward normalization (not noisy per-8-sample z-scores).
  2. KL-to-reference penalty during RL, so updates stay anchored to
     the known-good pretrained policy instead of random-walking into
     worse regions (this is what caused makespan to get WORSE, not
     better, from step 600->720).
  3. RANDOM preference vector sampled PER ROLLOUT during RL (instead
     of one fixed vector per training step), so gradients actually
     correlate "preference input changed" with "reward changed" --
     without this, the zero-init preference cross-attention gate has
     no learning pressure to ever open.
  4. PREFERENCE_SETS is now used only for periodic EVALUATION/logging
     (to see if preference-following is emerging), not as the sole
     training distribution.
"""

import os
import copy
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model2 import (
    ThreeHeadedDenoiser,
    sample_assignment_with_logprob,
    velocity_log_prob,
    sinkhorn_column_normalize,
)
from scenario_utils import (
    load_scenario_file,
    build_robot_features,
    build_task_features,
    build_balanced_random_schedule,
)
from objectives_and_reward import compute_objectives, _CAPACITY_TERMINAL_PENALTY

# =============================================================================
# HYPERPARAMETERS -- edit here
# =============================================================================

# --- Data & Directory ---
SCENARIO_FOLDER   = "./scenarios"
CHECKPOINT_DIR     = "./checkpoints"
RESUME_CHECKPOINT  = None            # path to .pt to resume RL from, or None
PRETRAIN_CHECKPOINT_TO_LOAD = None   # if set, skip Phase 1 and load this instead

# --- Preference vectors used ONLY for periodic evaluation/logging ---
PREFERENCE_SETS = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.34, 0.33, 0.33],
    
]

# --- Phase 1: Supervised Pretraining ---
PRETRAIN_EPOCHS          = 50
PRETRAIN_STEPS_PER_EPOCH = 50
PRETRAIN_LR               = 1e-4

# --- Phase 2: RL Fine-tuning ---
RL_ITERS_PER_FILE   = 15
RL_BATCH_SIZE        = 64          # larger batch = less noisy running stats
RL_LR                = 3e-4        # LOWERED 10x from before -- REINFORCE was too aggressive
EVAL_EVERY_N_STEPS   = 50           # run PREFERENCE_SETS eval + print every N RL steps

# --- KL-to-reference penalty (NEW) ---
KL_COEF              = 0.15         # weight of KL(current_policy || reference_policy)
REFERENCE_UPDATE_EVERY = 300        # re-snapshot reference policy every N RL steps
                                     # (slowly lets the anchor move forward once stable)

# --- Shared Optimizer Settings ---
WEIGHT_DECAY         = 1e-5
GRAD_CLIP_NORM        = 1.0
SAVE_INTERVAL_STEPS  = 50

# --- Model Architecture ---
D_MODEL      = 256
N_HEAD       = 8
NUM_LAYERS   = 3
PREF_DIM     = 3
MIN_VEL      = 1.0
MAX_VEL      = 5.0
MAX_R        = 20
MAX_T        = 60

# --- Diffusion (used for x_t conditioning during rollouts) ---
DIFFUSION_STEPS = 50

# --- Reward / Penalty ---
CAPACITY_PENALTY_WEIGHT = _CAPACITY_TERMINAL_PENALTY
BATTERY_PENALTY_WEIGHT   = 2.0
NORM_MOMENTUM            = 0.01     # running-normalizer EMA momentum

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_VELOCITY_FALLBACK = 5.0
UNASSIGNED_PENALTY_WEIGHT = 50.0   # NEW: penalty for each task left unassigned, scaled by fraction of total tasks
# =============================================================================


def save_checkpoint(step, model, optimizer, loss, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, filepath)
    print(f"[checkpoint] saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    if not os.path.exists(filepath):
        print(f"[checkpoint] {filepath} not found, starting fresh")
        return 0
    ckpt = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt.get("step", 0)
    print(f"[checkpoint] resumed from {filepath} (step={step})")
    return step


def list_scenario_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "scenario_*.json")))
    if not files:
        raise RuntimeError(f"No scenario files found in {folder}")
    return files


def sample_random_preference():
    """Dirichlet(1,1,1) sample -> a random 3-vector that sums to 1."""
    w = np.random.dirichlet([1.0, 1.0, 1.0])
    return w.tolist()


# =============================================================================
# NEW: Running (EMA) normalizer -- replaces noisy per-batch z-scoring
# =============================================================================
class RunningNormalizer:
    def __init__(self, eps=1e-8, momentum=NORM_MOMENTUM):
        self.mean = None
        self.var = 1.0
        self.eps = eps
        self.momentum = momentum

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return
        batch_mean, batch_var = values.mean(), values.var()
        if self.mean is None:
            self.mean, self.var = batch_mean, batch_var
        else:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var = (1 - self.momentum) * self.var + self.momentum * batch_var

    def normalize(self, values):
        values = np.asarray(values, dtype=np.float64)
        return (values - self.mean) / (np.sqrt(self.var) + self.eps)


# =============================================================================
# PHASE 1: Supervised pretraining (unchanged logic)
# =============================================================================
def pretrain(model, optimizer, scenario_files):
    print(f"\n=== PHASE 1: Supervised pretraining ({PRETRAIN_EPOCHS} epochs) ===")
    model.train()

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        epoch_loss = 0.0

        for step in range(PRETRAIN_STEPS_PER_EPOCH):
            optimizer.zero_grad()

            filepath = np.random.choice(scenario_files)
            scenario = load_scenario_file(filepath)
            R, T = scenario["R"], scenario["T"]

            r_feats = torch.tensor(build_robot_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t_feats = torch.tensor(build_task_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pref = torch.tensor([sample_random_preference()], dtype=torch.float32).to(DEVICE)

            target_schedule_np = build_balanced_random_schedule(scenario)
            target_assignments = np.full(T, R, dtype=np.int64)
            for t_idx in range(T):
                assigned_r = np.where(target_schedule_np[:, t_idx] > 0)[0]
                if len(assigned_r) > 0:
                    target_assignments[t_idx] = assigned_r[0]
            target_assignments_t = torch.tensor(target_assignments, dtype=torch.long).to(DEVICE)

            t = torch.randint(1, DIFFUSION_STEPS + 1, (1,)).to(DEVICE)
            noise_mask = torch.rand(1, T).to(DEVICE) < (t.float() / DIFFUSION_STEPS)
            random_assignments = torch.randint(0, R + 1, (1, T)).to(DEVICE)
            x_t_noisy = torch.where(noise_mask, random_assignments, target_assignments_t.unsqueeze(0))

            assign_logits, rank_raw, velocity, alpha, beta = model(
                r_feats, t_feats, pref, x_t=x_t_noisy, t=t
            )

            loss = F.cross_entropy(
                assign_logits.squeeze(0).transpose(0, 1),
                target_assignments_t,
                ignore_index=R,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / PRETRAIN_STEPS_PER_EPOCH
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [pretrain] epoch {epoch}/{PRETRAIN_EPOCHS}  avg_loss={avg:.4f}")

    print("=== PHASE 1 complete ===\n")


# =============================================================================
# Rollout + reward for ONE scenario, ONE (possibly random) preference vector
# =============================================================================
def rollout_and_score(model, scenario, weights):
    R, T = scenario["R"], scenario["T"]

    r_feats = torch.tensor(build_robot_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_feats = torch.tensor(build_task_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pref = torch.tensor([weights], dtype=torch.float32).to(DEVICE)

    t = torch.randint(1, DIFFUSION_STEPS + 1, (1,)).to(DEVICE)
    x_t_noisy = torch.randint(0, R + 1, (1, T)).to(DEVICE)

    assign_logits, rank_raw, velocity, alpha, beta = model(
        r_feats, t_feats, pref, x_t=x_t_noisy, t=t, sample_velocity=True
    )

    hard_assign, log_prob_assign, P_current = sample_assignment_with_logprob(assign_logits)
    log_prob_vel = velocity_log_prob(alpha, beta, velocity, MIN_VEL, MAX_VEL)
    total_log_prob = log_prob_assign + log_prob_vel

    hard_assign_np = hard_assign.squeeze(0).cpu().numpy()
    schedule = np.zeros((R, T), dtype=int)
    velocity_np = velocity.squeeze(0).detach().cpu().numpy()
    velocity_out = np.zeros((R, T), dtype=float)

    for r in range(R):
        task_ids = [t_idx for t_idx in range(T) if hard_assign_np[t_idx] == r]
        if not task_ids:
            continue
        ranks = [rank_raw.squeeze(0)[r, t_idx].item() for t_idx in task_ids]
        order = np.argsort(ranks)
        for pos, idx in enumerate(order):
            t_idx = task_ids[idx]
            schedule[r, t_idx] = pos + 1
            velocity_out[r, t_idx] = velocity_np[r, t_idx]

    velocity_full = np.where(velocity_out > 0, velocity_out, DEFAULT_VELOCITY_FALLBACK)
    obj = compute_objectives(scenario, schedule, velocity_full)

    return obj, total_log_prob, assign_logits, r_feats, t_feats, pref, x_t_noisy, t


# =============================================================================
# NEW: KL(current || reference) penalty, computed on the assignment
# distribution (Sinkhorn P), using a frozen snapshot of a known-good policy.
# =============================================================================
def kl_to_reference(reference_model, assign_logits, r_feats, t_feats, pref, x_t_noisy, t):
    with torch.no_grad():
        ref_logits, _, _, _, _ = reference_model(r_feats, t_feats, pref, x_t=x_t_noisy, t=t)
        P_ref = sinkhorn_column_normalize(ref_logits)

    P_current = sinkhorn_column_normalize(assign_logits)
    kl = (P_current * (torch.log(P_current + 1e-8) - torch.log(P_ref + 1e-8))).sum()
    return kl


# =============================================================================
# Periodic evaluation: does the model actually respond to different
# preference vectors? (deterministic, no grad)
# =============================================================================
def evaluate_preference_following(model, scenario):
    model.eval()
    print("  [eval] preference-following check:")
    with torch.no_grad():
        for weights in PREFERENCE_SETS:
            obj, _, _, _, _, _, _, _ = rollout_and_score(model, scenario, weights)
            print(f"    w={weights}  makespan={obj['makespan']:.2f}  "
                  f"energy_var={obj['workload_variance']:.2f}  energy={obj['total_energy']:.2f}  "
                  f"done={obj['done']}")
    model.train()


# =============================================================================
# PHASE 2: RL fine-tuning -- running normalization + KL anchor + random pref
# =============================================================================
def rl_finetune(model, optimizer, scenario_files, global_step_start=0):
    print(f"=== PHASE 2: RL fine-tuning ({len(scenario_files)} scenario files) ===")

    reference_model = copy.deepcopy(model).to(DEVICE)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)

    makespan_norm = RunningNormalizer()
    variance_norm = RunningNormalizer()
    energy_norm = RunningNormalizer()

    model.train()
    step = global_step_start

    for f_idx, filepath in enumerate(scenario_files):
        scenario = load_scenario_file(filepath)
        print(f"\n[{f_idx+1}/{len(scenario_files)}] {filepath} (R={scenario['R']}, T={scenario['T']})")

        for it in range(RL_ITERS_PER_FILE):
            optimizer.zero_grad()

            makespans, variances, energies = [], [], []
            dones, overload_fracs = [], []
            log_probs, kls = [], []
            weights_batch, obj_list = [], []          # NEW: track per-sample

            for _ in range(RL_BATCH_SIZE):
                weights = sample_random_preference()
                weights_batch.append(weights)          # NEW

                obj, log_prob, assign_logits, r_feats, t_feats, pref, x_t_noisy, t = \
                    rollout_and_score(model, scenario, weights)

                kl = kl_to_reference(reference_model, assign_logits, r_feats, t_feats, pref, x_t_noisy, t)

                makespans.append(obj["makespan"])
                variances.append(obj["workload_variance"])
                energies.append(obj["total_energy"])
                dones.append(obj["done"])
                overload_fracs.append(obj["max_overload_frac"])
                log_probs.append(log_prob)
                kls.append(kl)
                obj_list.append(obj)                    # NEW

            dones = np.array(dones, dtype=bool)
            overload_fracs = np.array(overload_fracs, dtype=np.float64)
            weights_batch = np.array(weights_batch)      # NEW, shape [RL_BATCH_SIZE, 3]
            log_probs_t = torch.cat(log_probs)
            kls_t = torch.stack(kls)

            reward = np.zeros(RL_BATCH_SIZE, dtype=np.float64)
            reward[dones] = -CAPACITY_PENALTY_WEIGHT * (1.0 + overload_fracs[dones])

            non_terminal = ~dones
            if non_terminal.any():
                idx = np.where(non_terminal)[0]
                makespan_norm.update(np.array(makespans)[idx])
                variance_norm.update(np.array(variances)[idx])
                energy_norm.update(np.array(energies)[idx])

                zm = makespan_norm.normalize(np.array(makespans)[idx])
                zv = variance_norm.normalize(np.array(variances)[idx])
                ze = energy_norm.normalize(np.array(energies)[idx])

                w_m_i = weights_batch[idx, 0]            # per-sample weights
                w_v_i = weights_batch[idx, 1]
                w_e_i = weights_batch[idx, 2]

                unassigned_fracs = np.array([obj_list[i]["unassigned_frac"] for i in idx])

                reward[idx] = -(zm * w_m_i + zv * w_v_i + ze * w_e_i) \
                            - UNASSIGNED_PENALTY_WEIGHT * unassigned_fracs

            reward_t = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            baseline = reward_t[non_terminal].mean().detach() if non_terminal.any() else torch.tensor(0.0)
            advantage = (reward_t - baseline).detach()

            loss = -(advantage * log_probs_t).mean() + KL_COEF * kls_t.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            step += 1
            print(f"    iter {it+1}/{RL_ITERS_PER_FILE}  mean_reward={reward.mean():.3f}  "
                  f"loss={loss.item():.3f}  kl={kls_t.mean().item():.4f}  "
                  f"terminal_frac={dones.mean():.2f}")

            if step % SAVE_INTERVAL_STEPS == 0:
                save_checkpoint(step, model, optimizer, loss.item(), f"rl_step_{step}.pt")

            if step % REFERENCE_UPDATE_EVERY == 0:
                reference_model.load_state_dict(model.state_dict())
                print(f"    [reference] anchor updated at step {step}")

            if step % EVAL_EVERY_N_STEPS == 0:
                evaluate_preference_following(model, scenario)

    save_checkpoint(step, model, optimizer, loss.item(), "rl_final.pt")
    print("\n=== PHASE 2 complete ===")


def main():
    scenario_files = list_scenario_files(SCENARIO_FOLDER)
    print(f"Found {len(scenario_files)} scenario file(s)")

    dummy_scenario = load_scenario_file(scenario_files[0])
    robot_feat_dim = build_robot_features(dummy_scenario).shape[-1]
    task_feat_dim = build_task_features(dummy_scenario).shape[-1]

    model = ThreeHeadedDenoiser(
        robot_feat_dim=robot_feat_dim, task_feat_dim=task_feat_dim,
        d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS,
        pref_dim=PREF_DIM, min_vel=MIN_VEL, max_vel=MAX_VEL,
        max_R=MAX_R, max_T=MAX_T,
    ).to(DEVICE)

    pretrain_optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    rl_optimizer = optim.AdamW(model.parameters(), lr=RL_LR, weight_decay=WEIGHT_DECAY)

    global_step = 0
    if RESUME_CHECKPOINT:
        global_step = load_checkpoint(RESUME_CHECKPOINT, model, rl_optimizer)
    elif PRETRAIN_CHECKPOINT_TO_LOAD:
        load_checkpoint(PRETRAIN_CHECKPOINT_TO_LOAD, model, pretrain_optimizer)
    else:
        pretrain(model, pretrain_optimizer, scenario_files)
        save_checkpoint(0, model, pretrain_optimizer, 0.0, "pretrain_final.pt")

    rl_finetune(model, rl_optimizer, scenario_files, global_step_start=global_step)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()