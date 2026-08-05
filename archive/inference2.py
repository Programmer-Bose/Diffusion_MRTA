"""
generate_diverse.py
--------------------
Generate a diverse set of high-quality schedules from a trained
ThreeHeadedDenoiser for a single user-specified preference vector.

Approach:
  1. Run N_CANDIDATES stochastic forward passes (assignment sampled via
     categorical draw from Sinkhorn probs, velocity sampled from its
     Beta distribution) -- this is where "generative diversity" comes
     from, since decode_schedule()/argmax alone always gives one
     deterministic output.
  2. Decode each candidate with the CAPACITY-AWARE decoder (so no
     candidate silently overloads a robot).
  3. Score every candidate with compute_objectives() + a preference-
     weighted, z-scored (across the candidate pool) objective distance
     -- "quality" relative to the given preference.
  4. Filter out infeasible (capacity-exceeded) candidates.
  5. De-duplicate near-identical schedules (Hamming-distance based) so
     the returned set is genuinely diverse, not N copies of the same
     good solution.
  6. Return the top N_SOLUTIONS diverse, high-quality schedules.
"""

import os
import numpy as np
import torch
import warnings

from model2 import ThreeHeadedDenoiser, sample_assignment_with_logprob
from scenario_utils import load_scenario_file, build_robot_features, build_task_features
from objectives_and_reward import compute_objectives

# =============================================================================
# HYPERPARAMETERS -- edit here
# =============================================================================
CHECKPOINT_PATH    = "checkpoints/rl_final.pt"
# CHECKPOINT_PATH    = "checkpoints/rl_step_50.pt"
SCENARIO_FILE       = "scenarios/scenario_3_15_1004.json"
PREFERENCE_VECTOR   = [0.3, 0.4, 0.3]     # [w_makespan, w_variance, w_energy], sums to 1

N_SOLUTIONS         = 1     # how many diverse solutions to return
N_CANDIDATES         = 10    # how many stochastic rollouts to generate before filtering
DIVERSITY_MIN_DIST  = 0.1  # min normalized Hamming distance between kept solutions (0-1)
QUALITY_TOP_FRACTION = 0.5  # only consider the top X% by objective quality before diversifying

D_MODEL      = 256
N_HEAD       = 8
NUM_LAYERS   = 3
PREF_DIM     = 3
MIN_VEL      = 1.0
MAX_VEL      = 5.0
MAX_R        = 20
MAX_T        = 60
DIFFUSION_STEPS = 50

DEFAULT_VELOCITY_FALLBACK = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = None   # set an int for reproducible sampling, or None for fresh randomness each run

# =============================================================================


def load_model(robot_feat_dim, task_feat_dim, checkpoint_path):
    model = ThreeHeadedDenoiser(
        robot_feat_dim=robot_feat_dim,
        task_feat_dim=task_feat_dim,
        d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS,
        pref_dim=PREF_DIM, min_vel=MIN_VEL, max_vel=MAX_VEL,
        max_R=MAX_R, max_T=MAX_T,
    ).to(DEVICE)

    # Suppress a known Transformer UserWarning triggered by PyTorch's
    # nested-tensor logic when `encoder_layer.norm_first` is True.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*enable_nested_tensor is True.*encoder_layer.norm_first.*",
    )

    # Be explicit about `weights_only` to avoid the FutureWarning about
    # implicit pickle behavior when loading checkpoints.
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded checkpoint (step={ckpt.get('step', '?')})")
    return model


def capacity_aware_decode_from_sample(hard_assign_np, rank_raw, velocity, scenario):
    """
    Builds an (R,T) schedule from a SAMPLED hard_assign (T,) vector,
    respecting robot payload capacity: processes tasks in order of
    (arbitrary, since assignment already sampled) index, only keeping
    the assignment if the robot still has capacity; otherwise the task
    is left unassigned. Mirrors decode_schedule_capacity_aware()'s
    intent but works off an already-sampled discrete assignment rather
    than re-deriving one from probabilities.
    """
    R, T = scenario["R"], scenario["T"]
    task_weight = np.array(scenario["task_weight"], dtype=np.float64)
    robot_max_payload = np.array(scenario["robot_max_payload"], dtype=np.float64)
    remaining_capacity = robot_max_payload.copy()

    rank_raw_np = rank_raw.squeeze(0).detach().cpu().numpy()
    velocity_np = velocity.squeeze(0).detach().cpu().numpy()

    final_assign = np.full(T, -1, dtype=int)
    for t_idx in range(T):
        r = hard_assign_np[t_idx]
        w = task_weight[t_idx]
        if remaining_capacity[r] >= w:
            final_assign[t_idx] = r
            remaining_capacity[r] -= w
        # else: left unassigned (-1), rather than overloading the robot

    schedule = np.zeros((R, T), dtype=int)
    velocity_out = np.zeros((R, T), dtype=float)

    for r in range(R):
        task_ids = [t for t in range(T) if final_assign[t] == r]
        if not task_ids:
            continue
        ranks = [rank_raw_np[r, t] for t in task_ids]
        order = np.argsort(ranks)
        for pos, idx in enumerate(order):
            t = task_ids[idx]
            schedule[r, t] = pos + 1
            velocity_out[r, t] = velocity_np[r, t]

    velocity_full = np.where(velocity_out > 0, velocity_out, DEFAULT_VELOCITY_FALLBACK)
    return schedule, velocity_full, final_assign


def generate_candidates(model, scenario, weights, n_candidates):
    R, T = scenario["R"], scenario["T"]

    r_feats = torch.tensor(build_robot_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_feats = torch.tensor(build_task_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pref = torch.tensor([weights], dtype=torch.float32).to(DEVICE)

    candidates = []
    with torch.no_grad():
        for _ in range(n_candidates):
            # Late diffusion timestep + random noisy state: treat this as a
            # single-shot generative draw conditioned on the preference,
            # relying on stochastic sampling (not diffusion trajectory) for
            # diversity, consistent with how the model was RL-trained.
            t = torch.full((1,), DIFFUSION_STEPS - 1, dtype=torch.long, device=DEVICE)
            x_t_noisy = torch.randint(0, R + 1, (1, T), device=DEVICE)
            # print("noisy x_t:", x_t_noisy)

            assign_logits, rank_raw, velocity, alpha, beta = model(
                r_feats, t_feats, pref, x_t=x_t_noisy, t=t, sample_velocity=True
            )

            hard_assign, _, _ = sample_assignment_with_logprob(assign_logits)
            hard_assign_np = hard_assign.squeeze(0).cpu().numpy()

            schedule, velocity_full, final_assign = capacity_aware_decode_from_sample(
                hard_assign_np, rank_raw, velocity, scenario
            )

            obj = compute_objectives(scenario, schedule, velocity_full)
            candidates.append({
                "schedule": schedule,
                "velocity": velocity_full,
                "obj": obj,
            })

    return candidates


def zscore(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    std = values.std() + eps
    return (values - mean) / std


def score_candidates(candidates, weights):
    """
    Filters infeasible candidates, then computes a preference-weighted
    quality score (lower = better) using z-scores computed ACROSS the
    feasible candidate pool -- consistent with the RL training's
    normalization approach.
    """
    feasible = [c for c in candidates if not c["obj"]["done"]]
    if not feasible:
        print("[warning] ALL candidates were capacity-infeasible; "
              "returning raw candidates unranked.")
        return candidates[:0], candidates

    w_m, w_v, w_e = weights
    makespans = [c["obj"]["makespan"] for c in feasible]
    variances = [c["obj"]["workload_variance"] for c in feasible]
    energies = [c["obj"]["total_energy"] for c in feasible]

    zm = zscore(makespans)
    zv = zscore(variances)
    ze = zscore(energies)

    for i, c in enumerate(feasible):
        c["quality_score"] = w_m * zm[i] + w_v * zv[i] + w_e * ze[i]  # lower = better

    feasible.sort(key=lambda c: c["quality_score"])
    return feasible, [c for c in candidates if c["obj"]["done"]]


def schedule_distance(sched_a, sched_b):
    """Normalized Hamming distance between two (R,T) schedules (fraction
    of cells that differ), used as the diversity metric."""
    diff = (sched_a != sched_b).sum()
    return diff / sched_a.size


def select_diverse_top(candidates, n_solutions, min_dist, top_fraction):
    """
    Greedy diverse selection: restrict to the top `top_fraction` by
    quality, then greedily pick solutions that are at least `min_dist`
    away (Hamming) from all previously selected ones, so the returned
    set isn't N near-identical copies of the single best schedule.
    """
    n_top = max(n_solutions, int(len(candidates) * top_fraction))
    pool = candidates[:n_top]

    selected = []
    for c in pool:
        if len(selected) >= n_solutions:
            break
        if all(schedule_distance(c["schedule"], s["schedule"]) >= min_dist for s in selected):
            selected.append(c)

    # If diversity constraint left us short, backfill with next-best
    # quality candidates regardless of distance (never return fewer
    # than requested, when the pool has enough feasible candidates).
    if len(selected) < n_solutions:
        for c in pool:
            if len(selected) >= n_solutions:
                break
            if c not in selected:
                selected.append(c)

    return selected


def print_solution(idx, c):
    obj = c["obj"]
    print(f"\n--- Solution {idx} (quality_score={c.get('quality_score', float('nan')):.3f}) ---")
    print(f"Makespan: {obj['makespan']:.2f}")
    print(f"Workload Variance (energy-based): {obj['workload_variance']:.3f}")
    print(f"Total Energy (Wh): {obj['total_energy']:.2f}")
    # print(f"Schedule:\n{c['schedule']}")


def main():
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    scenario = load_scenario_file(SCENARIO_FILE)
    robot_feat_dim = build_robot_features(scenario).shape[-1]
    task_feat_dim = build_task_features(scenario).shape[-1]

    model = load_model(robot_feat_dim, task_feat_dim, CHECKPOINT_PATH)

    print(f"\nGenerating {N_CANDIDATES} candidate schedules for preference {PREFERENCE_VECTOR} "
          f"on {SCENARIO_FILE} (R={scenario['R']}, T={scenario['T']}) ...")
    candidates = generate_candidates(model, scenario, PREFERENCE_VECTOR, N_CANDIDATES)

    feasible_sorted, infeasible = score_candidates(candidates, PREFERENCE_VECTOR)
    print(f"\nFeasible candidates: {len(feasible_sorted)}/{N_CANDIDATES}  "
          f"(infeasible/over-capacity: {len(infeasible)})")

    diverse_top = select_diverse_top(
        feasible_sorted, N_SOLUTIONS, DIVERSITY_MIN_DIST, QUALITY_TOP_FRACTION
    )

    print(f"\n=== Returning {len(diverse_top)} diverse, high-quality solutions ===")
    for i, c in enumerate(diverse_top, start=1):
        print_solution(i, c)

    return diverse_top


if __name__ == "__main__":
    main()