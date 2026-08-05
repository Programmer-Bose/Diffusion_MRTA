"""
generate_diverse_export.py
---------------------------
Generates N diverse, high-quality schedules for a given preference
vector (same stochastic sampling approach as generate_diverse.py),
then for EACH returned solution exports a CSV with per-robot:
  - assigned total payload weight (kg)
  - robot's max payload capacity (kg)
  - transition velocities between consecutive tasks (list)
  - completion time (route time, seconds)

Also prints a console summary per solution and produces a 3D route
plot (one figure per solution, tasks/depots annotated) using matplotlib.
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model2 import ThreeHeadedDenoiser, sample_assignment_with_logprob
from scenario_utils import load_scenario_file, build_robot_features, build_task_features
from objectives_and_reward import compute_objectives, decode_routes, compute_route_times

# =============================================================================
# HYPERPARAMETERS -- edit here
# =============================================================================
CHECKPOINT_PATH    = "./checkpoints/rl_step_50.pt"
SCENARIO_FILE       = "scenarios/scenario_10_50_1001.json"
PREFERENCE_VECTOR   = [0.3, 0.4, 0.3]

N_SOLUTIONS          = 1
N_CANDIDATES          = 40
DIVERSITY_MIN_DIST   = 0.1
QUALITY_TOP_FRACTION = 0.5

OUTPUT_DIR           = "./inference_outputs"
CSV_PREFIX            = "solution"
PLOT_PREFIX           = "solution"

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
SEED   = None

# =============================================================================


def load_model(robot_feat_dim, task_feat_dim, checkpoint_path):
    model = ThreeHeadedDenoiser(
        robot_feat_dim=robot_feat_dim, task_feat_dim=task_feat_dim,
        d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS,
        pref_dim=PREF_DIM, min_vel=MIN_VEL, max_vel=MAX_VEL,
        max_R=MAX_R, max_T=MAX_T,
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded checkpoint (step={ckpt.get('step', '?')})")
    return model


def capacity_aware_decode_from_sample(hard_assign_np, rank_raw, velocity, scenario):
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
            t = torch.full((1,), DIFFUSION_STEPS - 1, dtype=torch.long, device=DEVICE)
            x_t_noisy = torch.randint(0, R + 1, (1, T), device=DEVICE)  # fresh noise each call

            assign_logits, rank_raw, velocity, alpha, beta = model(
                r_feats, t_feats, pref, x_t=x_t_noisy, t=t, sample_velocity=True
            )
            hard_assign, _, _ = sample_assignment_with_logprob(assign_logits)
            hard_assign_np = hard_assign.squeeze(0).cpu().numpy()

            schedule, velocity_full, final_assign = capacity_aware_decode_from_sample(
                hard_assign_np, rank_raw, velocity, scenario
            )
            obj = compute_objectives(scenario, schedule, velocity_full)
            candidates.append({"schedule": schedule, "velocity": velocity_full, "obj": obj})

    return candidates


def zscore(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    mean, std = values.mean(), values.std() + eps
    return (values - mean) / std


def score_candidates(candidates, weights):
    feasible = [c for c in candidates if not c["obj"]["done"] and c["obj"]["num_unassigned"] == 0]
    if not feasible:
        print("[warning] No candidates were both capacity-feasible AND fully assigned; "
            "consider increasing N_CANDIDATES or checking model training.")
        return [], candidates

    w_m, w_v, w_e = weights
    zm = zscore([c["obj"]["makespan"] for c in feasible])
    zv = zscore([c["obj"]["workload_variance"] for c in feasible])
    ze = zscore([c["obj"]["total_energy"] for c in feasible])

    for i, c in enumerate(feasible):
        c["quality_score"] = w_m * zm[i] + w_v * zv[i] + w_e * ze[i]

    feasible.sort(key=lambda c: c["quality_score"])
    return feasible, [c for c in candidates if c["obj"]["done"]]


def schedule_distance(a, b):
    return (a != b).sum() / a.size


def select_diverse_top(candidates, n_solutions, min_dist, top_fraction):
    n_top = max(n_solutions, int(len(candidates) * top_fraction))
    pool = candidates[:n_top]
    selected = []
    for c in pool:
        if len(selected) >= n_solutions:
            break
        if all(schedule_distance(c["schedule"], s["schedule"]) >= min_dist for s in selected):
            selected.append(c)
    if len(selected) < n_solutions:
        for c in pool:
            if len(selected) >= n_solutions:
                break
            if c not in selected:
                selected.append(c)
    return selected


# =============================================================================
# Per-robot metrics extraction (weights, capacity, transition velocities,
# completion time) -- the data this script needs to export.
# =============================================================================
def extract_robot_metrics(scenario, schedule, velocity):
    R = scenario["R"]
    task_weight = np.array(scenario["task_weight"], dtype=np.float64)
    robot_max_payload = np.array(scenario["robot_max_payload"], dtype=np.float64)

    routes = decode_routes(schedule)
    robot_times = compute_route_times(scenario, schedule, velocity)

    rows = []
    for r in range(R):
        route_tasks = routes[r]
        assigned_weight = task_weight[route_tasks].sum() if route_tasks else 0.0
        transition_velocities = [round(float(velocity[r, t]), 3) for t in route_tasks]

        rows.append({
            "robot_id": r,
            "assigned_weight_kg": round(float(assigned_weight), 3),
            "capacity_kg": round(float(robot_max_payload[r]), 3),
            "num_tasks": len(route_tasks),
            "transition_velocities_m_s": transition_velocities,
            "completion_time_s": round(float(robot_times[r]), 3),
        })
    return rows


def export_solution_csv(scenario, candidate, sol_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{CSV_PREFIX}_{sol_idx}.csv")

    rows = extract_robot_metrics(scenario, candidate["schedule"], candidate["velocity"])

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "robot_id", "assigned_weight_kg", "capacity_kg",
            "num_tasks", "transition_velocities_m_s", "completion_time_s"
        ])
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["transition_velocities_m_s"] = str(row["transition_velocities_m_s"])
            writer.writerow(row_out)

    print(f"  [csv] saved: {filepath}")
    return filepath, rows


def print_console_summary(sol_idx, candidate, robot_rows):
    obj = candidate["obj"]
    print(f"\n--- Solution {sol_idx} (quality_score={candidate.get('quality_score', float('nan')):.3f}) ---")
    print(f"Makespan: {obj['makespan']:.2f} s | Workload Variance (energy): {obj['workload_variance']:.3f} "
          f"| Total Energy: {obj['total_energy']:.2f} Wh")
    for row in robot_rows:
        print(f"  Robot {row['robot_id']}: weight={row['assigned_weight_kg']}/{row['capacity_kg']} kg  "
              f"tasks={row['num_tasks']}  completion_time={row['completion_time_s']}s  "
              f"velocities={row['transition_velocities_m_s']}")


def plot_solution_3d(scenario, candidate, sol_idx, output_dir):
    robot_pos = np.array(scenario["robot_pos"])
    task_pos = np.array(scenario["task_pos"])
    routes = decode_routes(candidate["schedule"])
    obj = candidate["obj"]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(task_pos[:, 0], task_pos[:, 1], task_pos[:, 2],
               c="gray", marker="o", s=35, label="Tasks", alpha=0.5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_pos)))

    for r, route in enumerate(routes):
        depot = robot_pos[r]
        ax.scatter(*depot, c=[colors[r]], marker="^", s=140,
                   edgecolors="black", label=f"Robot {r} depot")
        ax.text(*depot, f"R{r}", fontsize=8, weight="bold")

        if not route:
            continue

        path = np.array([depot] + [task_pos[t] for t in route] + [depot])
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c=colors[r], linewidth=2, marker="o", markersize=4)

        for order_idx, t in enumerate(route, start=1):
            ax.text(*task_pos[t], f"{order_idx}", fontsize=7, color=colors[r])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (altitude)")
    ax.set_title(f"Solution {sol_idx} | Makespan={obj['makespan']:.1f}s | Energy={obj['total_energy']:.1f}Wh")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=7)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{PLOT_PREFIX}_{sol_idx}.png")
    plt.savefig(filepath, dpi=150)
    print(f"  [plot] saved: {filepath}")
    plt.show()
    plt.close(fig)


def main():
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    scenario = load_scenario_file(SCENARIO_FILE)
    robot_feat_dim = build_robot_features(scenario).shape[-1]
    task_feat_dim = build_task_features(scenario).shape[-1]

    model = load_model(robot_feat_dim, task_feat_dim, CHECKPOINT_PATH)

    print(f"\nGenerating {N_CANDIDATES} candidates for preference {PREFERENCE_VECTOR} "
          f"on {SCENARIO_FILE} (R={scenario['R']}, T={scenario['T']}) ...")
    candidates = generate_candidates(model, scenario, PREFERENCE_VECTOR, N_CANDIDATES)

    feasible_sorted, infeasible = score_candidates(candidates, PREFERENCE_VECTOR)
    print(f"Feasible: {len(feasible_sorted)}/{N_CANDIDATES}  (infeasible: {len(infeasible)})")

    diverse_top = select_diverse_top(feasible_sorted, N_SOLUTIONS, DIVERSITY_MIN_DIST, QUALITY_TOP_FRACTION)

    print(f"\n=== Exporting {len(diverse_top)} diverse, high-quality solutions to {OUTPUT_DIR} ===")
    for i, c in enumerate(diverse_top, start=1):
        _, robot_rows = export_solution_csv(scenario, c, i, OUTPUT_DIR)
        print_console_summary(i, c, robot_rows)
        plot_solution_3d(scenario, c, i, OUTPUT_DIR)

    return diverse_top


if __name__ == "__main__":
    main()