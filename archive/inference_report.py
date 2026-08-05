"""
inference_report.py
--------------------
Runs the trained ThreeHeadedDenoiser for a given preference vector,
decodes a (capacity-aware) schedule, and produces:

  1. CSV export -- per-robot: assigned task weights, robot payload
     capacity, transition velocities between tasks, completion time.
  2. Console summary of the same metrics.
  3. Annotated 3D Matplotlib plot of each robot's route.
"""

import os
import csv
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model2 import ThreeHeadedDenoiser, sample_assignment_with_logprob
from scenario_utils import load_scenario_file, build_robot_features, build_task_features
from objectives_and_reward import compute_objectives, decode_routes, compute_route_times

# =============================================================================
# HYPERPARAMETERS -- edit here
# =============================================================================
CHECKPOINT_PATH   = "./checkpoints/rl_step_300.pt"
SCENARIO_FILE      = "scenarios/scenario_3_15_1001.json"
PREFERENCE_VECTOR  = [0.8, 0.1, 0.1]   # [w_makespan, w_variance, w_energy], sums to 1

OUTPUT_DIR         = "./inference_outputs"
CSV_FILENAME        = "robot_report.csv"
PLOT_FILENAME        = "routes_3d.png"

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
SPACE_SCALE = 100.0   # must match compute_route_times default

# =============================================================================


def load_model(robot_feat_dim, task_feat_dim, checkpoint_path):
    model = ThreeHeadedDenoiser(
        robot_feat_dim=robot_feat_dim, task_feat_dim=task_feat_dim,
        d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS,
        pref_dim=PREF_DIM, min_vel=MIN_VEL, max_vel=MAX_VEL,
        max_R=MAX_R, max_T=MAX_T,
    ).to(DEVICE)

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*enable_nested_tensor is True.*encoder_layer.norm_first.*",
    )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
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
    return schedule, velocity_full


def generate_schedule(model, scenario, weights):
    R, T = scenario["R"], scenario["T"]
    r_feats = torch.tensor(build_robot_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    t_feats = torch.tensor(build_task_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pref = torch.tensor([weights], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        t = torch.full((1,), DIFFUSION_STEPS - 1, dtype=torch.long, device=DEVICE)
        x_t_noisy = torch.randint(0, R + 1, (1, T), device=DEVICE)
        assign_logits, rank_raw, velocity, alpha, beta = model(
            r_feats, t_feats, pref, x_t=x_t_noisy, t=t, sample_velocity=False  # deterministic mean for reporting
        )
        hard_assign, _, _ = sample_assignment_with_logprob(assign_logits)
        hard_assign_np = hard_assign.squeeze(0).cpu().numpy()

    schedule, velocity_full = capacity_aware_decode_from_sample(hard_assign_np, rank_raw, velocity, scenario)
    return schedule, velocity_full


# =============================================================================
# Build per-robot report rows: weights carried, capacity, leg velocities,
# completion time
# =============================================================================
def build_robot_report(scenario, schedule, velocity):
    R, T = scenario["R"], scenario["T"]
    task_weight = np.array(scenario["task_weight"], dtype=np.float64)
    robot_max_payload = np.array(scenario["robot_max_payload"], dtype=np.float64)

    routes = decode_routes(schedule)
    robot_times = compute_route_times(scenario, schedule, velocity, space_scale=SPACE_SCALE)

    rows = []
    for r in range(R):
        route_tasks = routes[r]
        assigned_weights = [round(float(task_weight[t]), 3) for t in route_tasks]
        total_weight = round(float(sum(assigned_weights)), 3)

        leg_velocities = [round(float(velocity[r, t]), 3) for t in route_tasks]

        rows.append({
            "robot_id": r,
            "capacity_kg": round(float(robot_max_payload[r]), 3),
            "n_tasks_assigned": len(route_tasks),
            "assigned_task_ids": route_tasks,
            "assigned_task_weights_kg": assigned_weights,
            "total_carried_weight_kg": total_weight,
            "transition_velocities_m_s": leg_velocities,
            "completion_time_s": round(float(robot_times[r]), 3),
        })

    return rows, routes


def export_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "robot_id", "capacity_kg", "n_tasks_assigned",
        "assigned_task_ids", "assigned_task_weights_kg",
        "total_carried_weight_kg", "transition_velocities_m_s",
        "completion_time_s",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["assigned_task_ids"] = ";".join(map(str, row["assigned_task_ids"]))
            out["assigned_task_weights_kg"] = ";".join(map(str, row["assigned_task_weights_kg"]))
            out["transition_velocities_m_s"] = ";".join(map(str, row["transition_velocities_m_s"]))
            writer.writerow(out)
    print(f"\n[csv] saved: {output_path}")


def print_summary(rows, obj):
    print("\n=== Per-Robot Summary ===")
    for row in rows:
        print(f"Robot {row['robot_id']}: "
              f"capacity={row['capacity_kg']}kg  "
              f"carried={row['total_carried_weight_kg']}kg  "
              f"tasks={row['n_tasks_assigned']}  "
              f"completion_time={row['completion_time_s']}s")
        print(f"    task_ids={row['assigned_task_ids']}")
        print(f"    task_weights={row['assigned_task_weights_kg']}")
        print(f"    transition_velocities={row['transition_velocities_m_s']}")

    print("\n=== Mission-Level Summary ===")
    print(f"Makespan: {obj['makespan']:.2f} s")
    print(f"Workload Variance (energy-based): {obj['workload_variance']:.3f}")
    print(f"Total Energy: {obj['total_energy']:.2f} Wh")
    print(f"Capacity exceeded (any robot): {obj['any_capacity_exceeded']}")


def plot_routes_3d(scenario, schedule, routes, rows, output_path):
    robot_pos = np.array(scenario["robot_pos"])
    task_pos = np.array(scenario["task_pos"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(task_pos[:, 0], task_pos[:, 1], task_pos[:, 2],
               c="gray", marker="o", s=35, alpha=0.5, label="Tasks")

    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_pos)))

    for r, route in enumerate(routes):
        depot = robot_pos[r]
        ax.scatter(*depot, c=[colors[r]], marker="^", s=160,
                   edgecolors="black", label=f"Robot {r}")

        if not route:
            continue

        path = np.array([depot] + [task_pos[t] for t in route] + [depot])
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c=colors[r], linewidth=2, marker="o", markersize=4)

        row = rows[r]
        for order_idx, t in enumerate(route):
            v = row["transition_velocities_m_s"][order_idx]
            ax.text(*task_pos[t], f"#{order_idx+1}\nv={v}m/s", fontsize=7, color=colors[r])

        # annotate completion time near the depot
        ax.text(*depot, f"  T={row['completion_time_s']}s", fontsize=8,
                 color=colors[r], weight="bold")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (altitude)")
    ax.set_title("UAV Mission Routes (annotated: visit order, leg velocity, completion time)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"[plot] saved: {output_path}")
    plt.show()


def main():
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    scenario = load_scenario_file(SCENARIO_FILE)
    robot_feat_dim = build_robot_features(scenario).shape[-1]
    task_feat_dim = build_task_features(scenario).shape[-1]

    model = load_model(robot_feat_dim, task_feat_dim, CHECKPOINT_PATH)

    print(f"\nGenerating schedule for preference {PREFERENCE_VECTOR} on {SCENARIO_FILE} "
          f"(R={scenario['R']}, T={scenario['T']}) ...")
    schedule, velocity = generate_schedule(model, scenario, PREFERENCE_VECTOR)
    obj = compute_objectives(scenario, schedule, velocity)

    rows, routes = build_robot_report(scenario, schedule, velocity)

    print_summary(rows, obj)

    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    export_csv(rows, csv_path)

    plot_path = os.path.join(OUTPUT_DIR, PLOT_FILENAME)
    plot_routes_3d(scenario, schedule, routes, rows, plot_path)


if __name__ == "__main__":
    main()