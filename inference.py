import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_utils import load_scenario_file, build_robot_features, build_task_features
from model import ThreeHeadedDenoiser, decode_schedule_capacity_aware
from objectives_and_reward import compute_objectives


def load_checkpoint_state_dict(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys: {incompatible.missing_keys[:10]}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys: {incompatible.unexpected_keys[:10]}")


def run_inference(scenario_path, model_checkpoint=None, device="cpu", preference=None):
    print(f"\n==========================================")
    print(f"Loading Scenario: {scenario_path}")
    print(f"==========================================")
    
    scenario = load_scenario_file(scenario_path)
    R = scenario["R"]
    T = scenario["T"]
    
    task_weight = np.array(scenario["task_weight"])
    robot_max_payload = np.array(scenario["robot_max_payload"])
    
    robot_feats = torch.tensor(build_robot_features(scenario), dtype=torch.float32).unsqueeze(0).to(device)
    task_feats = torch.tensor(build_task_features(scenario), dtype=torch.float32).unsqueeze(0).to(device)

    if preference is None:
        preference = [0.33, 0.33, 0.34]
    pref = torch.tensor([preference], dtype=torch.float32).to(device)
    print(f"Preference vector: {preference}")

    robot_feat_dim = robot_feats.shape[-1]
    task_feat_dim = task_feats.shape[-1]

    # Match the architecture used in train.py
    model = ThreeHeadedDenoiser(
        robot_feat_dim,
        task_feat_dim,
        d_model=128,
        nhead=8,
        num_layers=3,
        pref_dim=3,
        min_vel=1.0,
        max_vel=15.0,
        max_R=max(20, R),
        max_T=max(60, T),
    ).to(device)

    if model_checkpoint and os.path.exists(model_checkpoint):
        load_checkpoint_state_dict(model, model_checkpoint, device)
    else:
        print("Warning: No checkpoint provided or found. Running inference with randomly initialized model weights.")

    model.eval()

    with torch.no_grad():
        assign_logits, rank_raw, velocity, alpha, beta = model(robot_feats, task_feats, pref)

    schedule, velocity_out, hard_assign, unassigned = decode_schedule_capacity_aware(
        assign_logits, rank_raw, velocity, task_weight, robot_max_payload
    )

    print("\n------------------------------------------")
    print("MISSION ALLOCATION & VELOCITY RESULTS")
    print("------------------------------------------")
    if len(unassigned) > 0:
        print(f"⚠️ Unassigned Tasks (due to capacity constraints): {unassigned}")
    else:
        print("✅ All tasks successfully assigned!")

    objectives = compute_objectives(scenario, schedule, velocity_out)
    print("\nObjective values:")
    print(f"  Makespan: {objectives['makespan']:.4f}")
    print(f"  Workload Variance: {objectives['workload_variance']:.4f}")
    print(f"  Total Energy (Wh): {objectives['total_energy']:.4f}")

    for r in range(R):
        print(f"\nRobot {r} (Max Payload: {robot_max_payload[r]} kg):")
        r_tasks = [(t, schedule[r, t], velocity_out[r, t]) for t in range(T) if schedule[r, t] > 0]
        r_tasks.sort(key=lambda x: x[1])

        if not r_tasks:
            print("  [No tasks assigned]")
            continue

        for seq_idx, (t_id, step, vel) in enumerate(r_tasks, 1):
            w = task_weight[t_id]
            print(f"  Step {seq_idx}: Task {t_id} (Weight: {w:.2f} kg) | Travel Velocity to Task: {vel:.2f} m/s")

    plot_routes(scenario, schedule, velocity_out, hard_assign)


def plot_routes(scenario, schedule, velocity_out, hard_assign):
    """
    Plots the 3D positions of robots and tasks, connecting assigned tasks
    in their execution order per robot.
    """
    robot_pos = np.array(scenario["robot_pos"], dtype=float)
    task_pos = np.array(scenario["task_pos"], dtype=float)
    R = scenario["R"]
    T = scenario["T"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot task locations
    ax.scatter(
        task_pos[:, 0],
        task_pos[:, 1],
        task_pos[:, 2],
        c="dodgerblue",
        s=80,
        marker="o",
        label="Tasks",
        zorder=3,
    )
    for t_idx in range(T):
        ax.text(
            task_pos[t_idx, 0] + 0.02,
            task_pos[t_idx, 1] + 0.02,
            task_pos[t_idx, 2] + 0.02,
            f"T{t_idx}",
            fontsize=9,
            color="blue",
        )

    # Plot robot start positions
    colors = plt.cm.rainbow(np.linspace(0, 1, R))

    ax.scatter(
        robot_pos[:, 0],
        robot_pos[:, 1],
        robot_pos[:, 2],
        c="black",
        s=120,
        marker="s",
        label="Robot Depots",
        zorder=4,
    )
    for r_idx in range(R):
        ax.text(
            robot_pos[r_idx, 0] + 0.02,
            robot_pos[r_idx, 1] + 0.02,
            robot_pos[r_idx, 2] + 0.02,
            f"R{r_idx}",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # Plot routes per robot
    for r_idx in range(R):
        r_color = colors[r_idx]

        assigned_tasks = []
        for t_idx in range(T):
            step = schedule[r_idx, t_idx]
            if step > 0:
                assigned_tasks.append((step, t_idx))
        assigned_tasks.sort(key=lambda x: x[0])

        if not assigned_tasks:
            continue

        path_coords = [robot_pos[r_idx]]
        for _, t_idx in assigned_tasks:
            path_coords.append(task_pos[t_idx])

        path_coords = np.array(path_coords)

        ax.plot(
            path_coords[:, 0],
            path_coords[:, 1],
            path_coords[:, 2],
            color=r_color,
            linewidth=2,
            linestyle="-",
            marker=">",
            label=f"Robot {r_idx} Route",
            zorder=2,
        )

        for idx, (_, t_idx) in enumerate(assigned_tasks):
            p_start = path_coords[idx]
            p_end = path_coords[idx + 1]
            vel = velocity_out[r_idx, t_idx]

            mid = (p_start + p_end) / 2
            ax.text(
                mid[0],
                mid[1],
                mid[2],
                f"v={vel:.1f}",
                fontsize=8,
                color=r_color,
                backgroundcolor="white",
                alpha=0.8,
            )

    ax.set_title("Multi-UAV Planned Routes & Task Sequences", fontsize=12, fontweight="bold")
    ax.set_xlabel("X Position", fontsize=10)
    ax.set_ylabel("Y Position", fontsize=10)
    ax.set_zlabel("Z Position", fontsize=10)
    ax.view_init(elev=25, azim=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage: point to any valid scenario json file in your scenarios directory
    sample_scenario = "scenarios/scenario_3_15_1001.json"  # Adjust path/filename as needed
    checkpoint_path = "checkpoints/model_epoch_100.pt"           # Set to None if you want to test without training weights
    sample_preference = [0.8, 0.1, 0.1]
    
    if os.path.exists(sample_scenario):
        run_inference(
            sample_scenario,
            model_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None,
            preference=sample_preference,
        )
    else:
        print(f"Scenario file not found at '{sample_scenario}'. Please verify the path.")