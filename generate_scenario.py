import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import json


def generate_scenario(R, T, seed=42):
    """
    Generate a 3D heterogeneous UAV multi-task assignment scenario.

    Arena: [0, 1]^3

    R : number of robots (UAVs)
    T : number of tasks
    seed : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # ---------------- Depot positions: 4 corners of arena, round-robin ----------------
    corners = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    corner_idx = np.arange(R) % 4
    depot_xy = corners[corner_idx]
    depot_z = rng.uniform(0.0, 0.1, size=R)
    robot_pos = np.hstack([depot_xy, depot_z.reshape(-1, 1)])  # (R,3)

    # ---------------- Battery levels (85 - 100 %) ----------------
    battery = rng.uniform(85.0, 100.0, size=R)

    # ---------------- Heterogeneous categories ----------------
    # Payload capacity: 10, 15, 20, 25 kg  <->  Own weight: 5, 7.5, 10, 12.5 kg
    capacity_options = np.array([5.0, 10.0, 20.0, 25.0])
    own_weight_options = np.array([5.0, 7.5, 10.0, 12.5])

    cat_idx = rng.integers(0, 4, size=R)
    robot_max_payload = capacity_options[cat_idx]
    robot_own_weight = own_weight_options[cat_idx]
    robot_category = cat_idx + 1  # 1..4 for readability

    # ---------------- Task generation ----------------
    # Each task must be >= 0.2 units away (XY-plane) from the previous task,
    # within arena boundary. Z limited to [0.3, 0.8].
    min_dist = 0.2
    max_attempts_per_task = 5000

    task_pos = np.zeros((T, 3))
    task_pos[0, 0:2] = rng.uniform(0.0, 1.0, size=2)
    task_pos[0, 2] = rng.uniform(0.3, 0.8)

    for i in range(1, T):
        placed = False
        for _ in range(max_attempts_per_task):
            candidate_xy = rng.uniform(0.0, 1.0, size=2)
            dist = np.linalg.norm(candidate_xy - task_pos[i - 1, 0:2])
            if dist >= min_dist:
                task_pos[i, 0:2] = candidate_xy
                task_pos[i, 2] = rng.uniform(0.3, 0.8)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place task {i} with min spacing {min_dist} "
                f"after {max_attempts_per_task} attempts. Reduce T or min_dist."
            )

    # Service time: payload dropping time (seconds)
    task_service_time = rng.uniform(2.0, 5.0, size=T)

# ---------------- Task payload categories: 6 types (1-6 kg) ----------------
    # Service time scales with payload category: 6 levels from 4s to 10s
    weight_categories = np.array([1, 2, 3, 4, 5, 6])          # kg
    service_time_categories = np.linspace(4.0, 10.0, 6)        # sec, level-matched

    # ---------------- Task weights with feasibility check ----------------
    total_capacity = float(np.sum(robot_max_payload))
    max_attempts = 1000
    feasible = False
    task_weight = None
    task_service_time = None

    for _ in range(max_attempts):
        cat_choice = rng.integers(0, 6, size=T)          # pick category index 0-5 per task
        candidate_weight = weight_categories[cat_choice]
        if np.sum(candidate_weight) < total_capacity-5:
            task_weight = candidate_weight.astype(float)
            task_service_time = service_time_categories[cat_choice]
            feasible = True
            break

    if not feasible:
        raise RuntimeError(
            f"Infeasible scenario: total task demand exceeds total fleet capacity "
            f"even after {max_attempts} attempts. Increase R or reduce T."
        )

    total_demand = float(np.sum(task_weight))

    scenario = {
        "R": R,
        "T": T,
        "seed": seed,
        "robot_pos": robot_pos.tolist(),
        "robot_battery": battery.tolist(),
        "robot_category": robot_category.tolist(),
        "robot_max_payload": robot_max_payload.tolist(),
        "robot_own_weight": robot_own_weight.tolist(),
        "task_pos": task_pos.tolist(),
        "task_service_time": task_service_time.tolist(),
        "task_weight": task_weight.tolist(),
        "total_capacity": total_capacity,
        "total_demand": total_demand,
    }

    return scenario


def save_scenario(scenario, filepath="scenario.json"):
    with open(filepath, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"Scenario saved to {filepath}")


def load_scenario(filepath="scenario.json"):
    with open(filepath, "r") as f:
        scenario = json.load(f)
    print(f"Scenario loaded from {filepath}")
    return scenario


def plot_scenario(scenario):
    robot_pos = np.array(scenario["robot_pos"])
    task_pos = np.array(scenario["task_pos"])
    robot_category = np.array(scenario["robot_category"])
    robot_max_payload = np.array(scenario["robot_max_payload"])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 2],
               c="blue", s=120, marker="^", label="Robots (UAVs)")
    ax.scatter(task_pos[:, 0], task_pos[:, 1], task_pos[:, 2],
               c="red", s=80, marker="o", label="Tasks")

    for i in range(len(robot_pos)):
        ax.text(robot_pos[i, 0], robot_pos[i, 1], robot_pos[i, 2] + 0.02,
                f"R{i+1}\n{robot_max_payload[i]:.0f}kg", color="blue", fontsize=8)

    for i in range(len(task_pos)):
        ax.text(task_pos[i, 0], task_pos[i, 1], task_pos[i, 2] + 0.02,
                f"T{i+1}", color="red", fontsize=8)

    # Draw task sequence path (for visualization of spacing order)
    ax.plot(task_pos[:, 0], task_pos[:, 1], task_pos[:, 2],
            c="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D UAV Multi-Task Scenario (Heterogeneous Fleet)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    R = 10
    T = 50
    SEEDS = [2001, 2002, 2003, 2004, 2005, 2006]
    for seed in SEEDS:
        scenario = generate_scenario(R, T, seed=seed)

        print(f"Total fleet capacity: {scenario['total_capacity']:.2f} kg")
        print(f"Total task demand:    {scenario['total_demand']:.2f} kg")

        save_scenario(scenario, f"scenarios/scenario_{R}_{T}_{seed}.json")

        # To regenerate later:
        # scenario = load_scenario("scenario.json")

        plot_scenario(scenario)

