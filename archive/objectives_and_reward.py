"""
objectives_and_reward.py
-------------------------
Computes mission objectives (Makespan, Workload Variance, Total Energy)
from a decoded schedule + velocity matrix, using the physically grounded
F450EnergyModel for energy/SOC calculation. Also implements the
reference-point (achievement scalarization) reward function validated
in Test 3.

BUGFIX (this version): earlier versions summed a robot's assigned task
weights with NO cap against that robot's `robot_max_payload`. If the
schedule (random init, or an undertrained model) assigns more payload
than a robot can physically lift, `initial_payload_kg` blows up (e.g.
35 kg on a 1.5 kg-class frame), hover power scales as mass^1.5, and the
battery reads several hundred percent SOC drop -- not a battery-size
problem, a physics/feasibility problem. We now:
  1. Clip the payload actually carried to `robot_max_payload` (excess
     tasks still consume time/service but cannot exceed the lift limit
     used for energy -- this keeps energy numbers physically sane even
     for infeasible schedules instead of exploding).
  2. Report a `capacity_exceeded` flag/array per robot so training can
     penalize infeasible over-capacity assignments directly, rather
     than only implicitly via a huge SOC number.

Expected scenario fields (in addition to existing ones):
  scenario["robot_own_weight"]        -- kg, frame w/o payload
  scenario["robot_max_payload"]       -- kg, max liftable payload per robot
  scenario["robot_battery_capacity"]  -- mAh, one of (8000,12000,16000,20000)
  scenario["robot_battery"]           -- % state of charge at mission start
  scenario["task_weight"]             -- kg, per task
"""

import numpy as np
import torch

from archive.f450_energy2 import F450EnergyModel

# Fallback pack size if a scenario predates the variable-battery field.
_DEFAULT_BATTERY_CAPACITY_mAh = 8000.0


# ============================================================
# Build per-robot route coordinates (depot -> tasks -> depot)
# from a decoded schedule + velocity matrix
# ============================================================
def decode_routes(schedule):
    """Returns list of task-index lists, one per robot, in execution order."""
    R, T = schedule.shape
    routes = []
    for r in range(R):
        row = schedule[r]
        assigned = [(order, t) for t, order in enumerate(row) if order > 0]
        assigned.sort(key=lambda x: x[0])
        routes.append([t for _, t in assigned])
    return routes


def build_route_tensors(scenario, schedule, velocity):
    """
    Builds per-robot tensors needed for F450EnergyModel.compute_batch_with_soc:
      completed_coords     : [R, max_len+1, 3]
      speeds               : [R, max_len]
      service_times        : [R, max_len]
      payload_drop_kg       : [R, max_len]   -- CLIPPED so a robot's total
                                                 carried payload never exceeds
                                                 its robot_max_payload
      robot_own_weight_kg   : [R]
      initial_payload_kg    : [R]             -- clipped total (physically
                                                 carried), used for energy
      raw_assigned_weight_kg: [R]             -- UNclipped sum of assigned
                                                 task weights, for reporting
      battery_capacity_mAh  : [R]
      capacity_exceeded      : [R] bool        -- True if raw assigned weight
                                                 > robot_max_payload
      valid_mask            : [R, max_len]
    """
    R = scenario["R"]
    robot_pos = np.array(scenario["robot_pos"], dtype=np.float32)
    task_pos = np.array(scenario["task_pos"], dtype=np.float32)
    task_service_time = np.array(scenario["task_service_time"], dtype=np.float32)
    task_weight = np.array(scenario["task_weight"], dtype=np.float32)

    robot_own_weight_kg = np.array(scenario["robot_own_weight"], dtype=np.float32)
    robot_max_payload = np.array(
        scenario.get("robot_max_payload", [np.inf] * R), dtype=np.float32
    )
    battery_capacity_mAh = np.array(
        scenario.get("robot_battery_capacity", [_DEFAULT_BATTERY_CAPACITY_mAh] * R),
        dtype=np.float32,
    )

    routes = decode_routes(schedule)
    max_len = max(1, max(len(r) for r in routes)) + 1

    completed_coords = np.zeros((R, max_len + 1, 3), dtype=np.float32)
    speeds_norm = np.ones((R, max_len), dtype=np.float32)
    service_times = np.zeros((R, max_len), dtype=np.float32)
    payload_drop_kg = np.zeros((R, max_len), dtype=np.float32)
    valid_mask = np.zeros((R, max_len), dtype=np.float32)
    initial_payload_kg = np.zeros(R, dtype=np.float32)
    raw_assigned_weight_kg = np.zeros(R, dtype=np.float32)
    capacity_exceeded = np.zeros(R, dtype=bool)

    for r in range(R):
        depot = robot_pos[r]
        route_tasks = routes[r]
        n_legs = len(route_tasks)

        raw_total = task_weight[route_tasks].sum() if n_legs > 0 else 0.0
        raw_assigned_weight_kg[r] = raw_total

        cap = robot_max_payload[r]
        if raw_total > cap:
            capacity_exceeded[r] = True
            # Scale every task's contribution down proportionally so the
            # ENERGY MODEL never sees more mass than the frame can lift,
            # while still keeping relative per-task weight ratios and
            # still charging full service time for every task visited
            # (the infeasibility itself is penalized via capacity_exceeded,
            # not silently hidden in the energy number).
            scale = cap / raw_total if raw_total > 0 else 1.0
        else:
            scale = 1.0

        carried_total = 0.0
        completed_coords[r, 0, :] = depot

        for i, t in enumerate(route_tasks):
            completed_coords[r, i + 1, :] = task_pos[t]

            v_raw = velocity[r, t]
            speeds_norm[r, i] = v_raw / F450EnergyModel.SPEED_SCALE
            service_times[r, i] = task_service_time[t]

            dropped = task_weight[t] * scale
            payload_drop_kg[r, i] = dropped
            carried_total += dropped
            valid_mask[r, i] = 1.0

        initial_payload_kg[r] = carried_total

        completed_coords[r, n_legs + 1, :] = depot
        if n_legs < max_len:
            speeds_norm[r, n_legs] = 1.0
            service_times[r, n_legs] = 0.0
            payload_drop_kg[r, n_legs] = 0.0
            valid_mask[r, n_legs] = 1.0

        for i in range(n_legs + 1, max_len):
            completed_coords[r, i + 1, :] = depot

    return (
        torch.tensor(completed_coords),
        torch.tensor(speeds_norm),
        torch.tensor(service_times),
        torch.tensor(payload_drop_kg),
        torch.tensor(robot_own_weight_kg),
        torch.tensor(initial_payload_kg),
        torch.tensor(battery_capacity_mAh),
        torch.tensor(valid_mask),
        raw_assigned_weight_kg,
        capacity_exceeded,
    )


# ============================================================
# Compute per-robot travel+service TIME (for makespan / variance)
# ============================================================
def compute_route_times(scenario, schedule, velocity):
    R = scenario["R"]
    robot_pos = np.array(scenario["robot_pos"], dtype=np.float32)
    task_pos = np.array(scenario["task_pos"], dtype=np.float32)
    task_service_time = np.array(scenario["task_service_time"], dtype=np.float32)

    routes = decode_routes(schedule)
    robot_times = np.zeros(R, dtype=np.float32)

    for r in range(R):
        current_pos = robot_pos[r]
        total_time = 0.0

        for t in routes[r]:
            target_pos = task_pos[t]
            dist_m = np.linalg.norm(target_pos - current_pos) * F450EnergyModel.SPACE_SCALE
            v_ms = max(velocity[r, t], 1e-3)
            travel_time_s = dist_m / v_ms

            total_time += travel_time_s + task_service_time[t]
            current_pos = target_pos

        dist_home_m = np.linalg.norm(robot_pos[r] - current_pos) * F450EnergyModel.SPACE_SCALE
        avg_speed = velocity[r][schedule[r] > 0].mean() if np.any(schedule[r] > 0) else 5.0
        total_time += dist_home_m / max(avg_speed, 1e-3)

        robot_times[r] = total_time

    return robot_times


# ============================================================
# Full objective computation
# ============================================================
def compute_objectives(scenario, schedule, velocity):
    """
    Returns dict with makespan, workload_variance, total_energy,
    plus per-robot breakdowns, including capacity_exceeded so infeasible
    over-capacity assignments can be penalized in training.
    """
    robot_times = compute_route_times(scenario, schedule, velocity)

    (
        completed_coords,
        speeds_norm,
        service_times,
        payload_drop_kg,
        robot_own_weight_kg,
        initial_payload_kg,
        battery_capacity_mAh,
        valid_mask,
        raw_assigned_weight_kg,
        capacity_exceeded,
    ) = build_route_tensors(scenario, schedule, velocity)

    total_Wh, soc_drop = F450EnergyModel.compute_batch_with_soc(
        completed_coords,
        speeds_norm,
        service_times,
        payload_drop_kg,
        robot_own_weight_kg,
        initial_payload_kg,
        battery_capacity_mAh,
    )

    total_Wh_np = total_Wh.detach().cpu().numpy()

    makespan = float(robot_times.max())
    workload_variance = float(robot_times.var())
    total_energy = float(total_Wh_np.sum())

    battery_exceeded = F450EnergyModel.battery_exceeded(soc_drop).any().item()

    initial_battery_pct = np.array(scenario["robot_battery"], dtype=np.float32)
    soc_drop_np = soc_drop.detach().cpu().numpy()
    battery_left = np.clip(initial_battery_pct - soc_drop_np, 0.0, 100.0)

    return {
        "makespan": makespan,
        "workload_variance": workload_variance,
        "total_energy": total_energy,
        "robot_times": robot_times,
        "robot_energy_Wh": total_Wh_np,
        "robot_soc_drop": soc_drop_np,
        "battery_exceeded": battery_exceeded,
        "battery_left": battery_left,
        "raw_assigned_weight_kg": raw_assigned_weight_kg,
        "capacity_exceeded": capacity_exceeded,        # per-robot bool array
        "any_capacity_exceeded": bool(capacity_exceeded.any()),
    }


# ============================================================
# Reward function
# ============================================================
def zscore_batch(values, eps=1e-8):
    values = np.array(values, dtype=np.float64)
    mean = values.mean()
    std = values.std() + eps
    return (values - mean) / std, mean, std


def compute_rewards(
    objective_batch,
    reference_point,
    rho=0.05,
    capacity_penalty_weight=1.0,
    battery_penalty_weight=2.0,
):
    """
    objective_batch: dict with lists 'makespan', 'workload_variance',
                     'total_energy', and optionally:
                       'any_capacity_exceeded' -- list of bools
                       'max_soc_drop'          -- list of floats, the WORST
                                                   (max) per-robot SOC drop %
                                                   in each schedule (pass
                                                   obj['robot_soc_drop'].max()
                                                   from compute_objectives).
    reference_point: [target_makespan, target_variance, target_energy]
    capacity_penalty_weight: fixed penalty subtracted from reward for any
                     schedule that overloaded a robot's payload capacity.
    battery_penalty_weight: penalty subtracted from reward, PROPORTIONAL
                     to how far the worst robot's SOC drop exceeds 100%
                     (e.g. 150% drop -> penalty of 0.5 * battery_penalty_weight,
                     300% drop -> 2.0 * battery_penalty_weight). A robot that
                     runs out of battery mid-mission is a mission failure,
                     not just a "bad" objective value, so this is applied
                     on top of -- not instead of -- the energy term.
    """
    names = ["makespan", "workload_variance", "total_energy"]
    z_scores = {}
    stats = {}
    for name in names:
        z, mean, std = zscore_batch(objective_batch[name])
        z_scores[name] = z
        stats[name] = (mean, std)

    ref_z = {}
    for name, target in zip(names, reference_point):
        mean, std = stats[name]
        ref_z[name] = (target - mean) / std

    n = len(objective_batch["makespan"])
    penalty = np.zeros(n)
    aug = np.zeros(n)

    for name in names:
        diff = z_scores[name] - ref_z[name]
        penalty += np.maximum(diff, 0.0) ** 2
        aug += diff

    dist = np.sqrt(penalty) + rho * aug
    reward = -dist

    if "any_capacity_exceeded" in objective_batch:
        exceeded = np.array(objective_batch["any_capacity_exceeded"], dtype=bool)
        reward = reward - capacity_penalty_weight * exceeded.astype(np.float64)

    if "max_soc_drop" in objective_batch:
        max_soc_drop = np.array(objective_batch["max_soc_drop"], dtype=np.float64)
        # Only penalize the portion PAST 100% (a robot that used 90% of its
        # battery is fine; one that would need 300% is a hard mission failure).
        over_100_frac = np.maximum(max_soc_drop - 100.0, 0.0) / 100.0
        reward = reward - battery_penalty_weight * over_100_frac

    return reward, z_scores["makespan"], z_scores["workload_variance"], z_scores["total_energy"]


# ============================================================
# Quick self-test
# ============================================================
if __name__ == "__main__":
    import json

    with open("scenarios/scenario_10_50_2001.json", "r") as f:
        scenario = json.load(f)

    R, T = scenario["R"], scenario["T"]

    schedule = np.zeros((R, T), dtype=int)
    velocity = np.full((R, T), 5.0, dtype=float)
    slot = np.ones(R, dtype=int)
    for t in range(T):
        r = t % R
        schedule[r, t] = slot[r]
        slot[r] += 1

    obj = compute_objectives(scenario, schedule, velocity)
    print(f"Makespan: {obj['makespan']:.2f}")
    print(f"Workload Variance: {obj['workload_variance']:.2f}")
    print(f"Total Energy (Wh): {obj['total_energy']:.2f}")
    print(f"Per-robot energy (Wh): {obj['robot_energy_Wh']}")
    print(f"Battery exceeded anywhere: {obj['battery_exceeded']}")
    print(f"Battery left per robot (%): {obj['battery_left']}")
    print(f"Raw assigned weight per robot (kg): {obj['raw_assigned_weight_kg']}")
    print(f"Capacity exceeded per robot: {obj['capacity_exceeded']}")

    batch = {
        "makespan": [obj["makespan"]] * 5,
        "workload_variance": [obj["workload_variance"]] * 5,
        "total_energy": [obj["total_energy"]] * 5,
        "any_capacity_exceeded": [obj["any_capacity_exceeded"]] * 5,
    }
    reward, zm, zv, ze = compute_rewards(batch, reference_point=[30, 20, 9])
    print(f"\nReward (identical batch, sanity only): {reward}")