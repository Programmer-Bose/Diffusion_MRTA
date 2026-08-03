"""
objectives_and_reward.py
-------------------------
Computes mission objectives (Makespan, Workload Variance, Total Energy)
from a decoded schedule + velocity matrix, using the generalized
GeneralUAVEnergyModel (gen_uav_em.py) for energy calculation. Also
implements the reference-point (achievement scalarization) reward
function, now with a SEVERITY-SCALED, DOMINANT capacity-overload
penalty and an episode-termination signal.

CHANGE LOG (this version):
  1. Swapped F450EnergyModel -> GeneralUAVEnergyModel (gen_uav_em.py).
     Per-robot k_thrust / k_drag / motor_eff are now required scenario
     (or override) fields instead of a single hardcoded F450 baseline.
  2. Capacity-overload penalty is no longer a flat constant. It now
     scales with HOW FAR a robot's raw assigned weight exceeds its
     robot_max_payload, and the weight is large enough to dominate the
     achievement-scalarization distance term -- overloading a robot
     should never be "worth it" for a better makespan/energy score.
  3. compute_objectives() now returns "done": True whenever ANY robot
     is over capacity. This is a terminal-episode signal for RL
     training -- an over-capacity assignment is a physically invalid
     mission, not just a bad objective value, so the episode should
     end immediately rather than let the agent keep building on an
     infeasible schedule.
  4. compute_rewards() now branches on "done": when a schedule in the
     batch is terminal (over-capacity), the normal makespan/variance/
     energy scalarization is SKIPPED for that sample (those numbers
     are meaningless for an infeasible schedule) and replaced with a
     single large negative terminal reward.

Expected scenario fields:
  scenario["robot_own_weight"]        -- kg, frame w/o payload
  scenario["robot_max_payload"]       -- kg, max liftable payload per robot
  scenario["robot_battery_capacity"]  -- mAh (optional, for SOC reporting)
  scenario["robot_battery"]           -- % state of charge at mission start
  scenario["task_weight"]             -- kg, per task
  scenario["k_thrust"], ["k_drag"], ["motor_eff"]  -- per-robot, OPTIONAL;
        falls back to per-category defaults below if not present (see
        DEFAULT_K_THRUST etc.) so this still runs on existing scenario
        JSON files that predate these fields.
"""

import numpy as np
import torch

from gen_uav_em import GeneralUAVEnergyModel

# Fallback pack size if a scenario predates the variable-battery field.
_DEFAULT_BATTERY_CAPACITY_mAh = 8000.0

# Fallback per-robot physical coefficients if the scenario JSON doesn't
# carry them yet. These are NOT meant to be "the" values for any real
# drone -- just safe generic placeholders so existing scenario files
# keep working. Replace with per-category spec-sheet values when available.
_DEFAULT_K_THRUST = 0.010
_DEFAULT_K_DRAG = 0.06
_DEFAULT_MOTOR_EFF = 0.85

# Capacity-overload penalty is intentionally large relative to the
# achievement-scalarization distance term (which is typically O(1) in
# z-score units), so an overloaded schedule can never "win" against a
# feasible one on makespan/energy alone.
_CAPACITY_TERMINAL_PENALTY = 50.0


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
    Builds per-robot tensors needed for GeneralUAVEnergyModel.compute_batch:
      completed_coords      : [R, max_len+1, 3]
      speeds                : [R, max_len]
      service_times         : [R, max_len]
      payload_drop_kg        : [R, max_len]   -- CLIPPED so a robot's total
                                                  carried payload never exceeds
                                                  its robot_max_payload
      robot_own_weight_kg    : [R]
      initial_payload_kg     : [R]             -- clipped total (physically
                                                  carried), used for energy
      k_thrust, k_drag,
      motor_eff              : [R]             -- per-robot physical coeffs
      raw_assigned_weight_kg : [R]             -- UNclipped sum of assigned
                                                  task weights, for reporting
      battery_capacity_mAh   : [R]
      capacity_exceeded       : [R] bool        -- True if raw assigned weight
                                                  > robot_max_payload
      valid_mask             : [R, max_len]
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
    k_thrust = np.array(
        scenario.get("k_thrust", [_DEFAULT_K_THRUST] * R), dtype=np.float32
    )
    k_drag = np.array(
        scenario.get("k_drag", [_DEFAULT_K_DRAG] * R), dtype=np.float32
    )
    motor_eff = np.array(
        scenario.get("motor_eff", [_DEFAULT_MOTOR_EFF] * R), dtype=np.float32
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
            # (the infeasibility itself is penalized via capacity_exceeded
            # / the terminal reward, not silently hidden in the energy number).
            scale = cap / raw_total if raw_total > 0 else 1.0
        else:
            scale = 1.0

        carried_total = 0.0
        completed_coords[r, 0, :] = depot

        for i, t in enumerate(route_tasks):
            completed_coords[r, i + 1, :] = task_pos[t]

            v_raw = velocity[r, t]
            speeds_norm[r, i] = v_raw / GeneralUAVEnergyModel_SPEED_SCALE
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
        torch.tensor(k_thrust),
        torch.tensor(k_drag),
        torch.tensor(motor_eff),
        torch.tensor(battery_capacity_mAh),
        torch.tensor(valid_mask),
        raw_assigned_weight_kg,
        capacity_exceeded,
    )


# Module-level constant pulled out of the class for use above
# (GeneralUAVEnergyModel exposes it as a classmethod default; keeping a
# local alias avoids importing torch-heavy internals twice).
GeneralUAVEnergyModel_SPEED_SCALE = 10.0


# ============================================================
# Compute per-robot travel+service TIME (for makespan / variance)
# ============================================================
def compute_route_times(scenario, schedule, velocity, space_scale=100.0):
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
            dist_m = np.linalg.norm(target_pos - current_pos) * space_scale
            v_ms = max(velocity[r, t], 1e-3)
            travel_time_s = dist_m / v_ms

            total_time += travel_time_s + task_service_time[t]
            current_pos = target_pos

        dist_home_m = np.linalg.norm(robot_pos[r] - current_pos) * space_scale
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
    plus per-robot breakdowns, capacity_exceeded, and a "done" flag
    signalling the episode should terminate (any robot over capacity).
    """
    robot_times = compute_route_times(scenario, schedule, velocity)

    (
        completed_coords,
        speeds_norm,
        service_times,
        payload_drop_kg,
        robot_own_weight_kg,
        initial_payload_kg,
        k_thrust,
        k_drag,
        motor_eff,
        battery_capacity_mAh,
        valid_mask,
        raw_assigned_weight_kg,
        capacity_exceeded,
    ) = build_route_tensors(scenario, schedule, velocity)

    total_Wh, leg_Wh = GeneralUAVEnergyModel.compute_batch(
        completed_coords,
        speeds_norm,
        service_times,
        payload_drop_kg,
        robot_own_weight_kg,
        initial_payload_kg,
        k_thrust,
        k_drag,
        motor_eff,
    )

    total_Wh_np = total_Wh.detach().cpu().numpy()

    makespan = float(robot_times.max())
    workload_variance = float(robot_times.var())
    total_energy = float(total_Wh_np.sum())

    battery_Wh = GeneralUAVEnergyModel.battery_Wh_from_capacity(battery_capacity_mAh) \
        if hasattr(GeneralUAVEnergyModel, "battery_Wh_from_capacity") else None
    if battery_Wh is not None:
        soc_drop = GeneralUAVEnergyModel.soc_drop_from_energy(total_Wh, battery_Wh)
        soc_drop_np = soc_drop.detach().cpu().numpy()
        initial_battery_pct = np.array(scenario["robot_battery"], dtype=np.float32)
        battery_left = np.clip(initial_battery_pct - soc_drop_np, 0.0, 100.0)
        battery_exceeded = bool((soc_drop_np > 100.0).any())
    else:
        soc_drop_np = None
        battery_left = None
        battery_exceeded = False

    # ---- Severity of capacity overload, per robot -----------------------
    robot_max_payload = np.array(
        scenario.get("robot_max_payload", [np.inf] * scenario["R"]), dtype=np.float32
    )
    overload_kg = np.clip(raw_assigned_weight_kg - robot_max_payload, 0.0, None)
    # Fractional overload relative to that robot's own capacity, e.g.
    # 5kg over on a 10kg-capacity robot -> 0.5 ; 5kg over on a 25kg
    # robot -> 0.2. This is what the reward penalty scales with.
    overload_frac = np.divide(
        overload_kg, robot_max_payload,
        out=np.zeros_like(overload_kg), where=robot_max_payload > 0
    )

    done = bool(capacity_exceeded.any())

    return {
        "makespan": makespan,
        "workload_variance": workload_variance,
        "total_energy": total_energy,
        "robot_times": robot_times,
        "robot_energy_Wh": total_Wh_np,
        "robot_leg_energy_Wh": leg_Wh.detach().cpu().numpy(),
        "robot_soc_drop": soc_drop_np,
        "battery_exceeded": battery_exceeded,
        "battery_left": battery_left,
        "raw_assigned_weight_kg": raw_assigned_weight_kg,
        "capacity_exceeded": capacity_exceeded,        # per-robot bool array
        "any_capacity_exceeded": bool(capacity_exceeded.any()),
        "overload_frac": overload_frac,                # per-robot severity
        "max_overload_frac": float(overload_frac.max()) if len(overload_frac) else 0.0,
        "done": done,                                    # <-- terminal signal
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
    capacity_penalty_weight=_CAPACITY_TERMINAL_PENALTY,
    battery_penalty_weight=2.0,
):
    """
    objective_batch: dict with lists 'makespan', 'workload_variance',
                     'total_energy', and optionally:
                       'any_capacity_exceeded' -- list of bools
                       'max_overload_frac'     -- list of floats, worst
                                                   per-robot fractional
                                                   overload in each schedule
                                                   (pass obj['max_overload_frac']
                                                   from compute_objectives)
                       'done'                  -- list of bools, terminal
                                                   flag per schedule
                       'max_soc_drop'          -- list of floats, optional

    reference_point: [target_makespan, target_variance, target_energy]

    capacity_penalty_weight: BASE penalty for capacity overload. Actual
                     penalty = capacity_penalty_weight * (1 + max_overload_frac),
                     so a small overload still gets the full base penalty
                     (large by design, dominating dist), and a larger
                     overload gets progressively worse. Because this is a
                     TERMINAL condition, when 'done' is provided and True
                     for a sample, the normal makespan/variance/energy
                     scalarization is skipped entirely for that sample --
                     those objective values are not meaningful once a
                     robot's payload limit was violated -- and reward is
                     just -capacity_penalty_weight * (1 + max_overload_frac).

    battery_penalty_weight: penalty subtracted from reward, PROPORTIONAL
                     to how far the worst robot's SOC drop exceeds 100%.
                     Only applied to non-terminal (feasible-capacity)
                     samples, since a capacity-terminal sample already
                     gets its own dedicated terminal reward.
    """
    n = len(objective_batch["makespan"])

    done = np.array(
        objective_batch.get("done", objective_batch.get("any_capacity_exceeded", [False] * n)),
        dtype=bool,
    )
    max_overload_frac = np.array(
        objective_batch.get("max_overload_frac", [0.0] * n), dtype=np.float64
    )

    reward = np.zeros(n)

    # ---- Terminal (over-capacity) samples: dedicated large penalty ------
    terminal_penalty = -capacity_penalty_weight * (1.0 + max_overload_frac)
    reward[done] = terminal_penalty[done]

    # ---- Non-terminal samples: normal achievement-scalarization ---------
    non_terminal = ~done
    zm = np.zeros(n)
    zv = np.zeros(n)
    ze = np.zeros(n)

    if non_terminal.any():
        names = ["makespan", "workload_variance", "total_energy"]
        idx = np.where(non_terminal)[0]

        z_scores = {}
        stats = {}
        for name in names:
            vals = np.array(objective_batch[name], dtype=np.float64)[idx]
            z, mean, std = zscore_batch(vals)
            z_scores[name] = z
            stats[name] = (mean, std)

        ref_z = {}
        for name, target in zip(names, reference_point):
            mean, std = stats[name]
            ref_z[name] = (target - mean) / std

        penalty = np.zeros(len(idx))
        aug = np.zeros(len(idx))
        for name in names:
            diff = z_scores[name] - ref_z[name]
            penalty += np.maximum(diff, 0.0) ** 2
            aug += diff

        dist = np.sqrt(penalty) + rho * aug
        sub_reward = -dist

        if "max_soc_drop" in objective_batch:
            max_soc_drop = np.array(objective_batch["max_soc_drop"], dtype=np.float64)[idx]
            over_100_frac = np.maximum(max_soc_drop - 100.0, 0.0) / 100.0
            sub_reward = sub_reward - battery_penalty_weight * over_100_frac

        reward[idx] = sub_reward
        zm[idx] = z_scores["makespan"]
        zv[idx] = z_scores["workload_variance"]
        ze[idx] = z_scores["total_energy"]

    return reward, zm, zv, ze


# ============================================================
# Example usage -- loads a real scenario JSON via scenario_utils
# ============================================================
if __name__ == "__main__":
    from scenario_utils import load_scenario_file

    scenario = load_scenario_file("scenarios/scenario_10_50_2006.json")
    R, T = scenario["R"], scenario["T"]

    # ---- Case 1: a FEASIBLE round-robin schedule ------------------------
    schedule_ok = np.zeros((R, T), dtype=int)
    velocity = np.full((R, T), 5.0, dtype=float)
    slot = np.ones(R, dtype=int)
    for t in range(T):
        r = t % R
        schedule_ok[r, t] = slot[r]
        slot[r] += 1

    obj_ok = compute_objectives(scenario, schedule_ok, velocity)
    print("=== Feasible schedule ===")
    print(f"Makespan: {obj_ok['makespan']:.2f}")
    print(f"Workload Variance: {obj_ok['workload_variance']:.2f}")
    print(f"Total Energy (Wh): {obj_ok['total_energy']:.2f}")
    print(f"Capacity exceeded per robot: {obj_ok['capacity_exceeded']}")
    print(f"done (terminate episode?): {obj_ok['done']}")

    # ---- Case 2: force an INFEASIBLE schedule (dump everything on robot 0)
    schedule_bad = np.zeros((R, T), dtype=int)
    for t in range(T):
        schedule_bad[0, t] = t + 1  # every task assigned to robot 0

    obj_bad = compute_objectives(scenario, schedule_bad, velocity)
    print("\n=== Overloaded schedule (all tasks on robot 0) ===")
    print(f"Raw assigned weight per robot (kg): {obj_bad['raw_assigned_weight_kg']}")
    print(f"Robot max payload (kg): {scenario['robot_max_payload']}")
    print(f"Capacity exceeded per robot: {obj_bad['capacity_exceeded']}")
    print(f"Overload fraction per robot: {obj_bad['overload_frac']}")
    print(f"done (terminate episode?): {obj_bad['done']}")

    # ---- Reward comparison, batched -------------------------------------
    batch = {
        "makespan": [obj_ok["makespan"], obj_bad["makespan"]],
        "workload_variance": [obj_ok["workload_variance"], obj_bad["workload_variance"]],
        "total_energy": [obj_ok["total_energy"], obj_bad["total_energy"]],
        "any_capacity_exceeded": [obj_ok["any_capacity_exceeded"], obj_bad["any_capacity_exceeded"]],
        "max_overload_frac": [obj_ok["max_overload_frac"], obj_bad["max_overload_frac"]],
        "done": [obj_ok["done"], obj_bad["done"]],
    }
    reward, zm, zv, ze = compute_rewards(batch, reference_point=[30, 20, 9])
    print(f"\nReward [feasible, overloaded]: {reward}")
    # Expect: feasible schedule reward ~ small negative (normal distance);
    # overloaded schedule reward = large negative, dominated entirely by
    # the terminal capacity penalty, regardless of its makespan/energy.