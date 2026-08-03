"""
scenario_utils.py
-----------------
Utilities for loading pre-generated UAV mission scenarios, building
normalized model input features, and generating a random-but-balanced
initial schedule (used as the diffusion process's starting point --
NOT a ground truth to imitate).

Expected file structure:
    scenarios/scenario_{R}_{T}_{seed}.json

Three known scales are supported: (3,15), (5,25), (10,50) -- but the
loader is generic and will pick up any matching files present.
"""

import os
import re
import json
import random
import numpy as np


# ============================================================
# Scenario file discovery
# ============================================================
SCENARIO_DIR = "scenarios"
SCENARIO_PATTERN = re.compile(r"scenario_(\d+)_(\d+)_(\d+)\.json$")


def discover_scenarios(scenario_dir=SCENARIO_DIR):
    """
    Scan scenario_dir for files matching scenario_{R}_{T}_{seed}.json
    Returns a dict keyed by (R, T) -> list of full file paths.
    """
    if not os.path.isdir(scenario_dir):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    scale_map = {}
    for fname in os.listdir(scenario_dir):
        match = SCENARIO_PATTERN.match(fname)
        if match:
            R, T, seed = int(match.group(1)), int(match.group(2)), int(match.group(3))
            scale_map.setdefault((R, T), []).append(os.path.join(scenario_dir, fname))

    if not scale_map:
        raise RuntimeError(f"No scenario files matching pattern found in {scenario_dir}")

    return scale_map


def load_scenario_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================
# Random scenario sampler for training
# Picks a random scale, then a random scenario file within that scale
# ============================================================
class ScenarioSampler:
    def __init__(self, scenario_dir=SCENARIO_DIR, seed=None):
        self.scale_map = discover_scenarios(scenario_dir)
        self.scales = list(self.scale_map.keys())
        self.rng = random.Random(seed)

        print(f"Discovered {len(self.scales)} scale(s): {self.scales}")
        for scale, files in self.scale_map.items():
            print(f"  Scale {scale}: {len(files)} scenario file(s)")

    def sample(self, scale=None):
        """
        Returns (scenario_dict, filepath, (R, T)).
        If scale is None, picks a random scale first, then a random
        scenario file within it.
        """
        if scale is None:
            scale = self.rng.choice(self.scales)
        elif scale not in self.scale_map:
            raise ValueError(f"Requested scale {scale} not found. Available: {self.scales}")

        filepath = self.rng.choice(self.scale_map[scale])
        scenario = load_scenario_file(filepath)
        return scenario, filepath, scale


# ============================================================
# Normalization (z-score, per-feature)
# ============================================================
def zscore(x, axis=0, eps=1e-8):
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True) + eps
    return (x - mean) / std


def build_robot_features(scenario):
    """
    Robot feature vector per robot: [x, y, z, battery_z, capacity_z, own_weight_z]
    Positions kept as-is (already in [0,1]^3 arena). Battery/capacity/own_weight
    are z-scored since their raw ranges differ from position scale.
    """
    pos = np.array(scenario["robot_pos"])
    battery = zscore(np.array(scenario["robot_battery"]).reshape(-1, 1))
    capacity = zscore(np.array(scenario["robot_max_payload"]).reshape(-1, 1))
    own_weight = zscore(np.array(scenario["robot_own_weight"]).reshape(-1, 1))
    return np.hstack([pos, battery, capacity, own_weight]).astype(np.float32)


def build_task_features(scenario):
    """
    Task feature vector per task: [x, y, z, weight_z, service_time_z]
    """
    pos = np.array(scenario["task_pos"])
    weight = zscore(np.array(scenario["task_weight"]).reshape(-1, 1))
    service_time = zscore(np.array(scenario["task_service_time"]).reshape(-1, 1))
    return np.hstack([pos, weight, service_time]).astype(np.float32)


# ============================================================
# Random-but-BALANCED initial schedule
# (starting point for reverse diffusion -- NOT a ground truth to imitate)
#
# Unlike a purely random schedule (which can dump every task on one
# robot, as seen in untrained model tests), this spreads tasks evenly
# across robots first, then randomizes within that balanced structure,
# while still respecting capacity feasibility.
# ============================================================
def build_balanced_random_schedule(scenario, seed=None):
    rng = np.random.default_rng(seed)
    R, T = scenario["R"], scenario["T"]
    capacity = np.array(scenario["robot_max_payload"], dtype=float)
    weight = np.array(scenario["task_weight"], dtype=float)

    task_order = rng.permutation(T)
    remaining_capacity = capacity.copy()
    schedule = np.zeros((R, T), dtype=int)
    robot_next_slot = np.ones(R, dtype=int)

    # Round-robin robot pointer ensures tasks are spread across all
    # robots roughly evenly, instead of one robot absorbing everything.
    robot_cycle = rng.permutation(R).tolist()
    cycle_ptr = 0

    for t in task_order:
        assigned = False
        attempts = 0

        while attempts < R and not assigned:
            r = robot_cycle[cycle_ptr % R]
            cycle_ptr += 1
            attempts += 1

            if remaining_capacity[r] >= weight[t]:
                schedule[r, t] = robot_next_slot[r]
                robot_next_slot[r] += 1
                remaining_capacity[r] -= weight[t]
                assigned = True

        # If no robot in round-robin order could take it (capacity
        # exhausted everywhere for this task), fall back to any
        # feasible robot regardless of round-robin position.
        if not assigned:
            candidates = np.where(remaining_capacity >= weight[t])[0]
            if len(candidates) > 0:
                r = rng.choice(candidates)
                schedule[r, t] = robot_next_slot[r]
                robot_next_slot[r] += 1
                remaining_capacity[r] -= weight[t]
            # else: task genuinely infeasible given remaining capacity,
            # left unassigned (0) -- can happen near the demand/capacity
            # boundary; downstream training should handle unassigned cells.

    return schedule


# ============================================================
# Quick self-test / usage example
# ============================================================
if __name__ == "__main__":
    sampler = ScenarioSampler(seed=0)

    scenario, filepath, scale = sampler.sample()
    print(f"\nSampled scenario: {filepath} (scale={scale})")

    robot_feats = build_robot_features(scenario)
    task_feats = build_task_features(scenario)
    print(f"Robot features shape: {robot_feats.shape}")
    print(f"Task features shape: {task_feats.shape}")

    init_schedule = build_balanced_random_schedule(scenario, seed=0)
    print(f"\nBalanced random initial schedule:\n{init_schedule}")

    tasks_per_robot = [np.sum(init_schedule[r] > 0) for r in range(scenario["R"])]
    print(f"Tasks per robot: {tasks_per_robot}  (should be spread across robots, not concentrated)")