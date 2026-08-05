"""
Preprocessing pipeline: turn raw per-map/per-robot trajectory CSV into
fixed-length, normalized tensors + context vectors ready for diffusion
model training.

Expected input CSV columns (adjust COLUMN MAP below if yours differ):
    map_number, robot_id, t, x, y, vx, vy, speed

Output per trajectory sample:
    traj_tensor : (H, 4)  -> [x, y, vx, vy], resampled to fixed horizon H, normalized to [-1, 1]
    context     : dict with start (2,), goal (2,), obstacles (list), map_number, robot_id

Also builds a global normalization stats file (min/max) needed later to
de-normalize model outputs back to world units.
"""

import os
import json
import numpy as np
import pandas as pd

# ---------------------- CONFIG ----------------------
CSV_PATH = "outputs/multi_robot_trajectories.csv"   # <-- update to your real file
MAP_JSON_DIR = "all_traj"          # where map_XXX_robot_Y.json (obstacle files) live
OUT_DIR = "outputs/preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

H = 64            # fixed horizon (waypoints per trajectory) - matches MPD paper default
COLUMNS = {        # map your CSV's actual column names here if different
    "map_number": "map_number",
    "robot_id": "robot_id",
    "t": "t",
    "x": "x",
    "y": "y",
    "vx": "vx",
    "vy": "vy",
}


# ---------------------- LOAD ----------------------
def load_raw_csv(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    missing = [c for c in COLUMNS.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Update COLUMNS mapping in this script if your names differ."
        )
    if COLUMNS["map_number"] not in df.columns:
        raise ValueError("CSV must include a map_number column to distinguish maps.")
    return df


# ---------------------- RESAMPLE TO FIXED HORIZON ----------------------
def resample_trajectory(traj_df, h=H):
    """Resample a single (map_number, robot_id) trajectory to exactly h waypoints
    using linear interpolation over time."""
    t = traj_df[COLUMNS["t"]].values
    x = traj_df[COLUMNS["x"]].values
    y = traj_df[COLUMNS["y"]].values
    vx = traj_df[COLUMNS["vx"]].values
    vy = traj_df[COLUMNS["vy"]].values

    t_new = np.linspace(t.min(), t.max(), h)
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    vx_new = np.interp(t_new, t, vx)
    vy_new = np.interp(t_new, t, vy)

    return np.stack([x_new, y_new, vx_new, vy_new], axis=1)  # (h, 4)


# ---------------------- BUILD RAW SAMPLES ----------------------
def build_samples(df):
    """Group by (map_number, robot_id) -> resample -> collect raw (unnormalized) tensors + context."""
    samples = []
    grouped = df.groupby([COLUMNS["map_number"], COLUMNS["robot_id"]])

    for (map_number, robot_id), group in grouped:
        group_sorted = group.sort_values(COLUMNS["t"])
        traj = resample_trajectory(group_sorted)  # (H, 4) raw units

        start = traj[0, :2]
        goal = traj[-1, :2]

        obstacles = load_obstacles_for_map(map_number, robot_id)

        samples.append({
            "map_number": map_number,
            "robot_id": robot_id,
            "traj_raw": traj,       # (H, 4) in raw world units
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
        })
    return samples


def load_obstacles_for_map(map_number, robot_id, map_json_dir=MAP_JSON_DIR):
    """Try to load obstacle list from the corresponding map_XXX_robot_Y.json.
    Falls back to empty list if file not found (e.g. dummy/test data)."""
    candidates = [
        os.path.join(map_json_dir, f"map_{map_number}_robot_{robot_id}.json"),
        os.path.join(map_json_dir, f"{map_number}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data.get("obstacles", [])
    return []


# ---------------------- NORMALIZATION ----------------------
def compute_global_norm_stats(samples):
    """Compute global min/max across all trajectories (x,y,vx,vy) for [-1,1] scaling."""
    all_traj = np.concatenate([s["traj_raw"] for s in samples], axis=0)  # (N*H, 4)
    mins = all_traj.min(axis=0)
    maxs = all_traj.max(axis=0)
    # avoid divide-by-zero for constant dims
    ranges = np.where((maxs - mins) < 1e-6, 1.0, (maxs - mins))
    return {"min": mins, "max": maxs, "range": ranges}


def normalize(traj_raw, stats):
    """Map raw (H,4) trajectory to [-1, 1] using global min/max."""
    return 2 * (traj_raw - stats["min"]) / stats["range"] - 1


def denormalize(traj_norm, stats):
    return (traj_norm + 1) / 2 * stats["range"] + stats["min"]


# ---------------------- MAIN ----------------------
def preprocess(csv_path=CSV_PATH):
    df = load_raw_csv(csv_path)
    samples = build_samples(df)
    stats = compute_global_norm_stats(samples)

    for s in samples:
        s["traj_norm"] = normalize(s["traj_raw"], stats).astype(np.float32)

    # Save as a single .npz dataset for fast loading during training
    traj_array = np.stack([s["traj_norm"] for s in samples])     # (N, H, 4)
    starts = np.stack([s["start"] for s in samples])              # (N, 2) raw units
    goals = np.stack([s["goal"] for s in samples])                # (N, 2) raw units
    map_numbers = np.array([s["map_number"] for s in samples])
    robot_ids = np.array([s["robot_id"] for s in samples])

    out_path = os.path.join(OUT_DIR, "dataset.npz")
    np.savez(
        out_path,
        traj_norm=traj_array,
        starts=starts,
        goals=goals,
        map_numbers=map_numbers,
        robot_ids=robot_ids,
        norm_min=stats["min"],
        norm_max=stats["max"],
        norm_range=stats["range"],
    )

    # Save obstacles separately as JSON (variable-length, can't go in npz cleanly)
    obstacles_by_sample = {
        f"{s['map_number']}_{s['robot_id']}": s["obstacles"] for s in samples
    }
    with open(os.path.join(OUT_DIR, "obstacles.json"), "w") as f:
        json.dump(obstacles_by_sample, f, indent=2)

    print(f"Preprocessed {len(samples)} trajectories from {df[COLUMNS['map_number']].nunique()} maps.")
    print(f"Trajectory tensor shape: {traj_array.shape}  (N, H, 4)")
    print(f"Saved dataset: {out_path}")
    print(f"Saved obstacles: {os.path.join(OUT_DIR, 'obstacles.json')}")

    return traj_array, starts, goals, map_numbers, robot_ids, stats


# ---------------------- QUICK SANITY TEST WITH DUMMY DATA ----------------------
def _make_dummy_csv(path="/home/claude/dummy_all_maps.csv", n_maps=5, n_robots=3, n_points=50):
    """Generate a small synthetic multi-map CSV matching the expected schema,
    purely to verify the preprocessing pipeline runs correctly end-to-end."""
    rows = []
    rng = np.random.default_rng(0)
    for m in range(1, n_maps + 1):
        for r in range(1, n_robots + 1):
            t = np.linspace(0, 10, n_points)
            start = rng.uniform(0, 12, size=2)
            goal = rng.uniform(0, 12, size=2)
            x = np.linspace(start[0], goal[0], n_points) + rng.normal(0, 0.1, n_points)
            y = np.linspace(start[1], goal[1], n_points) + rng.normal(0, 0.1, n_points)
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
            for i in range(n_points):
                rows.append({
                    "map_number": m, "robot_id": r, "t": t[i],
                    "x": x[i], "y": y[i], "vx": vx[i], "vy": vy[i],
                    "speed": np.hypot(vx[i], vy[i]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Real CSV not found at {CSV_PATH}. Running sanity test with dummy synthetic data...\n")
        dummy_path = _make_dummy_csv()
        preprocess(csv_path=dummy_path)
    else:
        preprocess()
