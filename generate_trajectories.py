"""
Generate synthetic time-parameterized (x, y, v, t) trajectories for multiple
robots from B-spline control-point path files + map/obstacle files.

Expected input files (per robot i):
  map_XXX_robot_{i}.json                -> start/goal/obstacles/task points
  map_XXX_robot_{i}_control_points.json -> B-spline segments (control points)

Pipeline per robot:
  1. Load path segments (each segment = start_point, 6 control points, end_point)
  2. Build a smooth curve per segment using scipy B-spline (quadratic/cubic fit
     through control points), evaluate densely.
  3. Concatenate all segments into one continuous path (arc-length parameterized).
  4. Convert arc-length -> time using nominal speed (map-units/sec).
  5. Resample at fixed Delta t to get uniformly time-spaced waypoints.
  6. Compute velocity via finite differences.
  7. Save to CSV: robot_id, t, x, y, vx, vy, speed
  8. Visualize all robots' paths + obstacles, and speed profiles.

Coordinate note: map files use pixel coords (0-800). We rescale to a
12x12 matplotlib "world" (matching user's plotting convention) using
scale = 12/800, then apply nominal speed 0.8 world-units/sec in that
rescaled space.
"""

import json
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------- CONFIG ----------------------
PROJECT_DIR = "all_traj"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

PIXEL_SIZE = 800.0          # original map is 800x800 pixels
WORLD_SIZE = 12.0           # matplotlib world is 12x12
SCALE = WORLD_SIZE / PIXEL_SIZE   # pixel -> world units

NOMINAL_SPEED = 0.8         # world-units / sec (user specified)
DT = 0.1                    # fixed timestep for resampling (sec)
DENSE_SAMPLES_PER_SEGMENT = 200  # resolution for spline evaluation
CONTROL_POINT_SUFFIXES = ["_control_points.json", "_manual_control_points.json"]


# ---------------------- FILE DISCOVERY ----------------------
def pixel_to_world(point):
    """Convert map pixel coordinates to world coordinates, flipping the Y axis."""
    x_px, y_px = point
    return np.array([x_px, PIXEL_SIZE - y_px]) * SCALE


MAP_FILENAME_RE = re.compile(r"map_(\d+)_robot_(\d+)\.json$")

def parse_map_filename(filename):
    m = MAP_FILENAME_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"Unable to parse map filename: {filename}")
    return int(m.group(1)), m.group(2)


def find_robot_files(project_dir=PROJECT_DIR):
    """Find all (map_file, control_points_file) pairs for each robot found."""
    map_files = sorted(glob.glob(os.path.join(project_dir, "map_*_robot_*.json")))
    entries = []
    for f in map_files:
        base = os.path.basename(f)
        if any(base.endswith(suffix) for suffix in CONTROL_POINT_SUFFIXES):
            continue
        map_number, robot_id = parse_map_filename(base)
        cp_file = None
        for suffix in CONTROL_POINT_SUFFIXES:
            candidate = f.replace(".json", suffix)
            if os.path.exists(candidate):
                cp_file = candidate
                break
        if cp_file is not None:
            entries.append({
                "map_number": map_number,
                "robot_id": robot_id,
                "map": f,
                "control_points": cp_file,
            })
    return entries


# ---------------------- SPLINE RECONSTRUCTION ----------------------
def evaluate_segment(start_point, control_points, end_point, n_samples=DENSE_SAMPLES_PER_SEGMENT):
    """Fit a B-spline through start -> control_points -> end and densely sample it."""
    pts = np.array([start_point] + list(control_points) + [end_point])
    x, y = pts[:, 0], pts[:, 1]

    # degree: cubic if enough points, else lower
    k = min(3, len(pts) - 1)
    try:
        tck, u = splprep([x, y], k=k, s=0)
        u_fine = np.linspace(0, 1, n_samples)
        x_fine, y_fine = splev(u_fine, tck)
    except Exception:
        # fallback: linear interpolation through control polygon
        u_fine = np.linspace(0, 1, n_samples)
        x_fine = np.interp(u_fine, np.linspace(0, 1, len(x)), x)
        y_fine = np.interp(u_fine, np.linspace(0, 1, len(y)), y)

    return np.column_stack([x_fine, y_fine])


def build_full_path(control_points_data):
    """Concatenate all segments. Control-point files are ALREADY in world
    units (12x12), unlike the map files which are in pixel units (800x800) -
    so no scaling is applied here."""
    segments = control_points_data["segments"]
    full_dense = []
    for i, seg in enumerate(segments):
        dense = evaluate_segment(seg["start_point"], seg["control_points"], seg["end_point"])
        if i > 0:
            dense = dense[1:]  # avoid duplicating shared point between segments
        full_dense.append(dense)
    full_dense = np.vstack(full_dense)
    return full_dense  # (N, 2) already in world units


# ---------------------- ARC-LENGTH -> TIME -> RESAMPLE ----------------------
def arc_length_time_resample(path_world, nominal_speed=NOMINAL_SPEED, dt=DT):
    """
    Given a dense (N,2) path in world units:
      - compute cumulative arc length
      - convert to time using nominal speed
      - resample at fixed dt
      - compute velocity via finite differences
    Returns DataFrame with columns t, x, y, vx, vy, speed
    """
    diffs = np.diff(path_world, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = arc_len[-1]
    total_time = total_len / nominal_speed

    time_at_point = arc_len / nominal_speed  # time stamp for each dense point

    # resample at fixed dt
    t_resampled = np.arange(0.0, total_time, dt)
    if t_resampled[-1] < total_time:
        t_resampled = np.append(t_resampled, total_time)

    x_resampled = np.interp(t_resampled, time_at_point, path_world[:, 0])
    y_resampled = np.interp(t_resampled, time_at_point, path_world[:, 1])

    # velocity via finite differences (central where possible)
    vx = np.gradient(x_resampled, t_resampled)
    vy = np.gradient(y_resampled, t_resampled)
    speed = np.sqrt(vx**2 + vy**2)

    df = pd.DataFrame({
        "t": t_resampled,
        "x": x_resampled,
        "y": y_resampled,
        "vx": vx,
        "vy": vy,
        "speed": speed,
    })
    return df


# ---------------------- MAIN PROCESSING ----------------------
def process_all_robots():
    entries = find_robot_files()
    if not entries:
        raise FileNotFoundError(f"No robot map/control_point file pairs found in {PROJECT_DIR}")

    all_trajs = []
    robot_meta = {}
    obstacles_by_map = {}

    for entry in sorted(entries, key=lambda e: (e["map_number"], int(e["robot_id"]))):
        map_number = entry["map_number"]
        robot_id = entry["robot_id"]
        with open(entry["map"]) as f:
            map_data = json.load(f)
        with open(entry["control_points"]) as f:
            cp_data = json.load(f)

        path_world = build_full_path(cp_data)
        df = arc_length_time_resample(path_world)
        df.insert(0, "map_number", map_number)
        df.insert(1, "robot_id", robot_id)
        all_trajs.append(df)

        task_points = sorted(map_data.get("task_points", {}).items(), key=lambda item: int(item[0]))
        key = f"{map_number}_{robot_id}"
        robot_meta[key] = {
            "map_number": map_number,
            "robot_id": robot_id,
            "map_data": map_data,
            "start_world": pixel_to_world(map_data["start_position"]),
            "goal_world": pixel_to_world(map_data["goal_position"]),
            "task_points_world": [(task_id, pixel_to_world(point)) for task_id, point in task_points],
        }
        obstacles_by_map.setdefault(map_number, map_data["obstacles"])

    combined = pd.concat(all_trajs, ignore_index=True)
    return combined, robot_meta, obstacles_by_map, entries


def save_csv(combined_df, out_path=os.path.join(OUT_DIR, "multi_robot_trajectories.csv")):
    combined_df.to_csv(out_path, index=False)
    return out_path


def visualize_all_maps(combined_df, robot_meta, obstacles_by_map, out_dir=OUT_DIR):
    saved_paths = []
    for map_number in sorted(obstacles_by_map.keys()):
        map_df = combined_df[combined_df["map_number"] == map_number]
        if map_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]

        for obs in obstacles_by_map[map_number]:
            pos = pixel_to_world(obs["position"])
            if obs["type"] == "circle":
                r = obs["radius"] * SCALE
                circ = patches.Circle(pos, r, color="gray", alpha=0.6)
                ax.add_patch(circ)
            elif obs["type"] == "rectangle":
                w, h = obs["width"] * SCALE, obs["height"] * SCALE
                rect = patches.Rectangle((pos[0] - w/2, pos[1] - h/2), w, h, color="gray", alpha=0.6)
                ax.add_patch(rect)

        map_robot_ids = sorted(map_df["robot_id"].unique(), key=int)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(map_robot_ids), 3)))
        for i, robot_id in enumerate(map_robot_ids):
            traj = map_df[map_df["robot_id"] == robot_id]
            meta_key = f"{map_number}_{robot_id}"
            meta = robot_meta[meta_key]
            ax.plot(traj["x"], traj["y"], color=colors[i], label=f"Robot {robot_id}", linewidth=2)
            ax.scatter(*meta["start_world"], color=colors[i], marker="o", s=80, edgecolor="k", zorder=5)
            ax.scatter(*meta["goal_world"], color=colors[i], marker="*", s=150, edgecolor="k", zorder=5)
            ax.annotate(f"S{robot_id}", xy=meta["start_world"], xytext=(5, 5), textcoords="offset points", color=colors[i], fontsize=10)
            ax.annotate(f"G{robot_id}", xy=meta["goal_world"], xytext=(5, -14), textcoords="offset points", color=colors[i], fontsize=10)

            task_points = meta.get("task_points_world", [])
            for task_id, task_world in task_points:
                ax.scatter(task_world[0], task_world[1], color=colors[i], marker="D", s=60, edgecolor="k", zorder=6)
                ax.text(task_world[0] + 0.12, task_world[1] + 0.12, f"R{robot_id}-T{task_id}", color=colors[i], fontsize=9)

        ax.text(0.02, 0.98, f"Map {map_number}", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
        ax.set_xlim(-2, 14)
        ax.set_ylim(-2, 14)
        ax.set_aspect("equal")
        ax.set_title(f"Multi-Robot Trajectories (Map {map_number})")
        ax.set_xlabel("x (world units)")
        ax.set_ylabel("y (world units)")
        ax.legend()

        ax2 = axes[1]
        for i, robot_id in enumerate(map_robot_ids):
            traj = map_df[map_df["robot_id"] == robot_id]
            ax2.plot(traj["t"], traj["speed"], color=colors[i], label=f"Robot {robot_id}")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("speed (world units/s)")
        ax2.set_title(f"Speed Profiles (Map {map_number})")
        ax2.legend()

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"multi_robot_trajectories_map_{map_number:03d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def check_pairwise_min_distance(combined_df):
    """Report closest approach between each pair of robots (interpolated onto common time grid)."""
    robot_ids = sorted(combined_df["robot_id"].unique())
    report = []
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            a = combined_df[combined_df["robot_id"] == robot_ids[i]]
            b = combined_df[combined_df["robot_id"] == robot_ids[j]]
            t_common = np.linspace(
                max(a["t"].min(), b["t"].min()),
                min(a["t"].max(), b["t"].max()),
                200
            )
            if len(t_common) < 2 or t_common[0] >= t_common[-1]:
                continue
            ax_ = np.interp(t_common, a["t"], a["x"])
            ay_ = np.interp(t_common, a["t"], a["y"])
            bx_ = np.interp(t_common, b["t"], b["x"])
            by_ = np.interp(t_common, b["t"], b["y"])
            dist = np.sqrt((ax_ - bx_)**2 + (ay_ - by_)**2)
            report.append((robot_ids[i], robot_ids[j], dist.min()))
    return report


if __name__ == "__main__":
    combined, robot_meta, obstacles_by_map, entries = process_all_robots()
    map_numbers = sorted({entry["map_number"] for entry in entries})
    robot_keys = [f"{entry['map_number']}_{entry['robot_id']}" for entry in entries]
    print(f"Found maps: {map_numbers}")
    print(f"Found robot entries: {sorted(robot_keys)}")
    csv_path = save_csv(combined)
    print(f"Saved CSV: {csv_path}")
    png_paths = visualize_all_maps(combined, robot_meta, obstacles_by_map)
    for path in png_paths:
        print(f"Saved visualization: {path}")

    for map_number in map_numbers:
        map_df = combined[combined["map_number"] == map_number]
        if map_df.empty:
            continue
        report = check_pairwise_min_distance(map_df)
        print(f"\nPairwise minimum distances for Map {map_number}: ")
        if report:
            for r1, r2, d in report:
                print(f"  Robot {r1} vs Robot {r2}: min dist = {d:.3f} world units")
        else:
            print("  Insufficient data for pairwise analysis on this map.")
