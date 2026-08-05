"""
Manual B-Spline Path Editor (Pygame)
------------------------------------
Loads the same map JSON format used by the DE planner, lets you:
  1. Click "Plot Path"  -> draws straight lines between consecutive waypoints
                            and drops 6 default (draggable) B-spline control
                            points per segment.
  2. Drag any control point with the mouse to reshape each segment's curve
                            (redrawn live using the same clamped B-spline
                            formulation as the DE script).
  3. Click "Save"        -> writes a single JSON with start/end/control
                            points per segment (in the 0-12 matplotlib
                            world scale), and also renders + saves a static
                            matplotlib PNG of the final path.

No differential_evolution / optimization of any kind is used here - this
is a purely manual editing tool.
"""

import pygame
import numpy as np
import json
import os
import datetime
from scipy.interpolate import BSpline
import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")  # headless backend, we only use it to save a PNG
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle

# ----------------------------------------------------------------------
# 0. HYPERPARAMETERS / CONFIG
# ----------------------------------------------------------------------

MAP_JSON_PATH = "maps/map_011_robot_3.json"   # <-- point this at your map file
OUTPUT_DIR = "solves/multi"

TARGET_SCALE = 12.0                 # world coordinate scale, same as the DE script (0-12 x 0-12)
N_CONTROL_PER_SEGMENT = 6           # free control points per segment (matches DE script)
BSPLINE_DEGREE = 3
N_SAMPLES_PER_SEGMENT = 60          # curve smoothness for drawing / saving

SCREEN_SIZE = 800                   # pygame window is SCREEN_SIZE x SCREEN_SIZE pixels
PIXELS_PER_UNIT = SCREEN_SIZE / TARGET_SCALE

CONTROL_POINT_RADIUS_PX = 8
CONTROL_POINT_GRAB_RADIUS_PX = 14
WAYPOINT_RADIUS_PX = 9

# --- Live cost display (same cost formulation as the DE script) ---
ROBOT_RADIUS = 0.15
W_LENGTH = 1.0
W_COLLISION = 800.0
W_CURVATURE = 0.5
W_BOUNDARY = 800.0
COST_LABEL_COLOR = (20, 20, 20)

COLOR_BG = (255, 255, 255)
COLOR_OBSTACLE = (200, 100, 100)
COLOR_STRAIGHT_LINE = (180, 180, 180)
COLOR_CURVE = (70, 40, 140)
COLOR_CONTROL_POINT = (230, 140, 20)
COLOR_CONTROL_POINT_DRAG = (230, 30, 30)
COLOR_START = (30, 140, 30)
COLOR_GOAL = (200, 30, 30)
COLOR_TASK = (230, 140, 20)
COLOR_BUTTON = (60, 120, 200)
COLOR_BUTTON_TEXT = (255, 255, 255)

FPS = 60

# ----------------------------------------------------------------------
# 1. Map Loading (same scaling convention as the DE script)
# ----------------------------------------------------------------------

def load_map_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    meta_key = "map_metadata" if "map_metadata" in data else "robot_metadata"
    orig_w, orig_h = data[meta_key]["size"]
    scale_factor = TARGET_SCALE / orig_w

    def scale_pt(pt):
        return np.array([pt[0] * scale_factor, (orig_h - pt[1]) * scale_factor])

    waypoints = [scale_pt(data["start_position"])]
    for task_id in data["task_sequence"]:
        waypoints.append(scale_pt(data["task_points"][str(task_id)]))
    if data.get("goal_position"):
        waypoints.append(scale_pt(data["goal_position"]))

    obstacles = []
    for obs in data["obstacles"]:
        obs_type = obs["type"]
        cx, cy = scale_pt(obs["position"])

        if obs_type == "circle":
            r_scaled = obs["radius"] * scale_factor
            obstacles.append({"type": "circle", "center": (cx, cy), "radius": r_scaled})

        elif obs_type == "square":
            h = (obs["size"] * scale_factor) / 2.0
            geom = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
            obstacles.append({"type": "polygon", "shape": geom})

        elif obs_type == "rectangle":
            hw = (obs["width"] * scale_factor) / 2.0
            hh = (obs["height"] * scale_factor) / 2.0
            geom = Polygon([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)])
            obstacles.append({"type": "polygon", "shape": geom})

        elif obs_type == "u_shape":
            h = (obs["size"] * scale_factor) / 2.0
            t_scaled = obs["thickness"] * scale_factor
            left_arm = Polygon([(cx - h, cy - h), (cx - h + t_scaled, cy - h), (cx - h + t_scaled, cy + h), (cx - h, cy + h)])
            right_arm = Polygon([(cx + h - t_scaled, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx + h - t_scaled, cy + h)])
            bottom_bar = Polygon([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy - h + t_scaled), (cx - h, cy - h + t_scaled)])
            obstacles.append({"type": "polygon", "shape": unary_union([left_arm, right_arm, bottom_bar])})

    map_name = os.path.splitext(os.path.basename(json_path))[0]
    return waypoints, obstacles, map_name

# ----------------------------------------------------------------------
# 2. B-Spline Curve (identical formulation to the DE script)
# ----------------------------------------------------------------------

def make_clamped_knot_vector(n_ctrl_pts, degree):
    n_internal = n_ctrl_pts - degree - 1
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
    else:
        internal_knots = np.array([])
    return np.concatenate((np.zeros(degree + 1), internal_knots, np.ones(degree + 1)))

def bspline_curve(control_points, n_samples, degree=BSPLINE_DEGREE):
    control_points = np.asarray(control_points)
    n_ctrl_pts = len(control_points)
    k = min(degree, n_ctrl_pts - 1)
    knots = make_clamped_knot_vector(n_ctrl_pts, k)
    t = np.linspace(0.0, 1.0, n_samples)
    spline_x = BSpline(knots, control_points[:, 0], k)
    spline_y = BSpline(knots, control_points[:, 1], k)
    return np.column_stack([spline_x(t), spline_y(t)])

def default_control_points(seg_start, seg_end):
    """Evenly spaced along the straight line, same baseline as the DE script's
    zero-delta starting point."""
    segment_vec = seg_end - seg_start
    if np.allclose(segment_vec, 0):
        return np.repeat(seg_start[None, :], N_CONTROL_PER_SEGMENT, axis=0)
    t_values = np.linspace(
        1.0 / (N_CONTROL_PER_SEGMENT + 1),
        N_CONTROL_PER_SEGMENT / (N_CONTROL_PER_SEGMENT + 1),
        N_CONTROL_PER_SEGMENT,
    )
    return np.array([seg_start + t * segment_vec for t in t_values])

# ----------------------------------------------------------------------
# 2b. Live Segment Cost (same formulation as the DE script, for display only)
# ----------------------------------------------------------------------

def path_length(curve):
    diffs = np.diff(curve, axis=0)
    return np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))

def curvature_penalty(curve):
    if len(curve) < 3:
        return 0.0
    v1 = curve[1:-1] - curve[:-2]
    v2 = curve[2:] - curve[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cos_angle = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
    return np.sum(np.arccos(cos_angle) ** 2)

def collision_penalty(curve, obstacles):
    """Geometry-based (continuous curve, not just sample points) collision
    check, same approach as the corrected DE script."""
    path_line = LineString(curve)
    swept_path = path_line.buffer(ROBOT_RADIUS)

    collided = False
    for obstacle in obstacles:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            obs_geom = Point(cx, cy).buffer(r)
        else:
            obs_geom = obstacle["shape"]

        if swept_path.intersects(obs_geom):
            collided = True
            break

    return 1.0 if collided else 0.0

def boundary_penalty(curve):
    xs = curve[:, 0]
    ys = curve[:, 1]
    intrusion_x = np.clip(0.0 - xs, 0, None) + np.clip(xs - TARGET_SCALE, 0, None)
    intrusion_y = np.clip(0.0 - ys, 0, None) + np.clip(ys - TARGET_SCALE, 0, None)
    return np.sum(intrusion_x ** 2) + np.sum(intrusion_y ** 2)

def segment_cost(curve, obstacles):
    cost = 0.0
    cost += W_LENGTH * path_length(curve)
    cost += W_COLLISION * collision_penalty(curve, obstacles)
    cost += W_CURVATURE * curvature_penalty(curve)
    cost += W_BOUNDARY * boundary_penalty(curve)
    return cost

# ----------------------------------------------------------------------
# 3. World <-> Screen coordinate conversion
# ----------------------------------------------------------------------

def world_to_screen(pt):
    x, y = pt
    sx = x * PIXELS_PER_UNIT
    sy = SCREEN_SIZE - y * PIXELS_PER_UNIT  # flip y: world is y-up, pygame is y-down
    return int(sx), int(sy)

def screen_to_world(pt):
    sx, sy = pt
    x = sx / PIXELS_PER_UNIT
    y = (SCREEN_SIZE - sy) / PIXELS_PER_UNIT
    return np.array([x, y])

# ----------------------------------------------------------------------
# 4. Button helper
# ----------------------------------------------------------------------

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, screen, font):
        pygame.draw.rect(screen, COLOR_BUTTON, self.rect, border_radius=6)
        text_surf = font.render(self.label, True, COLOR_BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# ----------------------------------------------------------------------
# 5. Rendering
# ----------------------------------------------------------------------

def draw_obstacles(screen, obstacles):
    for obstacle in obstacles:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            center_px = world_to_screen((cx, cy))
            r_px = int(r * PIXELS_PER_UNIT)
            surf = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*COLOR_OBSTACLE, 140), center_px, r_px)
            screen.blit(surf, (0, 0))
        else:
            poly = obstacle["shape"]
            xs, ys = poly.exterior.xy
            pts_px = [world_to_screen((x, y)) for x, y in zip(xs, ys)]
            surf = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            pygame.draw.polygon(surf, (*COLOR_OBSTACLE, 140), pts_px)
            screen.blit(surf, (0, 0))

def draw_waypoints(screen, waypoints, font):
    for i, wp in enumerate(waypoints):
        px = world_to_screen(wp)
        if i == 0:
            color = COLOR_START
        elif i == len(waypoints) - 1:
            color = COLOR_GOAL
        else:
            color = COLOR_TASK
        pygame.draw.circle(screen, color, px, WAYPOINT_RADIUS_PX)

def draw_straight_lines(screen, waypoints):
    for i in range(len(waypoints) - 1):
        p1 = world_to_screen(waypoints[i])
        p2 = world_to_screen(waypoints[i + 1])
        pygame.draw.line(screen, COLOR_STRAIGHT_LINE, p1, p2, 2)

def draw_curves_and_controls(screen, waypoints, control_points_per_segment, dragging_idx, obstacles, font):
    total_cost = 0.0
    for seg_idx, free_points in enumerate(control_points_per_segment):
        seg_start = waypoints[seg_idx]
        seg_end = waypoints[seg_idx + 1]
        full_ctrl = np.vstack([seg_start, free_points, seg_end])
        curve = bspline_curve(full_ctrl, N_SAMPLES_PER_SEGMENT)
        pts_px = [world_to_screen(p) for p in curve]
        if len(pts_px) > 1:
            pygame.draw.lines(screen, COLOR_CURVE, False, pts_px, 3)

        cost = segment_cost(curve, obstacles)
        total_cost += cost

        # Label the cost near the curve's midpoint
        mid_px = pts_px[len(pts_px) // 2]
        label = font.render(f"seg{seg_idx + 1}: {cost:.2f}", True, COST_LABEL_COLOR)
        screen.blit(label, (mid_px[0] + 8, mid_px[1] - 10))

        for cp_idx, cp in enumerate(free_points):
            px = world_to_screen(cp)
            color = COLOR_CONTROL_POINT_DRAG if dragging_idx == (seg_idx, cp_idx) else COLOR_CONTROL_POINT
            pygame.draw.circle(screen, color, px, CONTROL_POINT_RADIUS_PX)
            pygame.draw.circle(screen, (0, 0, 0), px, CONTROL_POINT_RADIUS_PX, 1)

    return total_cost

# ----------------------------------------------------------------------
# 6. Save (JSON control points + matplotlib PNG)
# ----------------------------------------------------------------------

def save_everything(waypoints, obstacles, control_points_per_segment, map_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- JSON export: start, end, control points per segment (world/matplotlib scale) ---
    segment_records = []
    curves = []
    for seg_idx, free_points in enumerate(control_points_per_segment):
        seg_start = waypoints[seg_idx]
        seg_end = waypoints[seg_idx + 1]
        full_ctrl = np.vstack([seg_start, free_points, seg_end])
        curve = bspline_curve(full_ctrl, N_SAMPLES_PER_SEGMENT)
        curves.append(curve)
        segment_records.append({
            "segment_index": seg_idx,
            "start_point": seg_start.tolist(),
            "end_point": seg_end.tolist(),
            "control_points": np.asarray(free_points).tolist(),
        })

    json_path = os.path.join(OUTPUT_DIR, f"{map_name}_manual_control_points.json")
    with open(json_path, 'w') as f:
        json.dump({
            "map_name": map_name,
            "num_segments": len(segment_records),
            "segments": segment_records,
        }, f, indent=4)
    print(f"Saved manual control points to {json_path}")

    # --- Matplotlib PNG export (same style/scale as the DE script's plot) ---
    fig, ax = plt.subplots(figsize=(9, 7))

    for obstacle in obstacles:
        if obstacle["type"] == "circle":
            cx, cy = obstacle["center"]
            r = obstacle["radius"]
            ax.add_patch(MplCircle((cx, cy), r, color="firebrick", alpha=0.55, zorder=2))
        else:
            poly = obstacle["shape"]
            xs, ys = poly.exterior.xy
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, color="firebrick", alpha=0.55, zorder=2))

    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(curves), 1)))
    for seg_idx, curve in enumerate(curves):
        ax.plot(curve[:, 0], curve[:, 1], "-", color=colors[seg_idx], linewidth=2.5, zorder=3,
                 label=f"Segment {seg_idx + 1}")

    for i, wp in enumerate(waypoints):
        if i == 0:
            ax.plot(*wp, "gs", markersize=13, zorder=4, label="Start")
        elif i == len(waypoints) - 1:
            ax.plot(*wp, "r*", markersize=20, zorder=4, label="Goal")
        else:
            ax.plot(*wp, "D", color="orange", markersize=11, zorder=4, label=f"Task {i}")

    ax.set_xlim(0, TARGET_SCALE)
    ax.set_ylim(0, TARGET_SCALE)
    ax.set_aspect("equal")
    ax.set_title(f"Manually Edited B-Spline Path - {map_name}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{map_name}_manual_path_{timestamp}.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved matplotlib path image to {png_path}")

    return json_path, png_path

# ----------------------------------------------------------------------
# 7. Main Application Loop
# ----------------------------------------------------------------------

def main():
    waypoints, obstacles, map_name = load_map_config(MAP_JSON_PATH)
    n_segments = len(waypoints) - 1

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 70))
    pygame.display.set_caption(f"Manual B-Spline Path Editor - {map_name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    plot_button = Button((20, SCREEN_SIZE + 15, 150, 40), "Plot Path")
    save_button = Button((190, SCREEN_SIZE + 15, 150, 40), "Save")

    path_plotted = False
    control_points_per_segment = None  # list of arrays, one per segment
    dragging_idx = None  # (seg_idx, cp_idx) currently being dragged

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                if plot_button.is_clicked(mouse_pos):
                    control_points_per_segment = [
                        default_control_points(waypoints[i], waypoints[i + 1])
                        for i in range(n_segments)
                    ]
                    path_plotted = True

                elif save_button.is_clicked(mouse_pos) and path_plotted:
                    save_everything(waypoints, obstacles, control_points_per_segment, map_name)

                elif path_plotted:
                    # check if a control point was grabbed
                    for seg_idx, free_points in enumerate(control_points_per_segment):
                        for cp_idx, cp in enumerate(free_points):
                            px = world_to_screen(cp)
                            dist = np.hypot(px[0] - mouse_pos[0], px[1] - mouse_pos[1])
                            if dist <= CONTROL_POINT_GRAB_RADIUS_PX:
                                dragging_idx = (seg_idx, cp_idx)
                                break
                        if dragging_idx is not None:
                            break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_idx = None

            elif event.type == pygame.MOUSEMOTION and dragging_idx is not None:
                seg_idx, cp_idx = dragging_idx
                new_world_pos = screen_to_world(event.pos)
                new_world_pos = np.clip(new_world_pos, 0.0, TARGET_SCALE)  # keep inside arena
                control_points_per_segment[seg_idx][cp_idx] = new_world_pos

        screen.fill(COLOR_BG)
        draw_obstacles(screen, obstacles)

        if path_plotted:
            total_cost = draw_curves_and_controls(screen, waypoints, control_points_per_segment, dragging_idx, obstacles, font)
            total_label = font.render(f"Total path cost: {total_cost:.2f}", True, (0, 0, 0))
            screen.blit(total_label, (20, 5))
        else:
            draw_straight_lines(screen, waypoints)

        draw_waypoints(screen, waypoints, font)

        pygame.draw.rect(screen, (235, 235, 235), (0, SCREEN_SIZE, SCREEN_SIZE, 70))
        plot_button.draw(screen, font)
        save_button.draw(screen, font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()