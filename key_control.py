import mujoco
import cv2
import numpy as np
import json
import time
import os
import re

# ---------------- Constants ----------------
MAX_OBSTACLES = 30
ARENA_HALF_SIZE = 0.9   # matches the corner markers in mrs-world.xml
GOAL_REACH_THRESHOLD = 0.1

WHEEL_SEPARATION = 0.105
LINEAR_SPEED = 0.08
ANGULAR_SPEED = 0.2
RECORD_INTERVAL = 0.25
steps_per_frame = 16
KEY_TIMEOUT = 0.15   # key considered "released" if not seen within this window

MAP_FILE = "all_traj/map_002_robot_1.json"

# ---------------- Coordinate mapping (2D pixel map -> 3D mujoco world) ----------------
def pixel_to_world(px, py, img_size, arena_half=ARENA_HALF_SIZE):
    x = (px / img_size[0]) * 2 * arena_half - arena_half
    y = arena_half - (py / img_size[1]) * 2 * arena_half   # flip: image y grows down, world y grows up
    return x, y

def radius_to_world(r_px, img_size, arena_half=ARENA_HALF_SIZE):
    return (r_px / img_size[0]) * 2 * arena_half

def load_map(path):
    with open(path, "r") as f:
        m = json.load(f)
    img_size = m["robot_metadata"]["size"]

    start_x, start_y = pixel_to_world(*m["start_position"], img_size)
    goal_x, goal_y = pixel_to_world(*m["goal_position"], img_size)

    obstacles = []
    for obs in m["obstacles"][:MAX_OBSTACLES]:
        ox, oy = pixel_to_world(*obs["position"], img_size)
        orad = radius_to_world(obs["radius"], img_size)
        obstacles.append((ox, oy, orad))

    return (start_x, start_y), (goal_x, goal_y), obstacles

MAP_START, MAP_GOAL, MAP_OBSTACLES = load_map(MAP_FILE)

# ---------------- Episode reset (fixed map, no randomization) ----------------
def reset_episode(model, data):
    qpos_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]

    robot_yaw = 0.0
    data.qpos[qpos_addr:qpos_addr + 3] = [MAP_START[0], MAP_START[1], 0.031]
    quat = np.array([np.cos(robot_yaw / 2), 0, 0, np.sin(robot_yaw / 2)])
    data.qpos[qpos_addr + 3:qpos_addr + 7] = quat

    # Place obstacles from the loaded map; hide any unused slots below the floor.
    for i in range(MAX_OBSTACLES):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom_{i}")
        if i < len(MAP_OBSTACLES):
            ox, oy, orad = MAP_OBSTACLES[i]
            model.body_pos[body_id][:] = [ox, oy, 0.05]
            model.geom_size[geom_id][0] = orad
            model.geom_size[geom_id][1] = 0.05
        else:
            model.body_pos[body_id][:] = [5, 5, -5]

    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
    model.body_pos[goal_id][:] = [MAP_GOAL[0], MAP_GOAL[1], 0.05]

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return MAP_START, robot_yaw, MAP_OBSTACLES, MAP_GOAL

# ---------------- Main setup ----------------
model = mujoco.MjModel.from_xml_path("mrs-world.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=800, width=800)

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
root_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
qpos_addr = model.jnt_qposadr[root_jnt]

print("W/A/S/D: move+steer (combine for curves) | SPACE: brake | Q/ESC: quit")

episode_num = 1
last_seen = {"w": 0.0, "a": 0.0, "s": 0.0, "d": 0.0, " ": 0.0}

def key_active(k):
    return (time.time() - last_seen[k]) < KEY_TIMEOUT

# ---------------- Episode loop (never closes window) ----------------
while True:
    robot_start, robot_yaw0, obstacles, goal_xy = reset_episode(model, data)
    print(f"\nEpisode {episode_num} start: robot={robot_start}, goal={goal_xy}")

    episode_log = {"lidar": [], "action": [], "pose": [], "timestep": []}
    last_record_time = 0.0
    step_idx = 0
    recording_started = False
    quit_requested = False

    while True:
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key == 27 or key == ord('q'):
            quit_requested = True
            break

        # Update "held" state for every recognized key seen this poll.
        if key != 255:
            ch = chr(key) if key < 256 else None
            if ch in last_seen:
                last_seen[ch] = now

        # Compute v/omega from held-key state (robust to single-key polling).
        v, omega = 0.0, 0.0
        if key_active('w'):
            v += LINEAR_SPEED*0.1
        if key_active('s'):
            v -= LINEAR_SPEED*0.1
        if key_active('a'):
            omega += ANGULAR_SPEED*0.3
        if key_active('d'):
            omega -= ANGULAR_SPEED*0.3
        if key_active(' '):
            v, omega = 0.0, 0.0

        v=np.clip(v, -LINEAR_SPEED, LINEAR_SPEED)
        omega=np.clip(omega, -ANGULAR_SPEED, ANGULAR_SPEED)

        left_ctrl = v - omega * WHEEL_SEPARATION / 2
        right_ctrl = v + omega * WHEEL_SEPARATION / 2

        data.ctrl[left_id] = left_ctrl
        data.ctrl[right_id] = right_ctrl

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # --- Start recording only once movement begins ---
        if not recording_started and (abs(v) > 1e-6 or abs(omega) > 1e-6):
            recording_started = True
            last_record_time = data.time
            print("Movement detected — recording started.")

        if recording_started and data.time - last_record_time >= RECORD_INTERVAL:
            robot_x, robot_y = data.qpos[qpos_addr], data.qpos[qpos_addr + 1]
            quat = data.qpos[qpos_addr + 3:qpos_addr + 7]
            yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                              1 - 2 * (quat[2] ** 2 + quat[3] ** 2))

            episode_log["lidar"].append(data.sensordata.copy())
            episode_log["action"].append([v, omega])   # linear, angular velocity
            episode_log["pose"].append([robot_x, robot_y, yaw])
            episode_log["timestep"].append(step_idx)
            last_record_time = data.time
            step_idx += 1

            print(f"[rec step {step_idx}] v={v:.4f} m/s  omega={omega:.4f} rad/s")

            if np.linalg.norm([robot_x - goal_xy[0], robot_y - goal_xy[1]]) < GOAL_REACH_THRESHOLD:
                print("Goal reached!")
                break

        renderer.update_scene(data, camera="tpp_cam")
        pixels = renderer.render()
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"Ep {episode_num} | v:{v:.3f} w:{omega:.3f}",
                    (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.imshow("Robot Control", frame_bgr)

    if len(episode_log["timestep"]) > 0:
        # derive a map label from MAP_FILE (e.g. map_002_robot_1.json -> map_002)
        map_basename = os.path.basename(MAP_FILE)
        m = re.search(r"(map[_-]?\d+)", map_basename)
        map_label = m.group(1) if m else os.path.splitext(map_basename)[0]
        fname = f"episodes/{map_label}_epi{episode_num}.npz"
        np.savez(fname,
                 lidar=np.array(episode_log["lidar"]),
                 action=np.array(episode_log["action"]),
                 pose=np.array(episode_log["pose"]),
                 timestep=np.array(episode_log["timestep"]),
                 goal=goal_xy)
        print(f"Saved {len(episode_log['timestep'])} steps to {fname}")
        episode_num += 1

    if quit_requested:
        break

cv2.destroyAllWindows()