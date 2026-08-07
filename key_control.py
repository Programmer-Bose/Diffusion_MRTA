import mujoco
import cv2
import numpy as np
import random
import time

# ---------------- Constants ----------------
N_OBSTACLES = 10
N_LIDAR = 30
LIDAR_ANGLE_STEP = 360 / N_LIDAR
ARENA_HALF_SIZE = 0.85
ROBOT_SPAWN_HALF_SIZE = 0.5
OBSTACLE_RADIUS = 0.03
MIN_OBSTACLE_DIST = 0.5
MAX_EXTRA_DIST = 0.5
GOAL_MIN_CLEARANCE = 0.2
GOAL_REACH_THRESHOLD = 0.1

WHEEL_SEPARATION = 0.105
LINEAR_SPEED = 0.08
ANGULAR_SPEED = 0.2
RECORD_INTERVAL = 0.25
steps_per_frame = 16
KEY_TIMEOUT = 0.15   # key considered "released" if not seen within this window

# ---------------- Spawn logic ----------------
def clip_to_arena(p, margin=OBSTACLE_RADIUS):
    return np.clip(p, -ARENA_HALF_SIZE + margin, ARENA_HALF_SIZE - margin)

def reset_episode(model, data, n_obstacles=N_OBSTACLES):
    robot_xy = np.random.uniform(-ROBOT_SPAWN_HALF_SIZE, ROBOT_SPAWN_HALF_SIZE, size=2)
    robot_yaw = np.random.uniform(-np.pi, np.pi)

    qpos_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    data.qpos[qpos_addr:qpos_addr+3] = [robot_xy[0], robot_xy[1], 0.031]
    quat = np.array([np.cos(robot_yaw/2), 0, 0, np.sin(robot_yaw/2)])
    data.qpos[qpos_addr+3:qpos_addr+7] = quat

    candidate_indices = list(range(0, N_LIDAR, 2))
    chosen_indices = random.sample(candidate_indices, n_obstacles)
    obstacle_points = []

    for i, idx in enumerate(chosen_indices):
        angle = robot_yaw + np.radians(idx * LIDAR_ANGLE_STEP)
        dist = np.random.uniform(MIN_OBSTACLE_DIST, MIN_OBSTACLE_DIST + MAX_EXTRA_DIST)
        obs_xy = clip_to_arena(robot_xy + dist * np.array([np.cos(angle), np.sin(angle)]))
        obstacle_points.append(obs_xy)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        model.body_pos[body_id][:2] = obs_xy
        model.body_pos[body_id][2] = 0.05

    goal_xy = None
    for _ in range(200):
        candidate = np.random.uniform(-ARENA_HALF_SIZE + 0.05, ARENA_HALF_SIZE - 0.05, size=2)
        if all(np.linalg.norm(candidate - o) >= (OBSTACLE_RADIUS + GOAL_MIN_CLEARANCE) for o in obstacle_points):
            goal_xy = candidate
            break
    if goal_xy is None:
        goal_xy = candidate

    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
    model.body_pos[goal_id][:2] = goal_xy
    model.body_pos[goal_id][2] = 0.05

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return robot_xy, robot_yaw, obstacle_points, goal_xy

# ---------------- Main setup ----------------
model = mujoco.MjModel.from_xml_path("mrs-world.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=800, width=800)

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
root_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
qpos_addr = model.jnt_qposadr[root_jnt]

print("W/A/S/D: move+steer (combine for curves) | SPACE: brake | Q/ESC: quit")

episode_num = 29
last_seen = {"w": 0.0, "a": 0.0, "s": 0.0, "d": 0.0}
braking = False

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
        elif key == ord('w'):
            v, omega = LINEAR_SPEED, 0.0
        elif key == ord('s'):
            v, omega = -LINEAR_SPEED, 0.0
        elif key == ord('a'):
            v, omega = 0.0, ANGULAR_SPEED
        elif key == ord('d'):
            v, omega = 0.0, -ANGULAR_SPEED
        elif key == ord('w') and key == ord('a'):
            v, omega = LINEAR_SPEED*0.6, ANGULAR_SPEED
        elif key == ord('w') and key == ord('d'):
            v, omega = LINEAR_SPEED*0.6, -ANGULAR_SPEED
        elif key == ord(' '):
            braking = True
        elif key == 255:                      # no key pressed -> wheels stop immediately
            v, omega = 0.0, 0.0

        if braking:
            v, omega = 0.0, 0.0
            if not any(key_active(k) for k in last_seen):
                braking = False   # release brake once no drive key is held

        left_ctrl = v - omega * WHEEL_SEPARATION / 2
        right_ctrl = v + omega * WHEEL_SEPARATION / 2

        data.ctrl[left_id] = left_ctrl
        data.ctrl[right_id] = right_ctrl

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # --- Start recording only once movement begins ---
        if not recording_started and (abs(left_ctrl) > 1e-6 or abs(right_ctrl) > 1e-6):
            recording_started = True
            last_record_time = data.time
            print("Movement detected — recording started.")

        if recording_started and data.time - last_record_time >= RECORD_INTERVAL:
            robot_x, robot_y = data.qpos[qpos_addr], data.qpos[qpos_addr+1]
            quat = data.qpos[qpos_addr+3:qpos_addr+7]
            yaw = np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),
                              1-2*(quat[2]**2+quat[3]**2))

            episode_log["lidar"].append(data.sensordata.copy())
            episode_log["action"].append([left_ctrl, right_ctrl])
            episode_log["pose"].append([robot_x, robot_y, yaw])
            episode_log["timestep"].append(step_idx)
            last_record_time = data.time
            step_idx += 1

            if np.linalg.norm([robot_x - goal_xy[0], robot_y - goal_xy[1]]) < GOAL_REACH_THRESHOLD:
                print("Goal reached!")
                break

        renderer.update_scene(data, camera="overhead_cam")
        pixels = renderer.render()
        frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, f"Ep {episode_num} | L:{left_ctrl:.3f} R:{right_ctrl:.3f}",
                    (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.imshow("Robot Control", frame_bgr)

    if len(episode_log["timestep"]) > 0:
        fname = f"episode_{episode_num}.npz"
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