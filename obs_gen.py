import mujoco
import mujoco.viewer
import numpy as np
import time

N_OBSTACLES = 10
N_LIDAR = 30
LIDAR_ANGLE_STEP = 360 / N_LIDAR          # 12 degrees between rays
ARENA_HALF_SIZE = 0.65
OBSTACLE_MIN_DIST = 1.0                 # min distance from robot along chosen ray
OBSTACLE_MAX_DIST = 1.5                   # max distance from robot along chosen ray
GOAL_MIN_DIST_FROM_OBSTACLES = 0.2

def reset_episode(model, data, n_obstacles=N_OBSTACLES):
    # 1. Spawn robot at a random location/yaw in the arena
    robot_xy = np.random.uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE, size=2)
    robot_yaw = np.random.uniform(-np.pi, np.pi)

    qpos_addr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    data.qpos[qpos_addr:qpos_addr+3] = [robot_xy[0], robot_xy[1], 0.030]
    quat = np.array([np.cos(robot_yaw/2), 0, 0, np.sin(robot_yaw/2)])
    data.qpos[qpos_addr+3:qpos_addr+7] = quat

    # 2. Pick 10 random distinct LiDAR ray indices (out of 30), spawn 1 obstacle along each
    chosen_ray_idxs = np.random.choice(N_LIDAR, size=n_obstacles, replace=False)
    obstacle_points = []

    for i, ray_idx in enumerate(chosen_ray_idxs):
        local_angle = np.radians(ray_idx * LIDAR_ANGLE_STEP)
        world_angle = robot_yaw + local_angle          # ray direction in world frame
        dist = np.random.uniform(OBSTACLE_MIN_DIST, OBSTACLE_MAX_DIST)

        obs_xy = robot_xy + dist * np.array([np.cos(world_angle), np.sin(world_angle)])
        obs_xy = np.clip(obs_xy, -ARENA_HALF_SIZE, ARENA_HALF_SIZE)  # keep inside arena
        obstacle_points.append(obs_xy)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        model.body_pos[body_id][:2] = obs_xy
        model.body_pos[body_id][2] = 0.05

    # 3. Random goal position, at least 0.2 away from every obstacle
    for _ in range(200):
        goal_xy = np.random.uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE, size=2)
        if all(np.linalg.norm(goal_xy - o) >= GOAL_MIN_DIST_FROM_OBSTACLES for o in obstacle_points):
            break

    goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
    model.body_pos[goal_id][:2] = goal_xy
    model.body_pos[goal_id][2] = 0.05

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    return {"obstacles": obstacle_points, "robot_start": robot_xy, "goal": goal_xy}


# ---------------- Example usage ----------------
if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("mrs-world.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

        for ep in range(10):
            info = reset_episode(model, data)
            print(f"Episode {ep}: robot={info['robot_start']}, goal={info['goal']}")

            start_time = time.time()
            while time.time() - start_time < 5.0:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

            if not viewer.is_running():
                break