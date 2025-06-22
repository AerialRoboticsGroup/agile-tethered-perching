import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For plotting
from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
from gym_pybullet_drones.utils.utils import sync


mirror = False

def transform_demo(env, version, file_path):
    """Transform the demonstration into a JSON file."""
    df = pd.read_csv(file_path)
    
    if mirror == True:
        df['drone_x'] = -df['drone_x']
    
    waypoints = []

    has_hit = False
    for _, row in df.iterrows():
        if row['drone_x'] < 0.0:
            has_hit = True
        num_wraps = 2.0 if has_hit else 0.0
        waypoints.append((row['drone_x'], row['drone_y'], row['drone_z'], num_wraps))

    return step_pair_calculation(env, waypoints, version)

def step_pair_calculation(env, waypoints, version):
    """Calculate state-action-reward pairs from waypoints with synchronization."""
    state_action_reward = []
    reward_history = []  # Track rewards over time
    wrap_history = []
    curr_x, curr_y, curr_z, curr_w = waypoints[0]

    env.reset(position=np.array([curr_x, curr_y, curr_z]))

    start_time = time.time()

    for i in range(1, len(waypoints)):
        next_x, next_y, next_z, next_w = waypoints[i]
        action = np.array([next_x, next_y, next_z]).reshape((env.num_drones, -1))
        
        obs, reward, done, truncated, info = env.step(action)
        
        next_w = info['num_wraps']
        
        env.render()
      
        # Store the reward for visualization
        reward_history.append(reward)
        wrap_history.append(curr_w)
        action_diff_actual = (obs[0] - curr_x, obs[1] - curr_y, obs[2] - curr_z)

        state_action_reward.append(((curr_x, curr_y, curr_z, curr_w),
                                    (action[0][0], action[0][1], action[0][2]), reward,
                                    (obs[0], obs[1], obs[2], next_w)))

        curr_x, curr_y, curr_z, curr_w = obs[0], obs[1], obs[2], next_w

        sync(i, start_time, env.CTRL_TIMESTEP)
        
        # if done or truncated:
        #     break

    # After simulation is done, plot the reward history
    # plot_rewards(reward_history,wrap_history)

    return save_to_json(state_action_reward, version)

def plot_rewards(reward_history,wrap_history):
    """Plot rewards over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label='Reward')
    plt.plot(wrap_history, label='wraps')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

def save_to_json(state_action_reward, version):
    """Save state-action-reward pairs to a JSON file."""
    state_action_reward_serializable = [{
        "state": [float(x) for x in state],
        "action": [float(a) for a in action],
        "reward": float(reward),
        "next_state": [float(x) for x in next_state]
    } for state, action, reward, next_state in state_action_reward]

    os.makedirs("training_demos_json", exist_ok=True)
    if mirror:
        file_path = f"training_demos_json/rl_demo_new_mirror_{version}.json"
    else:
        file_path = f"training_demos_json/rl_demo_new_{version}.json"
    with open(file_path, 'w') as file:
        json.dump(state_action_reward_serializable, file, indent=4)

    print(f"Data saved to {file_path}")
    return file_path

def main():
    """Main function to process and transform demonstrations."""
    bulletDroneEnv = BulletDroneEnv()

    csv_files = ["rosbag2_2024_05_22-17_00_56_filtered_normalized.csv", 
                 "rosbag2_2024_05_22-17_03_00_filtered_normalized.csv",
                 "rosbag2_2024_05_22-17_20_43_filtered_normalized.csv", 
                 "rosbag2_2024_05_22-17_26_15_filtered_normalized.csv", 
                 "rosbag2_2024_05_22-18_10_51_filtered_normalized.csv", 
                 "rosbag2_2024_05_22-18_16_45_filtered_normalized.csv"]


    for i, csv_file in enumerate(csv_files):
        print(f"------ Processing {csv_file} ------")

        file_path = f"./optimised_demos/reduced_{csv_file}"
        json_file_path = transform_demo(bulletDroneEnv, i + 1, file_path)

        print("-------- Done ------------------")

if __name__ == "__main__":
    main()
