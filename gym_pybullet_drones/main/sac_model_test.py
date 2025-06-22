from gym_pybullet_drones.main.algorithms.wrappers.hovering_wrapper import HoveringWrapper

from gym_pybullet_drones.main.algorithms.wrappers.symmetric_wrapper import SymmetricWrapper

from gym_pybullet_drones.main.algorithms.wrappers.position_wrapper import PositionWrapper

from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv

from gym_pybullet_drones.main.algorithms.sacfd import SACfD
from gym_pybullet_drones.main.algorithms.dual_buffer import DualReplayBuffer
from stable_baselines3 import SAC,PPO
import numpy as np
import time
from gym_pybullet_drones.utils.utils import str2bool, sync
import pandas as pd
import csv
import os
import sys
import gym_pybullet_drones.main.algorithms
import gym_pybullet_drones.main.algorithms.lr_schedular
from types import ModuleType

def test_agent(agent, env, save_directory, num_episodes=5):
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = np.array(obs) 
        start_time = time.time()  # Start time of the episode
        done = False
        total_reward = 0
        counter = 0
        
        episode_positions = []  # To store positions and times for this episode
        episode_velocities = []  # To store velocities and times for this episode
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        
            # Get drone and payload positions
            drone_position = obs[:3]  # Assuming the first 3 values in obs are the drone's x, y, z positions
            payload_position = env.get_wrapper_attr('weight').get_position()
            
            # Calculate the real-world time elapsed since the start of the episode
            real_time_elapsed = time.time() - start_time
            
            # Append both drone and payload positions along with real time to the episode list
            episode_positions.append((real_time_elapsed, drone_position, payload_position))
            
            total_reward += reward
            env.render()
            
            # Get the velocity array
            vel = env.get_wrapper_attr('vel_arr')
            
            # Append velocity and the real_time_elapsed to the velocity list
            for v in vel:
                episode_velocities.append((real_time_elapsed, *v))
            
            sync(counter, start_time, env.get_wrapper_attr('CTRL_TIMESTEP'))
            if done or truncated:
                break
            counter += 1
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        # Save positions and real time for this episode to a separate CSV file
        episode_save_path = os.path.join(save_directory, f"drone_payload_positions_episode_{episode + 1}.csv")
        with open(episode_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Real_Time_Elapsed', 'Drone_X', 'Drone_Y', 'Drone_Z', 'Payload_X', 'Payload_Y', 'Payload_Z'])
            for timestep, (real_time, drone_position, payload_position) in enumerate(episode_positions):
                writer.writerow([real_time] + list(drone_position) + list(payload_position))
        
        # Save velocities and real_time_elapsed for this episode to a separate CSV file
        episode_velocities_save_path = os.path.join(save_directory, f"drone_velocities_episode_{episode + 1}.csv")
        vel_df = pd.DataFrame(episode_velocities, columns=["Real_Time_Elapsed", "vx", "vy", "vz"])
        vel_df.to_csv(episode_velocities_save_path, index=False)
    
    env.close()

import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the model directory (adjust the folder name as needed)
model_dir = os.path.join(BASE_DIR, "models/expr/SAC_1.2M_0_demos/")
model_dir = os.path.abspath(model_dir)

model_path = os.path.join(model_dir, "best_model.zip")
save_directory = os.path.join(model_dir, "episode_positions/")


sys.modules['gym_pybullet_drones.algorithms'] = gym_pybullet_drones.main.algorithms
# Create a fake 'rl' package in sys.modules to mimic the old structure
sys.modules['gym_pybullet_drones.utils.rl'] = ModuleType('gym_pybullet_drones.utils.rl')
# Map the old module path within the fake package to the new module location
sys.modules['gym_pybullet_drones.utils.rl.lr_schedular'] = gym_pybullet_drones.main.algorithms.lr_schedular


# Load the model
model = SAC.load(model_path)

# Create the test environment
test_env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(gui=True))))

# Test the agent and save positions for each episode in a separate file
test_agent(agent=model, env=test_env, save_directory=save_directory,num_episodes=3)
