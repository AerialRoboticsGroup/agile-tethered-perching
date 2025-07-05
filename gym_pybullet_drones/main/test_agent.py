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
import itertools


def test_agent(agent, env, save_directory, num_episodes=5):
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    for episode in range(num_episodes):
        #start from a specific position
        obs, info = env.reset(position=np.array([1.97,0.0,3.00]))
        
        #start from changing position
        obs, info = env.reset()
        
        
        obs = np.array(obs) 
        start_time = time.time()  # Start time of the episode
        done = False
        total_reward = 0
        counter = 0

        episode_positions = []  # To store positions and times for this episode
        episode_velocities = []  # To store velocities and times for this episode
        
        episode_thrusts = []  # <-- Add this list
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
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
                
                
                
            per_motor_thrust = info["thrusts"][0]
            total_thrust = info["total_thrust"][0] if "total_thrust" in info else sum(per_motor_thrust)    
            
            episode_thrusts.append([real_time_elapsed] + list(per_motor_thrust) + [total_thrust])  
            
            sync(counter, start_time, env.get_wrapper_attr('CTRL_TIMESTEP'))
            
            
            
            
            if done or truncated:
                break
            counter += 1
            
            # if info['num_wraps'] != 0.0:
            #     print(f"Episode {episode + 1}: Wraps = {info['num_wraps']}")
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        
        # Save thrusts for this episode
        thrust_save_path = os.path.join(save_directory, f"drone_thrusts_episode_{episode + 1}.csv")
        with open(thrust_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Real_Time_Elapsed', 'Motor0_Thrust', 'Motor1_Thrust', 'Motor2_Thrust', 'Motor3_Thrust', 'Total_Thrust'])
            for row in episode_thrusts:
                writer.writerow(row)
        
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
    # Close the environment after all episodes
    # if info['num_wraps'] >= 0.49:
    #     print("true")
    env.close()
