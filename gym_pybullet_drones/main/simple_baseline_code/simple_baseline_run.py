import argparse
import numpy as np
import time
import os
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync
from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
from gym_pybullet_drones.main.algorithms.wrappers.hovering_wrapper import HoveringWrapper
from gym_pybullet_drones.main.algorithms.wrappers.position_wrapper import PositionWrapper
from gym_pybullet_drones.main.algorithms.wrappers.symmetric_wrapper import SymmetricWrapper
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import itertools
import pandas as pd
import pybullet as p
import csv
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

thrust_log = []  # List to store thrusts per step

def load_and_process_trajectory(trajectory_file):
    # For TXT files, just load waypoints directly
    try:
        data = pd.read_csv(trajectory_file, delimiter=',', comment='/', header=0)
        waypoints = data.iloc[:, :3].values  # Use first three columns as x, y, z
        return waypoints
    except Exception as e:
        print(f"Error loading waypoints: {e}")
        return None

def run_simulation(env, target_pos, thrust_log_path=None):
    action = np.zeros((env.unwrapped.num_drones, 3))
    start_time = time.time()
    for i in range(0, int(env.unwrapped.duration_sec * env.unwrapped.control_freq_hz)):
        for j in range(env.unwrapped.num_drones):
            wp_index = min(env.unwrapped.wp_counters[j], len(target_pos) - 1)
            action[j, :] = target_pos[wp_index, :]
            obs, reward, terminated, truncated, info = env.step(action)
        env.unwrapped.wp_counters += 1
        
        per_motor_thrust = info["thrusts"][0]
        total_thrust = info["total_thrust"][0] if "total_thrust" in info else sum(per_motor_thrust)
        
        real_time_elapsed = time.time() - start_time
        thrust_log.append([real_time_elapsed] + list(per_motor_thrust) + [total_thrust])  # <-- Append per step
 
            
        if env.unwrapped.wp_counters[0] >= 300:
            print("Reached end of trajectory, resetting simulation.")
            break
        env.render()
        # if env.unwrapped.gui:
        sync(i, start_time, env.unwrapped.CTRL_TIMESTEP)
        
        
        # Save thrust log to CSV if path is provided
    if thrust_log_path is not None:
        with open(thrust_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['Real_Time_Elapsed', 'Motor0_Thrust', 'Motor1_Thrust', 'Motor2_Thrust', 'Motor3_Thrust', 'Total_Thrust'])
            writer.writerows(thrust_log)
    

def reset(self, position=None):
    p.resetSimulation(physicsClientId=self.CLIENT)
    # Reset drone and payload positions, velocities, etc.
    self.wp_counters = np.zeros(self.num_drones, dtype=int)
    # Reset any other stateful variables here
    # ...
    # Return initial observation and info
    return obs, info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone simulation using BulletDroneEnv and trajectory TXT')
    parser.add_argument('--drone', default=DroneModel("cf2x"), type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=1, type=int, help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics', default=Physics("pyb"), type=Physics, help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=True, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=False, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=True, type=str2bool, help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=False, type=str2bool, help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=False, type=str2bool, help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=120, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=48, type=int, help='Duration of the simulation in seconds (default: 60)', metavar='')
    parser.add_argument('--output_folder', default='results', type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=False, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--hover_height', default=1.5, type=float, help='Hover height in meters (default: 2.0)', metavar='')
    parser.add_argument('--trajectory_file', default=os.path.join(BASE_DIR, "simple_baseline_traj.csv"), metavar='')
    parser.add_argument('--processed_trajectory_file', default=os.path.join(BASE_DIR, "processed_simple_baseline_traj.csv"), type=str, help='CSV file with processed trajectory points (default: "processed_trajectory_1.csv")', metavar='')
    args = parser.parse_args()

    tether_lengths =[
        0.2, 0.25, 0.3, 0.35, 
        0.4, 0.45, 
        
        0.5, 0.55,
        0.6, 0.65, 0.7, 0.75, 
        0.8, 0.85, 0.9, 0.95,
        
        
        1.0, 
        
        1.05, 1.1, 1.15, 1.2,
        1.25, 1.3, 1.35, 1.4, 1.45,1.5,
        1.55, 1.6, 1.65, 1.75, 1.8,
        1.85, 1.9, 1.95,2.0,  
        
        2.2,2.4, 2.6, 2.7, 2.8,2.9, 3.0, 3.1,3.2
    ]
    
    
    tether_lengths = [1.0]
    
    
    
    weight_masses=[6e-7]
    
    # weight_masses = [
        
    #     2.00e-6, 1.95e-6, 1.90e-6, 1.85e-6, 1.80e-6, 1.75e-6, 1.70e-6,
    #     1.65e-6, 1.60e-6, 1.55e-6, 1.50e-6, 1.45e-6, 1.40e-6, 1.35e-6,
    #     1.30e-6, 1.25e-6, 1.20e-6, 1.15e-6, 1.10e-6, 1.05e-6, 
        
        
    #     1.00e-6,
        
        
        
    #     9.50e-7, 9.00e-7, 8.50e-7, 8.00e-7, 7.50e-7, 7.00e-7, 6.50e-7,
    #     6.00e-7, 5.50e-7, 5.00e-7, 4.50e-7, 4.00e-7, 3.50e-7, 3.00e-7,
    #     2.50e-7, 2.00e-7, 1.50e-7, 1.00e-7, 5.00e-8,1e-12
    # ]
    
    
    
    
    start_pos = np.array([1.97, 0.0, 3.0])

    for tether_length, weight_mass in itertools.product(tether_lengths, weight_masses):
        print(f"\n=== Running set: tether_length={tether_length}, weight_mass={weight_mass} ===\n")
        env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(gui=args.gui))))
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        base_env.tether_length = tether_length
        base_env.weight_mass = weight_mass

        # Always use env.unwrapped for wp_counters
        env.unwrapped.wp_counters = np.zeros(env.unwrapped.num_drones, dtype=int)

        target_pos = load_and_process_trajectory(
            args.trajectory_file
        )

        for run_idx in range(5):  # Run 5 times per combination
            print(f"\n--- Simulation run {run_idx + 1} | tether_length={tether_length}, weight_mass={weight_mass} ---\n")
            obs, info = env.reset(position=start_pos)
            env.unwrapped.wp_counters = np.zeros(env.unwrapped.num_drones, dtype=int)
            # Build a unique file path for each run
            thrust_log_path = os.path.join(
                BASE_DIR,
                f"thrust_log_tl{tether_length}_wm{weight_mass}_run{run_idx+1}.csv"
            )
            run_simulation(env, target_pos, thrust_log_path=thrust_log_path)


        env.close()
        # time.sleep(1)  # Give PyBullet time to release the GUI
