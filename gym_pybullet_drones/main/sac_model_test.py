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
from gym_pybullet_drones.main.test_agent import test_agent


# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the model directory (note the extra 'main/')
model_dir = os.path.join(BASE_DIR, "models/expr/SAC_1.2M_0_demos/")
model_dir = os.path.abspath(model_dir)

model_path = os.path.join(model_dir, "best_model.zip")
save_directory = os.path.join(model_dir, "episode_positions/")

#These files are here because the old version of the code was using different directory structure
#This is a temporary fix to allow models trained under the older version to run in the current settings
#If you will train a new model, you can remove these lines

sys.modules['gym_pybullet_drones.algorithms'] = gym_pybullet_drones.main.algorithms
# Create a fake 'rl' package in sys.modules to mimic the old structure
sys.modules['gym_pybullet_drones.utils.rl'] = ModuleType('gym_pybullet_drones.utils.rl')
# Map the old module path within the fake package to the new module location
sys.modules['gym_pybullet_drones.utils.rl.lr_schedular'] = gym_pybullet_drones.main.algorithms.lr_schedular


model = SAC.load(model_path)

# tether_lengths =[
#     0.2, 0.25, 0.3, 0.35, 
#     0.4, 0.45, 
    
#     0.5, 0.55,
#     0.6, 0.65, 0.7, 0.75, 
#     0.8, 0.85, 0.9, 0.95,
    
    
#     1.0, 
    
#     1.05, 1.1, 1.15, 1.2,
#     1.25, 1.3, 1.35, 1.4, 1.45,1.5,
#     1.55, 1.6, 1.65, 1.75, 1.8,
#     1.85, 1.9, 1.95,2.0,  
    
#     2.2,2.4, 2.6, 2.7, 2.8,2.9, 3.0, 3.1,3.2
# ]

tether_lengths = [
    1.0
]





weight_masses = [0.6e-6]


# weight_masses = [
    
#     2.00e-6, 1.95e-6, 1.90e-6, 1.85e-6, 1.80e-6, 1.75e-6, 1.70e-6,
#     1.65e-6, 1.60e-6, 1.55e-6, 1.50e-6, 1.45e-6, 1.40e-6, 1.35e-6,
#     1.30e-6, 1.25e-6, 1.20e-6, 1.15e-6, 1.10e-6, 1.05e-6, 
    
    
#     1.00e-6,
    
    
    
#     9.50e-7, 9.00e-7, 8.50e-7, 8.00e-7, 7.50e-7, 7.00e-7, 6.50e-7,
#     6.00e-7, 5.50e-7, 5.00e-7, 4.50e-7, 4.00e-7, 3.50e-7, 3.00e-7,
#     2.50e-7, 2.00e-7, 1.50e-7, 1.00e-7, 5.00e-8,1e-12
# ]

# --- Parameter sweep section ---

for tether_length, weight_mass in itertools.product(tether_lengths, weight_masses):
    print(f"Testing with tether_length={tether_length}, weight_mass={weight_mass}")

    t_env_start = time.time()
    def make_env():
        env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(gui=True))))
        # Drill down to the base environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        base_env.tether_length = tether_length
        base_env.weight_mass = weight_mass
        return env

    env = make_env()
    print(f"Environment creation took {time.time() - t_env_start:.2f} seconds")
    combo_save_dir = os.path.join(
        save_directory, f"tether_{tether_length}_weight_{weight_mass}"
    )
    os.makedirs(combo_save_dir, exist_ok=True)

    test_agent(agent=model, env=env, save_directory=combo_save_dir, num_episodes=5)
