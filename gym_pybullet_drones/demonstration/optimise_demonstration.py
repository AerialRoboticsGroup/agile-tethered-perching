import pandas as pd
import numpy as np
import json
import os
from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
from rdp import rdp
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

EXPR_BRANCH_POSITION_demo = [-0.705489013671875, 0.0519111213684082, 1.5764576416015625]
# EXPR_BRANCH_POSITION_demo = [-1.60869689941406,	-0.00543527841567993,	1.79931481933594]

NUM_DRONES = 1

def pre_process_waypoints(filename):
    """Load and preprocess waypoints from a CSV file."""
    data = pd.read_csv("./original_demos/" + filename)
    
    data['drone_x'] = data['drone_x'] / 1000.00 - EXPR_BRANCH_POSITION_demo[0]
    data['drone_y'] = data['drone_y'] / 1000.00 - EXPR_BRANCH_POSITION_demo[1]
    data['drone_z'] = data['drone_z'] / 1000.00 - EXPR_BRANCH_POSITION_demo[2] + 2.7
    
    # data = data[data['drone_z'] > 2.0]
    
    waypoints = data[['drone_x', 'drone_y', 'drone_z']].values
    
    return waypoints

def reduce_waypoints(waypoints, epsilon=0.05):
    """Simplify the trajectory using Ramer-Douglas-Peucker (RDP) algorithm."""
    simplified_waypoints = rdp(waypoints.tolist(), epsilon=epsilon)
    print(len( simplified_waypoints))
    return np.array(simplified_waypoints)

def smooth_waypoints(simplified_waypoints, filename):
    smoothed_waypoints = toppra_waypoints(simplified_waypoints, min(len(simplified_waypoints)*10,300),filename)
    smoothed_waypoints = smoothed_waypoints[['drone_x', 'drone_y', 'drone_z']].values
    print(f"waypoints no:{len(smoothed_waypoints)}")
    return simplified_waypoints

def toppra_waypoints(simplified_waypoints, num_samples, filename):
    
    """Toppra Algorithm to smooth the trajectory"""
    
    # Extract x, y, z columns for processing
    waypoints = simplified_waypoints
    
    # Define the velocity and acceleration limits
    vlim = np.array([1, 1, 1])  # velocity limits in each axis
    alim = np.array([0.5, 0.5, 0.5])  # acceleration limits in each axis

    # Create path from waypoints
    path = ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints)

    # Create velocity and acceleration constraints
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)

    # Setup the parameterization problem
    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

    # Compute the trajectory
    jnt_traj = instance.compute_trajectory(0, 0)

    # Sample the trajectory
    N_samples = num_samples
    ss = np.linspace(0, jnt_traj.duration, N_samples)
    qs = jnt_traj(ss)

    # Extract the x, y, z components of the trajectory
    x = qs[:, 0]
    y = qs[:, 1]
    z = qs[:, 2]

    # Save the processed waypoints to a new CSV file
    processed_waypoints = pd.DataFrame({
        'drone_x': x,
        'drone_y': y,
        'drone_z': z
    })
    
    os.makedirs("optimised_demos", exist_ok=True)
    processed_file_path = "./optimised_demos/reduced_" + filename
    processed_waypoints.to_csv(processed_file_path, index=False)
    
    return processed_waypoints

csv_file = ["rosbag2_2024_05_22-17_00_56_filtered_normalized.csv", "rosbag2_2024_05_22-17_03_00_filtered_normalized.csv",
            "rosbag2_2024_05_22-17_20_43_filtered_normalized.csv", "rosbag2_2024_05_22-17_26_15_filtered_normalized.csv", 
            "rosbag2_2024_05_22-18_10_51_filtered_normalized.csv", "rosbag2_2024_05_22-18_16_45_filtered_normalized.csv"]

for i in range(len(csv_file)):
    print(f"------processing {csv_file[i]}------")
    waypoints = pre_process_waypoints(csv_file[i])
    reduced_trajectory_points = reduce_waypoints(waypoints, epsilon=0.01)
    smooth_waypoints(reduced_trajectory_points, csv_file[i])
    # target_pos, reduced_file_path = add_hover_waypoints(csv_file[i], hover_position=[reduced_trajectory_points[0][0], reduced_trajectory_points[0][1], max(2,reduced_trajectory_points[0][2])], trajectory_points=reduced_trajectory_points)
    print(f"---------Done----------")
