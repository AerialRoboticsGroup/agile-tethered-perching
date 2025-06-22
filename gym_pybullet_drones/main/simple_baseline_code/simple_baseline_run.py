import argparse
import numpy as np
import time
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync
from gym_pybullet_drones.envs.TetherModelSimulationEnvPID import TetherModelSimulationEnvPID
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
import pandas as pd


def load_waypoints(filename):
    """Load waypoints from a CSV file."""
    try:
        data = pd.read_csv(filename)
        # data[['x', 'y', 'z']] -= [0,0,2.7]
        # data[['x', 'y', 'z']] += [1.6,0,1.8]
        
        # data['x'] = data['y'] 
        # data['y'] = data['x']

        waypoints = data[['x', 'y', 'z']].values  # Extract x, y, z columns
        
        # data['drone_z'] = data['drone_z'] / 1000.00
        
        # waypoints = data[['drone_x', 'drone_y', 'drone_z']].values
        
        # print(waypoints)
        return waypoints
    except Exception as e:
        print(f"Error loading waypoints: {e}")
        return None


def load_and_process_trajectory(trajectory_file, processed_trajectory_file, hover_height, control_freq_hz, start_pos):
    
    
    # process_trajectory_with_cubic_spline(trajectory_file, processed_trajectory_file, [start_pos[0], start_pos[1], hover_height])
    trajectory_points = load_waypoints(trajectory_file)
    
    return trajectory_points

def run_simulation(simulation, target_pos):

    # simulation.reset()

    action = np.zeros((simulation.num_drones, 3))
    start_time = time.time()
    
    for i in range(0, int(simulation.duration_sec * simulation.control_freq_hz)):

        for j in range(simulation.num_drones):
            wp_index = min(simulation.wp_counters[j], len(target_pos) - 1)
            action[j, :] = target_pos[wp_index, :]
            
            obs, reward, terminated, truncated, info  = simulation.step(action)
        simulation.wp_counters += 1
        payload_position = simulation.weight.get_position()
         
        simulation.render()
        if simulation.gui:
            sync(i, start_time, simulation.CTRL_TIMESTEP)
    
    simulation.close()
    simulation.logger.save()
    simulation.logger.save_as_csv("pid")
    if simulation.plot:
        simulation.logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone simulation using CtrlAviary and DSLPIDControl')
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
    parser.add_argument('--trajectory_file', default='trajectory_baseline.csv', type=str, help='CSV file with trajectory points (default: "trajectory_1.csv")', metavar='')
    parser.add_argument('--processed_trajectory_file', default='processed_simple_baseline_traj.csv', type=str, help='CSV file with processed trajectory points (default: "processed_trajectory_1.csv")', metavar='')
    
    args = parser.parse_args()

    
    start_pos = np.array([1.97,0,3])
    simulation = TetherModelSimulationEnvPID(start_pos=start_pos,
        drone=args.drone,
        num_drones=args.num_drones,
        physics=args.physics,
        gui=args.gui,
        record_video=args.record_video,
        plot=args.plot,
        user_debug_gui=args.user_debug_gui,
        obstacles=args.obstacles,
        simulation_freq_hz=args.simulation_freq_hz,
        control_freq_hz=args.control_freq_hz,
        duration_sec=args.duration_sec,
        output_folder=args.output_folder,
        colab=args.colab,
        hover_height=args.hover_height,
        obs=ObservationType.KIN, 
        act=ActionType.PID,
        )

    # simulation.reset(pos=np.array([1.97,0,3]))

    target_pos = load_and_process_trajectory(args.trajectory_file, args.processed_trajectory_file, args.hover_height, args.control_freq_hz, start_pos)
    
    run_simulation(simulation, target_pos)
