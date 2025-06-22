import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from gym_pybullet_drones.envs.BaseAviary import BaseAviary

from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.envs.simulationComponents.tether import Tether
from gym_pybullet_drones.envs.simulationComponents.weight import Weight
from gym_pybullet_drones.envs.simulationComponents.branch import Branch


from gym_pybullet_drones.main.algorithms.rewards.reward_system import RewardSystem

from gym_pybullet_drones.utils.utils import sync, print_green, print_red, load_json, StoreDict

class TetherModelSimulationEnvPID(BaseAviary):
    """Drone simulation class inheriting from BaseRLAviary."""

    def __init__(self, 
                 start_pos=[0, 0, 0],
                 drone=DroneModel.CF2X, 
                 num_drones=1, 
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB, 
                 gui=True, 
                 record_video=False, 
                 plot=True, 
                 user_debug_gui=False, 
                 obstacles=False, 
                 simulation_freq_hz=240, 
                 control_freq_hz=120, 
                 duration_sec=48, 
                 output_folder='results', 
                 colab=False, 
                 hover_height=2.0,
                 obs=ObservationType.KIN, 
                 act=ActionType.PID,
                 branch_init_pos = [0,0,2.7]):
        
        self.drone_init_pos = start_pos
        self.branch_init_pos = branch_init_pos
        self.INIT_XYZS = np.array([self.drone_init_pos for _ in range(num_drones)])
        # self.init_rpys = np.array([[0, 0, i * (np.pi / 2) / num_drones] for i in range(num_drones)])
        self.wp_counters = np.array([0 for _ in range(num_drones)])
        
        
        self.drone = drone
        self.num_drones = num_drones
        self.physics = physics
        self.gui = gui
        self.record_video = record_video
        self.plot = plot
        self.user_debug_gui = user_debug_gui
        self.obstacles = obstacles
        self.simulation_freq_hz = simulation_freq_hz
        self.control_freq_hz = control_freq_hz
        self.duration_sec = duration_sec
        self.output_folder = output_folder
        self.colab = colab
        self.hover_height = hover_height
        self.INIT_RPYS = np.zeros((self.num_drones, 3))
     
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(control_freq_hz//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
       
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
    
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
            
        
        # Call the parent class constructor with the necessary parameters
        # Call the parent class constructor with the necessary parameters
        super().__init__(drone_model=drone,
                         num_drones=num_drones,
                         neighbourhood_radius=np.inf,
                         initial_xyzs=self.INIT_XYZS,
                         initial_rpys=self.INIT_RPYS,
                         physics=physics,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=False, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False
                        )
        
        self.ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
        
        self.drone_id = self.getDroneIds()
        self.branch = None
        self.weight = None
        self.tether = None
        
        self.CTRL_TIMESTEP = 1.0 / self.control_freq_hz
        self.reward_system = None
        # Initialize custom simulation components like tether, branch, etc.
        self.initialize_simulation_components(self.branch_init_pos)
        
        self.vel_arr=[]        
        
        self.EPISODE_LEN_SEC = 15
        
        self.reward_info = 0.0
        self.dist_drone_branch = 0.0
        self.num_rotations = 0.0
        
        
        
        self.weight_prev_angle = None
       
        self.weight_cumulative_angle_change = 0.0

        self.max_steps = 300
        
        self.steps = 0
        
        self.step_size =1
        
        self.TARGET_POS = None
        
        self.remaining_length_tether = 0.0
        
        self.weight_wraps =0
        
        
       

    def initialize_simulation_components(self, branch_pos=None, variable=None):
        
        
        self.branch = Branch()
        if branch_pos is not None:
            self.branch.add_tree_branch(position=branch_pos)
        else:
            self.branch.add_tree_branch(position=self.branch_init_pos)
            
        self.branch_body_id = self.branch.get_body_id()  # Store the branch body ID
        
        
        if variable is not None:
            self.tether = Tether(self.getDroneIds(), length=1.0, drone_position=self.drone_init_pos, physics_client=self.CLIENT)
        else:
            random_tether_length = np.random.choice([1.0])
            self.tether = Tether(self.getDroneIds(), length=random_tether_length, drone_position=self.drone_init_pos, physics_client=self.CLIENT)
        
        
        
        
        
        self.tether.attach_to_drone(self.getDroneIds())
        payload_start_position_top = self.tether.get_world_centre_bottom()
        
        if variable is not None:
            self.weight = Weight(payload_start_position_top)
        else:
            random_weight_mass = np.random.choice([1e-6])
            self.weight = Weight(payload_start_position_top, mass = random_weight_mass )
            
        print_red(f"Tether Length:{self.tether.length}, Weight Mass:{self.weight.mass_weight}")
        
        self.tether.attach_weight(self.weight)
        
        # self.TARGET_POS = np.array([0,0,3.2])
        # print(self.TARGET_POS)
        
        
        contact_tether_length = (2 / 3 ) *  self.tether.length
        
        self.target = np.array([
            self.branch.get_tree_branch_midpoint()[0] - contact_tether_length * np.cos(np.radians(45)),  # Convert 45 degrees to radians
            self.branch.get_tree_branch_midpoint()[1],
            self.branch.get_tree_branch_midpoint()[2] + contact_tether_length * np.cos(np.radians(45))])
        
        line_length = 0.1  # Define how long you want the line to be
        target_start = self.target
        target_end = self.target + np.array([0, 0, line_length])  # Add a small offset in the Z direction to create a vertical line

        # Adding the line in the simulation
        p.addUserDebugLine(target_start, target_end, lineColorRGB=[1, 0, 0], lineWidth=2)


        self.reward_system = RewardSystem(branch_pos=self.branch.branch_pos, tether_length=self.tether.length)
        # print(self.reward_system)
        
       
    def reset (self,pos=None,seed=None, branch_pos=None, variable=None):
        if pos is not None:
            self.drone_init_pos = pos
        else:
            self.drone_init_pos = np.array([0.0,0.0,0.0])
        
        self.INIT_XYZS = np.array([self.drone_init_pos for _ in range(self.num_drones)])
        self.wp_counters = np.array([0 for _ in range(self.num_drones)])
        self.INIT_RPYS = np.zeros((self.num_drones, 3))
        
        self.reward_system.reset()
        self.step_size = 1
        super().reset()
        self.steps = 0
        
        self.weight_prev_angle = None
        self.num_rotations = 0
        self.weight_wraps = 0
        self.weight_cumulative_angle_change = 0
        self.initialize_simulation_components(branch_pos = branch_pos, variable=variable)
         
    def step(self, action, num_wraps= None):
        action = np.reshape(action, (self.NUM_DRONES, -1))

        is_wrapping = self.check_if_wrapping()

        if is_wrapping:
            force_magnitude, direction = self.apply_tether_force()
            new_action = self.compute_new_action(action, force_magnitude, direction)
            # print(force_magnitude)
        else:
            new_action = action
        
        drone_pos = self.get_drone_currrent_pos()
        weight_pos, _ = p.getBasePositionAndOrientation(self.weight.get_weight_id())  # Bottom of the tether

        # Calculate the distance between the drone and the weight
        distance = np.linalg.norm(np.array(drone_pos) - np.array(weight_pos))
        # print(distance)
        
        threshold = self.tether.length +self.tether.offset + self.weight.RADIUS + 0.03
        
        if num_wraps is None:
            obs, reward, terminated, truncated, info = super().step(new_action)
        else:
            obs, reward, terminated, truncated, info = super().step(new_action, num_wraps)
        # If the distance exceeds the tether length, restrict the drone's movement
        # if distance > threshold:
          
        #     p.resetBasePositionAndOrientation(int(self.getDroneIds()), drone_pos, p.getQuaternionFromEuler([0, 0, 0]))  # Keep the drone at its current position
            
            
        
        self.steps += 1
        
        return obs, reward, terminated, truncated, info
    
    ##############################################################################################
    '''reference: https://github.com/utiasDSL/gym-pybullet-drones'''
    def _actionSpace(self):
        """
        
        Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE==ActionType.PID:
            size = 3
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        # act_lower_bound = np.array([-10*np.ones(size) for i in range(self.NUM_DRONES)])
        # act_upper_bound = np.array([+10*np.ones(size) for i in range(self.NUM_DRONES)])
        
        act_lower_bound = np.array([[-5,-0.05, 0] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([ [5,0.05, 6] for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        # print(act_lower_bound)
        # print(f"pid:{spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)}")
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _preprocessAction(self,
                          action
                          ):
        '''reference: https://github.com/utiasDSL/gym-pybullet-drones'''
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        
        
        rpm = np.zeros((self.NUM_DRONES,4))
        
        
        for k in range(action.shape[0]):
            target = action[k, :] 
            # print(f"{target} = {action[k, :] } + {self.get_drone_currrent_pos()}")
            # print(f"target before process: {target}")
            if self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                # print(f"get state vector: {state[0:3]}")
                
                if self.check_if_wrapping():
                    next_pos = target
                else:
                    next_pos = self._calculateNextStep(
                        current_position=state[0:3],
                        destination=target,
                        step_size=self.step_size,
                        )
                # print(f"pid:{next_pos}")
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[-5,-0.05,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[5,0.05,6,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            # act_lo = -10
            # act_hi = +10
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE==ActionType.PID:
                    # obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    # obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[-5,-0 ,0] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[5,0, 6]for i in range(self.NUM_DRONES)])])
            
            # print(spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32))
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    def _computeObs(self):
        '''reference: https://github.com/utiasDSL/gym-pybullet-drones'''
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            # print(len(ret))
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

    def _computeReward(self, num_wraps= None):
        
        """Computes the reward based on the current state and action."""
        
        state = self.get_drone_currrent_pos()
        
        has_tether_contacted_branch = self.any_tether_contact_branch() #if so, start to count num_wrap
        
        dist_tether_branch = self._distance(self.tether.get_last_one_third_point(), self.branch.get_tree_branch_midpoint())
        dist_drone_branch = self._distance(state, self.branch.get_tree_branch_midpoint())
        if num_wraps is None:
            num_rotations = self.compute_total_rotation() if has_tether_contacted_branch else 0.0
        else:
            num_rotations = num_wraps
        velocity = self._getDroneStateVector(0)[10:13]
        weight_pos = self.weight.get_position()
        reward = self.reward_system.calculate(state, has_tether_contacted_branch, dist_tether_branch, dist_drone_branch, num_rotations,weight_pos)
        self.dist_drone_branch = dist_drone_branch
        self.num_rotations = num_rotations
        self.reward_info = reward
        
        return reward

    def _computeTerminated(self):
        # return False
        # state = self._getDroneStateVector(0)
        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
        #     return True
        # else:
        #     return False
        
        # done = self.current_waypoint_index >= len(self.waypoints) or self.steps >= self.max_steps
        # done = self.steps >= self.max_steps
        done = self.reward_system.refer_terminated()
        return done
    
    def _computeTruncated(self):
        # return
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        
        if np.linalg.norm(self.get_drone_currrent_pos() - self.branch.get_tree_branch_midpoint())>3:
            print("drone is way too far from the branch.")
            return True
        
        # Check for extreme tilt (crash due to flipping over)
        # roll, pitch = state[7], state[8]
        # if abs(roll) > np.radians(90) or abs(pitch) > np.radians(90):  # 60 degrees as a crash threshold
        #     print("Crash detected due to extreme tilt!")
        #     return True

        # Check for low altitude (crash into the ground)
        altitude = state[2]
        if altitude <=  0.0:  # Assuming 0.1 meters is the threshold for ground impact
            print("Crash detected due to ground impact!")
            return True

        # Check for sudden, large changes in velocity (high impact crash)
        velocity = state[10:13]
        if np.linalg.norm(velocity) > 10:  # Threshold for crash due to high-speed impact
            print("Crash detected due to high-speed impact!")
            return True
        
        
        if self.steps >= self.max_steps:
            print("Too long episode!")
            return True
  
    def _computeInfo(self):
        info = {"distance_to_goal": -self.reward_info, "has_crashed": bool(self.dist_drone_branch < 0.1), "num_wraps": self.num_rotations}

        return info #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################
    
    def compute_angle_change(self, current_angle, prev_angle):
        """Compute the angular change, considering wrap-around at ±180°."""
        angle_change = current_angle - prev_angle
        # Adjust the angle change to stay within the range of -180 to +180 degrees
        if angle_change > 180:
            angle_change -= 360
        elif angle_change < -180:
            angle_change += 360

        # # Ignore very small changes to reduce noise
        # if abs(angle_change) < 1:  # Adjust this threshold as needed
        #     return 0
            
        return angle_change
        
    def compute_total_rotation(self):
        '''reference: https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
        # Get the position of the weight
        (weight_x, _, weight_y), _ = p.getBasePositionAndOrientation(self.weight.get_weight_id())
        
        # Get the midpoint of the branch (where wrapping should happen)
        branch_mid_x, _, branch_mid_y = self.branch.get_tree_branch_midpoint()

        # Compute the deltas relative to the branch midpoint
        weight_delta_x = weight_x - branch_mid_x
        weight_delta_y = weight_y - branch_mid_y
        
        # Compute the angle using arctan2, which considers quadrant location
        weight_angle_radians = np.arctan2(weight_delta_x, weight_delta_y)
        weight_angle_degrees = np.degrees(weight_angle_radians)

        # Compute the distance between the weight and the branch
        distance_to_branch = np.sqrt(weight_delta_x**2 + weight_delta_y**2)

        # Define a threshold for when the weight is close enough to the branch to count as wrapping
        branch_proximity_threshold = 0.5  # Adjust based on your environment's scale

        # Check if the tether is taut (i.e., the drone/weight is engaged with the branch)
        if distance_to_branch < branch_proximity_threshold:
            # If the weight is near the branch, continue tracking the angle
            if self.weight_prev_angle is None:
                # Initialize the previous angle on first contact
                self.weight_prev_angle = weight_angle_degrees
                # print(f"hiiiiiiiiiiii{ self.weight_prev_angle}")
                return 0  # No wraps to return on first call
            
            # Calculate the angle change considering the wrap around at 180/-180
            weight_angle_change = self.compute_angle_change(weight_angle_degrees, self.weight_prev_angle)

            # Update cumulative angle change based on dynamic starting point
            self.weight_cumulative_angle_change += weight_angle_change

            # Update wraps as a float, allowing for both wrapping and unwrapping
            self.weight_wraps = self.weight_cumulative_angle_change / 360.0

            # Update the previous angle for the next call
            self.weight_prev_angle = weight_angle_degrees  # << Keep updating it with the current angle
        else:
                # If the weight is not near the branch, don't update the angle or wraps
                # You could choose to reset or pause wrap tracking here
                # self.weight_wraps = 0
                self.weight_prev_angle = None  # This pauses wrap tracking until the weight is near the branch again
                # print(f"back to leave { self.weight_prev_angle}")
        return abs(self.weight_wraps)  # or self.weight_wraps for signed wraps


    def compute_remaining_tether_length(self):
       
        num_wrapping = self.compute_total_rotation()
        
        # wrapped_length = num_wrapping * self.branch.get_branch_circumference() + num_wrapping * (self.tether.get_radius()*2 * 2 * np.pi)
        
        wrapped_length = num_wrapping * (self.branch.get_branch_circumference())
        
        # Remaining tether length
        remaining_length = self.tether.length - wrapped_length
        
        return max(0, remaining_length)  # Ensure it's non-negative
    
    def check_if_wrapping(self):
        
        # Check if any part of the tether contacts the branch
        if self.any_tether_contact_branch():
            
            
            # return True
            # # Calculate the number of wraps around the branch
            num_wrapping = self.compute_total_rotation()
            
            # Wrapping starts if the num_wrapping exceeds 0.75
            if num_wrapping > 0.1:
                return True
        
        return False
    
    def apply_tether_force(self):
        """
        Apply tether force to the drone if it tries to move beyond the remaining tether length.
        The drone is pulled towards the branch if the tether is taut.
        
        Returns:
            force_magnitude (float): The magnitude of the force applied to the drone.
            direction (np.ndarray): The direction in which the force is applied.
        """
        
        drone_pos = self.get_drone_currrent_pos()

        branch_midpoint = self.branch.get_tree_branch_midpoint()

        # Calculate the relative position of the drone to the branch (vector from branch to drone)
        relative_position = drone_pos - branch_midpoint

        # Calculate the distance from the drone to the branch (magnitude of the relative position vector)
        dist_drone_to_branch = np.linalg.norm(relative_position)

        # Compute the remaining tether length after accounting for wrapping
        self.remaining_length_tether = self.compute_remaining_tether_length()
        
        # self.remaining_length_tether -= self.tether.segment_length - self.tether.offset

        # Check if the drone is beyond the remaining tether length
        if dist_drone_to_branch > self.remaining_length_tether:
            # print(dist_drone_to_branch > self.remaining_length_tether)
            # Normalize the relative position vector to get the direction of the force
            direction = -relative_position / dist_drone_to_branch  # Negative to pull back towards the branch
            
            # Apply a force propo
            # rtional to how much the tether is stretched
            force_magnitude = 1.2* (dist_drone_to_branch -   self.remaining_length_tether)

            # Debugging logs to check force and direction
            # print(f"Applying tether force. Force magnitude: {force_magnitude}, Direction: {direction}")
            
            # Apply the force to the drone in the direction of the branch
            # if self.tether.top_position[0] - branch_midpoint[0] >= 0:
            # print(f"force{force_magnitude * direction}")
            p.applyExternalForce(self.drone_id[0], -1,  force_magnitude * direction, drone_pos, flags=p.WORLD_FRAME)
            # else:
            #     p.applyExternalForce(self.drone_id[0], -1, 0.5 *force_magnitude * direction, drone_pos, flags=p.WORLD_FRAME)
                
            # Return the force magnitude and direction to adjust the drone's target position
            return force_magnitude, direction
        
        return 0.0, None  # No force if the tether is not taut

    def compute_new_action(self, current_action, force_magnitude, direction):
        """
        Compute the new action (target position) based on the current action and tether force.
        
        Args:
            current_action (np.ndarray): The original action (target positions) for the drone.
            force_magnitude (float): The magnitude of the tether force.
            direction (np.ndarray): The direction in which the tether is pulling the drone.
        
        Returns:
            np.ndarray: The new action with the force considered.
            
        """
        # return current_action
        
        if direction is not None:
            adjustment = force_magnitude * direction  # Force applied in the direction of the branch
            new_action = current_action + adjustment  # Adjust the original target position based on the force
            # print(f"Original Action: {current_action}, Adjustment: {adjustment}, New Action: {new_action}")
        else:
            new_action = current_action  # No change if there's no force applied

        return new_action
    
    
    def any_tether_contact_branch(self):
        '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
        for part_id in self.tether.get_segments():
            contacts = p.getContactPoints(bodyA=self.branch_body_id, bodyB=part_id)
            if contacts:
                return True
        return False
    

    def render(self):
        super().render()
        
        weight_pos = self.weight.get_position()
        
        self.vel_arr.append(self.vel[0])
        text_pos = [-0.8, -0.9, 0.5]
        # print("[INFO] Payload Position: "
        #           "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(weight_pos[0], weight_pos[1], weight_pos[2]))
        
        if self.step_counter % 100 == 0:
        
            p.addUserDebugText(
            text=f"Tether Length: {self.tether.length:.3f} m",
            textPosition=text_pos,
            textColorRGB=[1, 0, 0],  # White text
            textSize=2,
            replaceItemUniqueId=1  # Replace previous text
            )

            # Display Weight Mass
            p.addUserDebugText(
                text=f"Weight Mass: {self.weight.mass_weight} kg",
                textPosition=[text_pos[0], text_pos[1], text_pos[2] - 0.1],  # Slightly below tether length text
                textColorRGB=[1, 0, 0],
                textSize=2,
                replaceItemUniqueId=2  # Replace previous text
            )
        
    def get_drone_currrent_pos(self):
        self.drone_cur_pos = self._getDroneStateVector(0)[0:3]
        return self.drone_cur_pos
    
    def _distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
      