import matplotlib.pyplot as plt
import math
from typing import Tuple
import numpy as np
def interpolate_distance(distance, max_dist_value, max_penalty, min_dist_value=0.2, min_penalty=0):
    penalty = min_penalty + ((max_penalty - min_penalty) / (max_dist_value - min_dist_value)) * (distance - min_dist_value)
    return penalty

class CircularApproachingReward():
    def __init__(self, branch_pos, tether_length) -> None:

        self.down_centre_x, self.down_centre_y, self.down_centre_z, self.down_start, self.down_end = 0,0, 2.7, 170, 10
        self.down_centre_x
        self.up_centre_x, self.up_centre_y, self.up_centre_z, self.up_start, self.up_end = 0,0, 4.5, 0, 180
        self.radius = 3
        self.reward_penalty = 3


        self.branch_pos = branch_pos
        self.tether_length = tether_length


        contact_tether_length = (2 / 3 ) * tether_length

        self.target = np.array([
            branch_pos[0] - contact_tether_length * np.cos(np.radians(45)),
            branch_pos[1],
            branch_pos[2] + contact_tether_length * np.sin(np.radians(45))])


        self.collision_threshold = self.tether_length/10  # Threshold to decide whether a collision occurs
        self.contact_timesteps = 0  # To track sustained tether contact
        self.sustained_contact_reward = 0  # Accumulated reward for sustained contact

    def reward_fun(self, state, has_any_tether_contacted, dist_tether_branch, dist_drone_branch, num_wraps, weight_pos) -> Tuple[float, bool, bool]:



        # print(self.target)
        distance = np.linalg.norm(state - self.target)
        reward = 1 * (1 - (distance / 2))  # Normalize
        # distance = np.clip(distance, -1,0)
        distance_reward = np.tanh(reward)

        # 2. Calculate the collision avoidance penalty (penalize drone for being too close to the branch)
        collision_penalty = self.drone_collision_avoidance_reward(dist_drone_branch)

        # 3. Combine penalties by summing them (they are now both penalties, not rewards)
        total_penalty = distance_reward + collision_penalty

        # 4. Calculate the tether contact reward
        tether_contact_reward = self._calc_tether_branch_reward(dist_tether_branch)

        # 5. Update sustained tether contact reward
        self._update_sustained_contact_reward(has_any_tether_contacted)

        # 6. Final reward is the combined penalty, tether contact reward, and sustained contact reward
        final_reward = total_penalty + tether_contact_reward + self.sustained_contact_reward + self._calculate_sector_reward(state)

        if distance < 0.05:
            ring_reward = 1.0  # Layer 5: closest to the target
        elif distance < 0.1:
            ring_reward = 0.75  # Layer 4
        elif distance < 0.25:
            ring_reward = 0.5  # Layer 3
        elif distance < 0.5:
            ring_reward = 0.25  # Layer 2
        elif distance < 1.0:
            ring_reward = 0.1  # Layer 1
        else:
            ring_reward = 0.0  

        final_reward += ring_reward




        # 7. Ensure the total reward is clipped between -5 and 2
        final_reward = np.tanh(final_reward)

        return final_reward




    def _calculate_sector_reward(self, state):
        '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
        x, y, z = state
        #down_center)xyz is the branch position
        is_within, norm_distance = self._within_sector(self.down_centre_x, self.down_centre_y, self.down_centre_z, self.radius,
                                                       self.down_start, self.down_end, x, y, z)
        if is_within:
            return - self.reward_penalty * norm_distance

        is_within, norm_distance = self._within_sector(self.up_centre_x, self.up_centre_y, self.up_centre_z, self.radius,
                                                       self.up_start, self.up_end, x,y, z)
        if is_within:
            return - self.reward_penalty * norm_distance

        return 0.0

    def _within_sector(self, center_x, center_y, center_z, radius, start_angle, end_angle, point_x, point_y,point_z):
        '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
        distance = math.sqrt((point_x - center_x) ** 2 + (point_y - center_y) ** 2 +(point_z - center_z) ** 2)
        # distance = math.sqrt((point_x - center_x) ** 2 +(point_z - center_z) ** 2)

        # Check if the point is within the circle's radius
        if distance > radius:
            return False, None

        # # Calculate the angle from the center to the point in radians
        angle_radians = math.atan2(point_z - center_z, point_x - center_x)
        
        # Convert angle to degrees for easier handling, normalizing to [0, 360)
        angle_degrees = math.degrees(angle_radians) % 360
        start_angle %= 360
        end_angle %= 360
        
        if start_angle > end_angle:
            within = (angle_degrees >= start_angle and angle_degrees <= 360) or (
                angle_degrees >= 0 and angle_degrees <= end_angle)
        else:
            within = (start_angle <= angle_degrees <= end_angle)

        return within, 1 - (distance / radius)


    def _update_sustained_contact_reward(self, has_any_tether_contacted):
    
        if has_any_tether_contacted:
            self.contact_timesteps += 1
            # Incrementally increase the sustained contact reward
            self.sustained_contact_reward = min(1.0, self.sustained_contact_reward + 0.1 * self.contact_timesteps)
        else:
            # Reset sustained contact when tether loses contact
            self.contact_timesteps = 0
            self.sustained_contact_reward = 0

    def drone_collision_avoidance_reward(self, dist_drone_branch):
   
        return self._calc_drone_branch_penalty(dist_drone_branch)

    def _calc_drone_branch_penalty(self, dist_drone_branch: np.ndarray) -> float:
      
        if dist_drone_branch < self.collision_threshold:  # Collision
            return -1.0
        elif dist_drone_branch < self.collision_threshold * 5:  # Near-collision, interpolate penalty
            return interpolate_distance(distance=dist_drone_branch, max_dist_value=self.collision_threshold, max_penalty=-1, min_dist_value=self.collision_threshold * 5, min_penalty=0)
        else:
            return 0

    def _calc_tether_branch_reward(self, dist_tether_branch: np.ndarray) -> float:
       
        if dist_tether_branch < self.collision_threshold / 5:  # Close to the center
            return 1
        elif dist_tether_branch < self.collision_threshold * 2:  # Quite close to the branch
            return interpolate_distance(distance=dist_tether_branch, max_dist_value=self.collision_threshold/5, max_penalty=1, min_dist_value=self.collision_threshold * 2, min_penalty=0)
        else:
            return 0
