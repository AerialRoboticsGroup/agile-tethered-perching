import numpy as np
import matplotlib.pyplot as plt

class Hanging:
    def __init__(self, branch_pos, tether_length, box_offset=0.3):
        self.branch_pos = branch_pos  
        self.tether_length = tether_length  
        self.box_offset = box_offset  # Offset for safe zone box
        self.max_reward = 1.0  # Maximum reward (when perfectly hanging)
        
    def signed_distance_to_box(self, x, y, z):
        y = 0
        """Calculate signed distance to the edges of the safe zone box."""
        # Define the ideal hanging position
        ideal_hanging_x, ideal_hanging_y = self.branch_pos[0], 0
        ideal_hanging_z = self.branch_pos[2] - self.tether_length / 2
        
        # Box boundaries
        lower_bound_x = ideal_hanging_x - self.box_offset * 2
        upper_bound_x = ideal_hanging_x + self.box_offset * 2
        lower_bound_y = ideal_hanging_y - self.box_offset
        upper_bound_y = ideal_hanging_y + self.box_offset
        lower_bound_z = ideal_hanging_z - self.box_offset
        upper_bound_z = ideal_hanging_z + self.box_offset

        # Compute signed distance (negative if inside the box, positive if outside)
        dist_x = max(0, lower_bound_x - x, x - upper_bound_x)
        dist_y = max(0, lower_bound_y - y, y - upper_bound_y)
        dist_z = max(0, lower_bound_z - z, z - upper_bound_z)
        dist_y = 0
        # Calculate the Euclidean distance to the nearest point on the box
        distance_to_box = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        return distance_to_box

    def reward_fun(self, position):
        """
        Reward function for hanging with a box-shaped safe zone.
        - position: [x, y, z] position of the drone.
        """
        x, y, z = position
        
        # Calculate the distance to the safe zone box
        distance_to_safe_zone = self.signed_distance_to_box(x, y, z)

        # Inside the box: reward between 0 and 1, with max_reward for ideal hanging
        if distance_to_safe_zone <= 0:
            reward = 1.0  # Perfect hanging
        else:
            # Outside the box: reward between -1 and 0, scaling with distance
            reward = -np.tanh(distance_to_safe_zone)  # Negative reward for being outside

        return reward, distance_to_safe_zone <= 0

