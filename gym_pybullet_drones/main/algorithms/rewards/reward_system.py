import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.main.algorithms.rewards.Approaching import CircularApproachingReward
from gym_pybullet_drones.main.algorithms.rewards.Hanging import Hanging


def tanh_wrap_reward(num_wraps, k=2, max_reward=1):
    """
    Smoothly returns a reward between 0 and max_reward based on the number of wraps (num_wraps).
    """
    return (max_reward / 2) * (1 + np.tanh(k * (num_wraps - 1)))


def continuous_distance_penalty(dist_drone_branch, max_dist=0.1, max_penalty=-0.5):
    """
    Provides a smooth distance-based penalty. The closer the drone is to the branch, the smaller the penalty.
    """
    penalty = np.clip(dist_drone_branch / max_dist, 0, 1) * max_penalty
    return penalty


class RewardSystem():
    def __init__(self, branch_pos, tether_length) -> None:
        self.approaching_reward = CircularApproachingReward(branch_pos, tether_length)
        self.hanging_reward = Hanging(branch_pos, tether_length)

        # Initialize tracking lists for plotting
        self.approaching_rewards = []
        self.wrapping_rewards = []
        self.hanging_rewards = []
        self.distance_rewards = []
        self.total_rewards = []
        self.num_wraps_list = []

        # Other relevant settings
        self.self_done_states = False
        
        self.branch_pos = branch_pos
        self.tether_length = tether_length
        self.previous_num_rotations = 0

    def calculate(self, state, has_contacted, dist_tether_branch, dist_drone_branch, num_wraps, weight_pos):
        """
        Calculate the total reward based on the drone's progress in wrapping, hanging, and avoiding collisions.

        Args:
            state: The drone's current position.
            has_contacted: Whether the tether has contacted the branch.
            dist_tether_branch: Distance between the tether and the branch.
            dist_drone_branch: Distance between the drone and the branch.
            num_wraps: The number of wraps around the branch.

        Returns:
            total_reward: The combined reward based on progress.
        """
        ### 1. Approaching reward - focuses on getting close to the branch
        approaching_reward = self.approaching_reward.reward_fun(state, has_contacted, dist_tether_branch, dist_drone_branch, num_wraps,weight_pos)
        assert -1 <= approaching_reward <= 1, f"Invalid approaching_reward: {approaching_reward}"


        ### 2. Wrapping reward - smoothly increases with number of wraps
        wrapping_reward = tanh_wrap_reward(num_wraps, k=2, max_reward=1) 
        assert 0 <= wrapping_reward <= 1, f"Invalid wrapping_reward: {wrapping_reward}"

        ### 3. Hanging reward - increases when drone hangs successfully
        hanging_reward, hanging_done = self.hanging_reward.reward_fun(state)

        assert -1 <= hanging_reward <= 1, f"Invalid hanging_reward: {hanging_reward}"

        ### 4. Distance penalty - softly penalizes the drone for being too close to the branch
        distance_reward = continuous_distance_penalty(dist_drone_branch)
        assert -0.5 <= distance_reward <= 0, f"Invalid distance_reward: {distance_reward}"

        # if num_wraps > 0:
        #     print(f"num:{num_wraps}")
        
        total_reward = 0
        
        if num_wraps > 0.5:
            # 1 + (0~2) = 1~3
            # print("++++++++++++wrapping+++++++-")  
            total_reward = 2 + 2*wrapping_reward + distance_reward #-1,1+ 1,2, -1,1 = 1,3
    
            if num_wraps > 1.0:
                # print("##############################")
                # print("########hanging#############")
                # (-1,1) + 1~3 = 0,2 ~ 2,4 => 0,4
                total_reward +=  hanging_reward
                done = hanging_done
            else:
                done = False
                
                
            # print(done)
        else:
            # print("----------approaching--------------------")
            total_reward = 2*approaching_reward + wrapping_reward # -1,1 + 0,0.5 -1,1.5
            done = False
        
        
        
        self.previous_num_rotations = num_wraps
        # total_reward = approaching_reward + wrapping_reward
        # total_reward = np.clip(total_reward, -1, 1)  # Normalize rewards
        
        # print(total_reward)
        
        
        ### Track and save reward components for analysis
        self.approaching_rewards.append(approaching_reward)
        self.wrapping_rewards.append(wrapping_reward)
        self.hanging_rewards.append(hanging_reward)
        # self.distance_rewards.append(distance_reward)
        # self.num_wraps_list.append(num_wraps)
        self.total_rewards.append(total_reward)

        # self.self_done_states = done
        return total_reward

    def refer_terminated(self):
        """Return the done state for termination checks."""
        return self.self_done_states

    def reset(self):
        """Reset reward tracking and plot the results."""
        self.self_done_states = False

        # plt.figure(figsize=(14, 7))
        # plt.plot(self.total_rewards, label='Total Reward', marker='o', linestyle='--', alpha=0.2)
        # # plt.plot(self.num_wraps_list, label='No. Wraps', marker='o', linestyle='--', alpha=0.2)
        # plt.plot(self.approaching_rewards, label='approaching', marker='o', linestyle='--', alpha=0.2)
        # plt.plot(self.hanging_rewards, label='hanging', marker='o', linestyle='--', alpha=0.2)
        # plt.plot(self.wrapping_rewards, label='wrapping', marker='o', linestyle='--', alpha=0.2)
        # plt.xlabel('Time Steps')
        # plt.ylabel('Reward Value')
        # plt.title('Reward Components Over Time')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Clear reward logs
        self.approaching_rewards = []
        self.wrapping_rewards = []
        self.hanging_rewards = []
        self.distance_rewards = []
        self.total_rewards = []
        self.num_wraps_list = []

