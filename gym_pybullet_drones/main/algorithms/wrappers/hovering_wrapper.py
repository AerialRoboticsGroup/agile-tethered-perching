import gymnasium as gym
from gymnasium import spaces
import numpy as np

'''reference:https://github.com/TommyWoodley/TommyWoodleyMEngProject'''

class HoveringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming the original observation space is a Box of some shape
        low = np.append(env.observation_space.low, 0)
        high = np.append(env.observation_space.high, 1)

        # Update observation space to include one additional integer for the hover count
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
       
        self.episode_count = 0
        # print(self.observation_space.shape)
        self.NUM_DRONES = env.get_wrapper_attr('NUM_DRONES')

    def reset(self, seed=None, options=None, degrees=None, position=None, branch_pos = None, variable=None):
        # Reset the environment and hover count
        obs, info = self.env.reset(seed, options, degrees, position,branch_pos,variable=variable)
        self.episode_count = 0
        # Return the augmented observation
        augmented_obs = np.append(obs, self.get_hover_ratio()).astype(np.float32)
        return augmented_obs, info

    def get_hover_ratio(self):
        return min(self.episode_count, 300.0) / 300.0

    def step(self, action, num_wraps=None):
        
        action = np.reshape(action, (self.NUM_DRONES, -1))
    
        if num_wraps is None:
            obs, reward, done, truncated, info = self.env.step(action)
        else:
            obs, reward, done, truncated, info = self.env.step(action, num_wraps)
       
        self.episode_count += 1
      
        augmented_obs = np.append(obs, self.get_hover_ratio()).astype(np.float32)
  
        return augmented_obs, reward, done, truncated, info
