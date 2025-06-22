import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np
'''reference:https://github.com/TommyWoodley/TommyWoodleyMEngProject'''

class PositionWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = env.action_space

        low = env.observation_space.low[0][:3]
        high = env.observation_space.high[0][:3]
        
        low = np.append(low, 0)
        high = np.append(high, np.inf)
        
        
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_state = None
        env.unwrapped.should_render = False
        self.num_steps = 0
        self.NUM_DRONES = env.get_wrapper_attr('NUM_DRONES')
        
        self.previous_action = None
        self.counter = 0
        
        
    def step(self, action: np.ndarray, num_wraps=None) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        
        action = np.reshape(action, (self.NUM_DRONES, -1))
        
        self.counter += 1
        # print(f"in position, the action given is: {action}")
        # Execute the step in the environment with the intended action
        state, reward, terminated, truncated, info = self._take_single_step(action, num_wraps)
        
        avg_reward = reward

      
        return state, avg_reward, terminated, truncated, info
    
    
    
    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None,branch_pos = None, variable=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position, branch_pos = None, variable=variable)
        self.current_state = state
        self.num_steps = 0
        self.counter = 0
        return state, info


    def _take_single_step(self, action: np.ndarray, num_wraps=None) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # Calculate direction vector
        if num_wraps is None:
            state, reward, terminated, truncated, info = self.env.step(action)
        else:
            state, reward, terminated, truncated, info = self.env.step(action,num_wraps)

        self.current_state = state

        # return state, reward, terminated, truncated or self.num_steps >= 100, info
        return state, reward, terminated, truncated or self.counter >= 300, info