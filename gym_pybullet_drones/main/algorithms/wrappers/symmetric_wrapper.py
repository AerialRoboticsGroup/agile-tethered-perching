import gymnasium as gym
from typing import Dict, Any, Tuple
import numpy as np
'''reference:https://github.com/TommyWoodley/TommyWoodleyMEngProject'''

class SymmetricWrapper(gym.Wrapper):

    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = env.action_space
        # print(f"symm:{self.action_space}")
        self.observation_space = env.observation_space
        # print(self.observation_space)
        self.positive = True
        self.NUM_DRONES =  env.get_wrapper_attr('NUM_DRONES')

    def step(self, action: np.ndarray, num_wraps=None) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        action = np.reshape(action, (self.NUM_DRONES, -1))
        # print(f"symmatric wrapper action given:{action}")
        # Adjust action
        if self.positive:
            new_action = action[0]
        else:
            x, y, z = action[0]
            new_action = np.reshape(np.array([-1 * x, y, z]),(self.NUM_DRONES, -1))

        # print(f"in symm, the action given is: {action}")
        if num_wraps is None:
            state, reward, terminated, truncated, info = self.env.step(new_action)
        else:
            state, reward, terminated, truncated, info = self.env.step(new_action, num_wraps)
        # print(f"state-symmetric{state}")

        info["original_state"] = state

        if self.positive:
            new_state = state
        else:
            x, y, z, t = state
            new_state = (-1 * x, y, z, t)
        # print(f"symmetric wrapper obs:{new_state}")
        return new_state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None, branch_pos=None, variable=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position, branch_pos, variable=variable)
        x, y, z, t = state  # Do we need this line?
        if x >= 0:
            self.positive = True
        else:
            self.positive = False

        info["original_state"] = state

        if self.positive:
            new_state = state
        else:
            x, y, z, t = state
            new_state = (-1 * x, y, z, t)
        return new_state, info
