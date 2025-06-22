import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging rewards and additional values in TensorBoard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []  # To store rewards for each episode
        self.episode_lengths = []  # To store episode lengths
        self.current_episode_reward = 0.0  # To accumulate reward for the current episode
        self.episode_count = 0  # Track the number of episodes

    def _on_step(self) -> bool:
        # Get the reward for the current timestep
        if 'rewards' in self.locals:
            reward = self.locals['rewards']
            self.current_episode_reward += np.mean(reward)  # Accumulate rewards

        # Check if the episode has ended
        done = self.locals.get('dones')  # Check if the current timestep is the end of an episode
        if done and np.any(done):  # If at least one environment is done
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.logger.record("train/episode_mean_reward", np.mean(self.episode_rewards))  # Log the mean episode reward

            # Reset the current episode reward for the next episode
            self.current_episode_reward = 0.0

        return True
