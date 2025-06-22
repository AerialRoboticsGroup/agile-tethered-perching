from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import matplotlib.pyplot as plt

class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, plot_rewards=False, render_train = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_rewards = plot_rewards
        self.evaluation_means = []  # Store the mean reward of each evaluation (5 episodes)
        self.evaluation_stds = []   # Store the std dev of each evaluation (5 episodes)
        self.timesteps = []         # Store timesteps at which evaluations are done
        self.render_train = render_train

    def _on_step(self) -> bool:
        if self.render_train:
            self.eval_env.render()
        # Call the super method to handle evaluation logic
        result = super(CustomEvalCallback, self)._on_step()

        # After each evaluation, store the mean and std for the 5 episodes
        if result and len(self.evaluations_results) > 0:
            # The last evaluation's rewards are stored in self.evaluations_results[-1]
            episode_rewards = self.evaluations_results[-1]
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            # Append the mean, std, and current timestep to track progress
            self.evaluation_means.append(mean_reward)
            self.evaluation_stds.append(std_reward)
            self.timesteps.append(self.num_timesteps)

        return result

    def _on_training_end(self) -> None:
        """At the end of training, plot the mean and std for each evaluation."""
        if len(self.evaluation_means) > 0 and self.plot_rewards:
            self._plot_evaluation_rewards()

    def _plot_evaluation_rewards(self):
        """Plot mean and standard deviation for each evaluation step."""
        plt.figure(figsize=(10, 5))

        # Convert lists to numpy arrays for easier plotting
        timesteps = np.array(self.timesteps)
        means = np.array(self.evaluation_means)
        stds = np.array(self.evaluation_stds)

        # Plot the mean reward with standard deviation as shaded area
        plt.plot(timesteps, means, label="Evaluation Mean Reward", color="blue")
        plt.fill_between(timesteps, means - stds, means + stds, color="blue", alpha=0.2)

        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title("Evaluation Rewards Over Time (Mean ± Std)")
        plt.legend()
        plt.grid()
        plt.show()
