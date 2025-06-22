from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer


class DualReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        weighting: int = 2,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        
        
        self.online_replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, device, n_envs,
                                                 optimize_memory_usage, handle_timeout_termination)
        self.offline_replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, device, n_envs,
                                                  optimize_memory_usage, handle_timeout_termination)
        self.weighting = weighting
        print(f"Dual Buffer Weighting {weighting}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self._add_online(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        ideal_online_batch_size = batch_size // self.weighting
        actual_online_batch_size = min(ideal_online_batch_size, self.online_replay_buffer.size())
        offline_batch_size = batch_size - actual_online_batch_size

        online_samples = self.online_replay_buffer.sample(actual_online_batch_size, env)
        offline_samples = self.offline_replay_buffer.sample(offline_batch_size, env)

        combined_samples = self._concat_samples(online_samples, offline_samples)
        assert len(combined_samples.observations) == batch_size, f"was {len(combined_samples.observations)}"

        return combined_samples

    def _add_online(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.online_replay_buffer.add(obs, next_obs, action, reward, done, infos)

    def _add_offline(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.offline_replay_buffer.add(obs, next_obs, action, reward, done, infos)

    @staticmethod
    def _concat_samples(first_sample: ReplayBufferSamples, second_sample: ReplayBufferSamples) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            observations=th.cat((first_sample.observations, second_sample.observations)),
            actions=th.cat((first_sample.actions, second_sample.actions)),
            next_observations=th.cat((first_sample.next_observations, second_sample.next_observations)),
            dones=th.cat((first_sample.dones, second_sample.dones)),
            rewards=th.cat((first_sample.rewards, second_sample.rewards)),
        )

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()
