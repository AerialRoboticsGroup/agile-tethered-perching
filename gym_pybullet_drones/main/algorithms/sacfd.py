from typing import Any, Dict, Tuple, Union
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from torch.nn import functional as F
from torch._C import device
import torch
import torch as th


class SACfD(SAC):
    def __init__(
            self,
            policy: str | type[SACPolicy],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: int | Tuple[int, str] = 1,
            gradient_steps: int = 1,
            action_noise: ActionNoise | None = None,
            replay_buffer_class: type[ReplayBuffer] | None = None,
            replay_buffer_kwargs: Dict[str, Any] | None = None,
            optimize_memory_usage: bool = False,
            ent_coef: str | float = "auto",
            target_update_interval: int = 1,
            target_entropy: str | float = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: str | None = None,
            policy_kwargs: Dict[str, Any] | None = None,
            verbose: int = 0,
            seed: int | None = None,
            device: device | str = "auto",
            _init_setup_model: bool = True):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model)
 