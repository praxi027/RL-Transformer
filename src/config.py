from dataclasses import dataclass


@dataclass
class DQNConfig:
    gamma: float = 0.9
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    target_update_alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    hidden_dims: tuple = (128, 128)
    total_timesteps: int = 100_000
    log_interval: int = 1_000
    save_interval: int = 25_000


@dataclass
class FrozenLakeConfig:
    map_sizes: tuple = (3, 4, 5)
    ood_map_sizes: tuple = (6, 7)
    is_slippery: bool = False
    max_episode_steps: int = 200
    reward_scale: float = 30.0
