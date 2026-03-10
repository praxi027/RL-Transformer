from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    total_timesteps: int = 100_000
    eval_interval: int = 5_000
    eval_episodes: int = 20
    log_interval: int = 1_000
    save_interval: int = 25_000
    experiment_dir: str = "experiments"
    device: str = "cpu"


@dataclass
class EvalConfig:
    seed: int = 0
    num_episodes: int = 100
    render: bool = False
    checkpoint_path: str = ""
