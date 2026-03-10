import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seeding import set_global_seed
from src.config.base import TrainConfig
from src.config.dqn_frozen_lake import DQNConfig, FrozenLakeConfig
from src.envs.frozen_lake import FrozenLakeWrapper
from src.networks.mlp import QNetwork
from src.agents.dqn import DQNAgent
from src.training.trainer import Trainer
from src.evaluation.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(description="Train DQN on FrozenLake")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.1, help="Polyak averaging")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--slippery", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)

    train_config = TrainConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        device=args.device,
    )
    dqn_config = DQNConfig(
        lr=args.lr,
        gamma=args.gamma,
        target_update_alpha=args.alpha,
    )
    env_config = FrozenLakeConfig(is_slippery=args.slippery)

    env = FrozenLakeWrapper(env_config)
    q_net = QNetwork(env.observation_dim, env.num_actions, hidden_dims=dqn_config.hidden_dims)
    agent = DQNAgent(q_net, dqn_config, env.observation_dim, device=args.device)

    # Train
    trainer = Trainer(agent, env, train_config)
    metrics = trainer.train()

    # Evaluate in-distribution
    print("\n--- In-Distribution Evaluation ---")
    eval_summary, _ = evaluate(agent, env, num_episodes=50)
    print(f"Success rate: {eval_summary['success_rate']:.2%} | "
          f"Mean reward: {eval_summary['mean_reward']:.2f}")

    # Evaluate out-of-distribution
    print("\n--- Out-of-Distribution Evaluation ---")
    ood_config = FrozenLakeConfig(
        map_sizes=env_config.ood_map_sizes,
        is_slippery=env_config.is_slippery,
    )
    ood_env = FrozenLakeWrapper(ood_config)
    ood_summary, _ = evaluate(agent, ood_env, num_episodes=50)
    print(f"Success rate: {ood_summary['success_rate']:.2%} | "
          f"Mean reward: {ood_summary['mean_reward']:.2f}")

    env.close()
    ood_env.close()


if __name__ == "__main__":
    main()
