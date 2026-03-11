import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import DQNConfig, FrozenLakeConfig
from src.train import train_dqn, save_results


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

    dqn_config = DQNConfig(
        lr=args.lr,
        gamma=args.gamma,
        target_update_alpha=args.alpha,
        total_timesteps=args.total_timesteps,
    )
    env_config = FrozenLakeConfig(is_slippery=args.slippery)

    agent, ep_rewards, ep_successes, desc = train_dqn(
        dqn_config, env_config, seed=args.seed, device=args.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("experiments", timestamp)
    save_results(run_dir, agent, ep_rewards, ep_successes)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
