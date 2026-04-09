import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from src.config import DQNConfig, FrozenLakeConfig
from src.collect import train_and_collect, save_map_data


def main():
    parser = argparse.ArgumentParser(description="Collect DQN trajectories on random FrozenLake maps")
    parser.add_argument("--num-maps", type=int, default=250)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="data/trajectories")
    args = parser.parse_args()

    dqn_config = DQNConfig(
        total_timesteps=args.total_timesteps,
        epsilon_decay_steps=args.total_timesteps // 2,
    )
    env_config = FrozenLakeConfig()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    print(f"Using device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []

    for map_id in tqdm(range(args.num_maps), desc="Training maps"):
        seed = args.seed + map_id
        data = train_and_collect(
            map_id, dqn_config, env_config, seed=seed, device=args.device,
        )
        save_map_data(args.out_dir, data)

        manifest.append({
            "map_id": data["map_id"],
            "map_desc": data["map_desc"],
            "map_size": data["map_size"],
            "num_episodes": data["num_episodes"],
            "num_successes": data["num_successes"],
        })

        tqdm.write(
            f"Map {data['map_id']:3d}: {data['map_size']}x{data['map_size']}, "
            f"{data['num_episodes']} eps, "
            f"{data['num_successes']} successes "
            f"({data['num_successes'] / max(data['num_episodes'], 1):.0%})"
        )

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_eps = sum(m["num_episodes"] for m in manifest)
    total_succ = sum(m["num_successes"] for m in manifest)
    print(f"\nDone. {args.num_maps} maps, {total_eps} total episodes, {total_succ} successes")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
