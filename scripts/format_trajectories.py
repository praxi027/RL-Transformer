import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from src.format import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Format DQN trajectories for ICRL training")
    parser.add_argument("--trajectory-dir", type=str, default="data/trajectories")
    parser.add_argument("--output", type=str, default="data/formatted/mid.jsonl")
    parser.add_argument("--num-slices", type=int, default=10_000)
    parser.add_argument("--quality", type=str, default="mid", choices=["high", "mid", "low"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total_tokens = 0
    total_sets = 0
    total_episodes = 0

    with open(args.output, "w") as f:
        for sl in tqdm(
            build_dataset(
                args.trajectory_dir,
                args.num_slices,
                quality=args.quality,
                max_tokens=args.max_tokens,
                seed=args.seed,
            ),
            total=args.num_slices,
            desc=f"Building slices ({args.quality})",
        ):
            f.write(json.dumps(sl) + "\n")
            total_tokens += sl["num_tokens_approx"]
            total_sets += sl["num_sets"]
            total_episodes += sl["num_episodes"]

    n = args.num_slices
    print(f"\nDone. {n} slices written to {args.output}")
    print(f"Avg tokens/slice: {total_tokens / n:.0f}")
    print(f"Avg sets/slice: {total_sets / n:.1f}")
    print(f"Avg episodes/slice: {total_episodes / n:.1f}")


if __name__ == "__main__":
    main()
