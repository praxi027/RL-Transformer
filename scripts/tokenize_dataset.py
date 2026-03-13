import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tokenize_data import tokenize_dataset


def main():
    parser = argparse.ArgumentParser(description="Tokenize formatted trajectories for ICRL training")
    parser.add_argument("--input", type=str, default="data/formatted/mid.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/tokenized")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    print(f"Tokenizing {args.input} ...")
    stats = tokenize_dataset(args.input, args.output_dir, args.max_length)

    print(f"\nDone. Saved to {args.output_dir}/dataset.pt")
    print(f"  Slices: {stats['num_slices']}")
    print(f"  Total action positions: {stats['total_action_positions']}")
    print(f"  Total terminal positions: {stats['total_terminal_positions']}")
    print(f"  Avg actions/slice: {stats['avg_actions_per_slice']:.1f}")


if __name__ == "__main__":
    main()
