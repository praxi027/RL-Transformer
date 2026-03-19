import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.icrl_model import ICRLModel
from src.icrl_eval import evaluate
from src.tokenize_data import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICRL on unseen FrozenLake maps")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/eval")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-maps", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--map-sizes", type=str, default="3,4,5",
                        help="Comma-separated map sizes (3,4,5 for ID; 6,7 for OOD)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    map_sizes = tuple(int(s) for s in args.map_sizes.split(","))

    print(f"Loading model from {args.checkpoint} ...")
    model = ICRLModel(model_id=args.model_id, device=args.device)
    model.load(args.checkpoint)
    tokenizer = load_tokenizer()

    print(f"Evaluating on {args.num_maps} maps, sizes={map_sizes}")
    results = evaluate(
        model, tokenizer,
        num_maps=args.num_maps,
        num_episodes=args.num_episodes,
        map_sizes=map_sizes,
        seed=args.seed,
    )

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Compute mean reward curve across maps
    all_rewards = np.array([r["rewards"] for r in results])
    mean_rewards = all_rewards.mean(axis=0)

    # Summary
    avg_all = mean_rewards.mean()
    avg_last10 = mean_rewards[-10:].mean()
    print(f"\nOverall avg reward: {avg_all:.3f}")
    print(f"Last 10 episodes avg: {avg_last10:.3f}")

    # Plot (reproduces Figures 3/4 from paper)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(mean_rewards) + 1), mean_rewards, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Cumulative Reward")
    ax.set_title(f"ICRL Evaluation ({args.num_maps} maps, sizes={map_sizes})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(args.output_dir, "reward_curve.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
