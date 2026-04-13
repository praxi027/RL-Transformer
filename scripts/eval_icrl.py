import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.icrl_model import ICRLModel
from src.icrl_eval import evaluate
from src.tokenize_data import load_tokenizer


def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        return 1, 0, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_world_size(), dist.get_rank(), local_rank


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
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--load-in-4bit", action="store_true")
    quant_group.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    world_size, rank, local_rank = init_distributed()
    is_main = rank == 0
    distributed = world_size > 1
    device = f"cuda:{local_rank}" if distributed else args.device

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    map_sizes = tuple(int(s) for s in args.map_sizes.split(","))

    print(f"[rank {rank}] Loading model from {args.checkpoint} ...", flush=True)
    model = ICRLModel(
        model_id=args.model_id,
        device=device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    model.load(args.checkpoint)
    tokenizer = load_tokenizer()

    if is_main:
        print(
            f"Evaluating on {args.num_maps} maps, sizes={map_sizes}, "
            f"world_size={world_size}",
            flush=True,
        )

    local_results = evaluate(
        model, tokenizer,
        num_maps=args.num_maps,
        num_episodes=args.num_episodes,
        map_sizes=map_sizes,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)
        all_results = [r for shard in gathered for r in shard]
    else:
        all_results = local_results

    if is_main:
        all_results.sort(key=lambda r: r["map_id"])

        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        all_rewards = np.array([r["rewards"] for r in all_results])
        mean_rewards = all_rewards.mean(axis=0)

        avg_all = mean_rewards.mean()
        avg_last10 = mean_rewards[-10:].mean()
        print(f"\nOverall avg reward: {avg_all:.3f}")
        print(f"Last 10 episodes avg: {avg_last10:.3f}")

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
        print(f"Saved {args.output_dir}/reward_curve.png")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
