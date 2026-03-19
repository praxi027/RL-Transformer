import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.icrl_train import train_icrl


def main():
    parser = argparse.ArgumentParser(description="Train ICRL (LLaMA + IA3 + DQN loss)")
    parser.add_argument("--dataset", type=str, default="data/tokenized/dataset.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/icrl")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.1, help="Polyak averaging")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    train_icrl(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gamma=args.gamma,
        alpha=args.alpha,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
