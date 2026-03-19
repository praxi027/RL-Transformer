import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

from src.icrl_model import ICRLModel
from src.tokenize_data import MODEL_ID


class TrajectoryDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, weights_only=True)
        self.n = self.data["input_ids"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def train_icrl(
    dataset_path,
    output_dir,
    model_id=MODEL_ID,
    device="cuda",
    lr=1e-2,
    warmup_steps=10,
    batch_size=10,
    micro_batch_size=1,
    gamma=0.9,
    alpha=0.1,
    num_epochs=1,
    log_interval=1,
    save_interval=100,
):
    os.makedirs(output_dir, exist_ok=True)

    dataset = TrajectoryDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

    grad_accum_steps = batch_size // micro_batch_size
    print(f"Dataset: {len(dataset)} slices")
    print(f"Batch size: {batch_size} (micro={micro_batch_size}, accum={grad_accum_steps})")
    print(f"Steps per epoch: {len(dataset) // batch_size}")

    model = ICRLModel(model_id=model_id, device=device, alpha=alpha, gamma=gamma)
    model.model.print_trainable_parameters()

    optimizer = torch.optim.Adam(model.trainable_params(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
    )

    log_path = os.path.join(output_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    step = 0
    accum_count = 0
    accum_loss = 0.0
    accum_info = {}

    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for micro_batch in loader:
            loss, info = model.compute_loss(micro_batch)

            if loss.item() == 0.0:
                continue

            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
            for k, v in info.items():
                accum_info[k] = accum_info.get(k, 0.0) + v
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                model.polyak_update()
                step += 1

                avg_loss = accum_loss / grad_accum_steps
                avg_info = {k: v / grad_accum_steps for k, v in accum_info.items()}
                avg_info["lr"] = optimizer.param_groups[0]["lr"]
                avg_info["step"] = step
                avg_info["epoch"] = epoch

                if step % log_interval == 0:
                    print(
                        f"Step {step:4d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Q: {avg_info.get('q_mean', 0):.2f} | "
                        f"Target: {avg_info.get('target_mean', 0):.2f} | "
                        f"LR: {avg_info['lr']:.6f}"
                    )

                log_file.write(json.dumps(avg_info) + "\n")
                log_file.flush()

                accum_loss = 0.0
                accum_info = {}

                if step % save_interval == 0:
                    ckpt_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
                    model.save(ckpt_path)
                    print(f"  Saved {ckpt_path}")

        print(f"Epoch {epoch + 1}/{num_epochs} complete ({step} steps)")

    final_path = os.path.join(output_dir, "checkpoint_final.pt")
    model.save(final_path)
    log_file.close()
    print(f"Training complete. {step} steps. Saved to {final_path}")
