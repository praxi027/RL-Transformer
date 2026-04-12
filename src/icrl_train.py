import json
import os
import re

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.icrl_model import ICRLModel
from src.tokenize_data import MODEL_ID


def _init_distributed():
    """Initialize distributed training if launched with torchrun.

    Returns (world_size, rank, local_rank). If not launched with torchrun,
    returns (1, 0, 0) and does not initialize a process group.
    """
    if "LOCAL_RANK" not in os.environ:
        return 1, 0, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_world_size(), dist.get_rank(), local_rank


class TrajectoryDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, weights_only=True)
        self.n = self.data["input_ids"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def trim_log_to_step(log_path, step):
    if not os.path.exists(log_path):
        return

    kept = []
    with open(log_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("step", 0) <= step:
                kept.append(line)

    with open(log_path, "w") as f:
        f.writelines(kept)


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
    load_in_4bit=False,
    load_in_8bit=False,
    gradient_checkpointing=False,
    resume=None,
):
    world_size, rank, local_rank = _init_distributed()
    is_main = rank == 0
    distributed = world_size > 1
    if distributed:
        device = f"cuda:{local_rank}"

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if batch_size % (world_size * micro_batch_size) != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by "
            f"world_size * micro_batch_size ({world_size} * {micro_batch_size} "
            f"= {world_size * micro_batch_size})"
        )

    dataset = TrajectoryDataset(dataset_path)
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            sampler=sampler,
            pin_memory=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            pin_memory=device.startswith("cuda"),
        )

    grad_accum_steps = batch_size // (world_size * micro_batch_size)
    if is_main:
        print(f"World size: {world_size}")
        print(f"Dataset: {len(dataset)} slices")
        print(
            f"Batch size: {batch_size} "
            f"(per-rank micro={micro_batch_size}, "
            f"per-rank accum={grad_accum_steps}, world={world_size})"
        )
    steps_per_epoch = len(dataset) // batch_size
    total_steps = steps_per_epoch * num_epochs
    if is_main:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total target steps: {total_steps}")
        if resume:
            print(f"Resume checkpoint: {resume}")
        if load_in_4bit:
            print("Quantization: 4-bit NF4")
        elif load_in_8bit:
            print("Quantization: 8-bit")
        else:
            print("Quantization: none")
        print(f"Gradient checkpointing: {gradient_checkpointing}")

    model = ICRLModel(
        model_id=model_id, device=device, alpha=alpha, gamma=gamma,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    if distributed:
        model.wrap_ddp(local_rank)
    if is_main:
        model.model.print_trainable_parameters()

    optimizer = torch.optim.Adam(model.trainable_params(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
    )

    step = 0
    start_epoch = 0
    accum_count = 0
    accum_loss = 0.0
    accum_info = {}

    if resume:
        ckpt = model.load(resume)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        loaded_scheduler = False
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            loaded_scheduler = True
        if "step" in ckpt:
            step = ckpt["step"]
        else:
            match = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(resume))
            if match:
                step = int(match.group(1))
        if step and not loaded_scheduler:
            if step >= warmup_steps:
                factor = 1.0
                scheduler.last_epoch = warmup_steps
            else:
                factor = 1e-8 + (1.0 - 1e-8) * (step / warmup_steps)
                scheduler.last_epoch = step
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * factor
        start_epoch = min(step // max(steps_per_epoch, 1), num_epochs)
        if is_main:
            print(f"Resumed from step {step}, epoch {start_epoch}")

    log_path = os.path.join(output_dir, "train_log.jsonl")
    log_file = None
    if is_main:
        if resume:
            trim_log_to_step(log_path, step)
        log_file = open(log_path, "a" if resume else "w")

    optimizer.zero_grad()

    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for micro_batch in loader:
            if step >= total_steps:
                break
            loss, info = model.compute_loss(micro_batch)

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

                if is_main and step % log_interval == 0:
                    print(
                        f"Step {step:4d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Q: {avg_info.get('q_mean', 0):.2f} | "
                        f"Target: {avg_info.get('target_mean', 0):.2f} | "
                        f"LR: {avg_info['lr']:.6f}"
                    )

                if is_main:
                    log_file.write(json.dumps(avg_info) + "\n")
                    log_file.flush()

                accum_loss = 0.0
                accum_info = {}

                if step % save_interval == 0:
                    if is_main:
                        ckpt_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
                        model.save(
                            ckpt_path,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            step=step,
                            epoch=epoch,
                        )
                        print(f"  Saved {ckpt_path}")
                    if distributed:
                        dist.barrier()
        if step >= total_steps:
            break

        if is_main:
            print(f"Epoch {epoch + 1}/{num_epochs} complete ({step} steps)")

    final_path = os.path.join(output_dir, "checkpoint_final.pt")
    if is_main:
        model.save(
            final_path,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            step=step,
            epoch=num_epochs,
        )
        log_file.close()
        print(f"Training complete. {step} steps. Saved to {final_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
