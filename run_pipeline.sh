#!/bin/bash
set -e

# Re-launch inside a tmux session if not already in one
if [ -z "$TMUX" ]; then
    tmux new-session -d -s icrl "bash $0"
    echo "Started in tmux session 'icrl'. Attach with: tmux attach -t icrl"
    exit 0
fi

echo "=== Step 0: Setup ==="
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

if [ ! -f venv/.deps_installed ]; then
    pip install --upgrade pip
    pip install torch gymnasium numpy matplotlib tqdm
    pip install transformers peft accelerate datasets bitsandbytes
    touch venv/.deps_installed
fi

echo "=== Step 1: Collect trajectories (250 maps) ==="
if [ ! -f data/trajectories/manifest.json ]; then
    python scripts/collect_trajectories.py --num-maps 250 --device cuda --total-timesteps 30000
else
    echo "Skipping - data/trajectories/manifest.json already exists"
fi

echo "=== Step 2: Format trajectories ==="
if [ ! -f data/formatted/mid.jsonl ]; then
    python scripts/format_trajectories.py --output data/formatted/mid.jsonl
else
    echo "Skipping - data/formatted/mid.jsonl already exists"
fi

echo "=== Step 3: Tokenize ==="
if [ ! -f data/tokenized/dataset.pt ]; then
    python scripts/tokenize_dataset.py --input data/formatted/mid.jsonl --output-dir data/tokenized
else
    echo "Skipping - data/tokenized/dataset.pt already exists"
fi

echo "=== Step 4: Train ICRL (DDP across all GPUs) ==="
if [ ! -f experiments/icrl/checkpoint_final.pt ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Launching torchrun with ${NUM_GPUS} GPUs"
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/train_icrl.py \
        --micro-batch-size 1 \
        --batch-size 8
else
    echo "Skipping - experiments/icrl/checkpoint_final.pt already exists"
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "=== Step 5: Evaluate (in-distribution) ==="
torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/eval_icrl.py \
    --checkpoint experiments/icrl/checkpoint_final.pt \
    --num-maps 50 \
    --map-sizes 3,4,5

echo "=== Step 5b: Evaluate (out-of-distribution) ==="
torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/eval_icrl.py \
    --checkpoint experiments/icrl/checkpoint_final.pt \
    --num-maps 50 \
    --map-sizes 6,7 \
    --output-dir experiments/eval_ood

echo "=== Done ==="
echo "Results in experiments/eval/reward_curve.png and experiments/eval_ood/reward_curve.png"
