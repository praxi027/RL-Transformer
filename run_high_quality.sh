#!/bin/bash
set -e

if [ -z "$TMUX" ]; then
    tmux new-session -d -s icrl_hq "bash $0"
    echo "Queued in tmux session 'icrl_hq'. Attach with: tmux attach -t icrl_hq"
    exit 0
fi

cd /home/ruslan/RL-Transformer
source venv/bin/activate

echo "=== Waiting for existing 'icrl' tmux session to finish ==="
while tmux has-session -t icrl 2>/dev/null; do
    sleep 30
done
echo "'icrl' session ended, starting high-quality pipeline."

echo "=== Step 2: Format trajectories (high quality) ==="
if [ ! -f data/formatted/high.jsonl ]; then
    python scripts/format_trajectories.py --quality high --output data/formatted/high.jsonl
else
    echo "Skipping - data/formatted/high.jsonl already exists"
fi

echo "=== Step 3: Tokenize (high quality) ==="
if [ ! -f data/tokenized_high/dataset.pt ]; then
    python scripts/tokenize_dataset.py --input data/formatted/high.jsonl --output-dir data/tokenized_high
else
    echo "Skipping - data/tokenized_high/dataset.pt already exists"
fi

echo "=== Step 4: Train ICRL (DDP, batch 8, high quality) ==="
if [ ! -f experiments/icrl_high/checkpoint_final.pt ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Launching torchrun with ${NUM_GPUS} GPUs"
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/train_icrl.py \
        --dataset data/tokenized_high/dataset.pt \
        --output-dir experiments/icrl_high \
        --micro-batch-size 1 \
        --batch-size 8
else
    echo "Skipping - experiments/icrl_high/checkpoint_final.pt already exists"
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "=== Step 5: Evaluate (in-distribution) ==="
torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/eval_icrl.py \
    --checkpoint experiments/icrl_high/checkpoint_final.pt \
    --num-maps 50 \
    --map-sizes 3,4,5 \
    --output-dir experiments/eval_high

echo "=== Step 5b: Evaluate (out-of-distribution) ==="
torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/eval_icrl.py \
    --checkpoint experiments/icrl_high/checkpoint_final.pt \
    --num-maps 50 \
    --map-sizes 6,7 \
    --output-dir experiments/eval_high_ood

echo "=== Done ==="
echo "Results in experiments/eval_high/reward_curve.png and experiments/eval_high_ood/reward_curve.png"
