#!/bin/bash
set -e

# Re-launch inside a tmux session if not already in one
if [ -z "$TMUX" ]; then
    tmux new-session -d -s icrl "bash $0"
    echo "Started in tmux session 'icrl'. Attach with: tmux attach -t icrl"
    exit 0
fi

source venv/bin/activate

echo "=== Step 2: Format trajectories ==="
python scripts/format_trajectories.py --output data/formatted/mid.jsonl

echo "=== Step 3: Tokenize ==="
python scripts/tokenize_dataset.py --input data/formatted/mid.jsonl --output-dir data/tokenized

echo "=== Step 4: Train ICRL ==="
python scripts/train_icrl.py --load-in-4bit --micro-batch-size 1

echo "=== Step 5: Evaluate (in-distribution) ==="
python scripts/eval_icrl.py \
    --checkpoint experiments/icrl/checkpoint_final.pt \
    --load-in-4bit \
    --num-maps 50 \
    --map-sizes 3,4,5

echo "=== Step 5b: Evaluate (out-of-distribution) ==="
python scripts/eval_icrl.py \
    --checkpoint experiments/icrl/checkpoint_final.pt \
    --load-in-4bit \
    --num-maps 50 \
    --map-sizes 6,7 \
    --output-dir experiments/eval_ood

echo "=== Done ==="
echo "Results in experiments/eval/reward_curve.png and experiments/eval_ood/reward_curve.png"
