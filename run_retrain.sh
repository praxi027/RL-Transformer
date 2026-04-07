#!/bin/bash
set -e

# Re-launch inside a tmux session if not already in one
if [ -z "$TMUX" ]; then
    tmux new-session -d -s icrl_retrain "bash $0"
    echo "Started in tmux session 'icrl_retrain'. Attach with: tmux attach -t icrl_retrain"
    exit 0
fi

source venv/bin/activate

echo "=== Step 4: Train ICRL (no quantization, Double DQN) ==="
TRAIN_ARGS=(
    --micro-batch-size 2 \
    --batch-size 10 \
    --lr 1e-2 \
    --output-dir experiments/icrl_fp16
)

if [ ! -f experiments/icrl_fp16/checkpoint_final.pt ]; then
    latest_checkpoint="$(
        find experiments/icrl_fp16 -maxdepth 1 -name 'checkpoint_*.pt' \
            | sed -E 's/.*checkpoint_([0-9]+)\.pt$/\1 &/' \
            | sort -n \
            | tail -n 1 \
            | cut -d' ' -f2-
    )"
    if [ -n "$latest_checkpoint" ]; then
        echo "Resuming from $latest_checkpoint"
        TRAIN_ARGS+=(--resume "$latest_checkpoint")
    fi
fi

python scripts/train_icrl.py "${TRAIN_ARGS[@]}"

echo "=== Step 5a: Evaluate in-distribution (sizes 3-5) ==="
python scripts/eval_icrl.py \
    --checkpoint experiments/icrl_fp16/checkpoint_final.pt \
    --num-maps 50 \
    --map-sizes 3,4,5 \
    --output-dir experiments/eval_id_fp16

echo "=== Step 5b: Evaluate out-of-distribution (sizes 6-7) ==="
python scripts/eval_icrl.py \
    --checkpoint experiments/icrl_fp16/checkpoint_final.pt \
    --num-maps 50 \
    --map-sizes 6,7 \
    --output-dir experiments/eval_ood_fp16

echo "=== Done ==="
echo "Results in experiments/eval_id_fp16/ and experiments/eval_ood_fp16/"
