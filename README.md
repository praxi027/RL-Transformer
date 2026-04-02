# RL + Transformer = A General-Purpose Problem Solver

Reproduction of [arXiv 2501.14176](https://arxiv.org/abs/2501.14176). Fine-tunes LLaMA 3.1 8B Instruct with IA3 adapters using a DQN loss to create an In-Context Reinforcement Learning (ICRL) agent on FrozenLake.

## Requirements

- Python 3.14+
- CUDA GPU with 12+ GB VRAM
- HuggingFace account with access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Setup

```bash
python -m venv venv
source venv/Scripts/activate  # Windows bash

pip install torch gymnasium numpy matplotlib tqdm
pip install transformers peft accelerate datasets bitsandbytes
```

Log in to HuggingFace (needed to download LLaMA):
```bash
python -c "from huggingface_hub import login; login()"
```

## Pipeline

The full pipeline has 5 steps. Run them in order.

### Step 1: Collect DQN Trajectories (~1-2 hours on CPU)

Trains a separate DQN agent on each of 250 random FrozenLake maps (sizes 3-5) and collects all episode trajectories.

```bash
python scripts/collect_trajectories.py --num-maps 250
```

Output: `data/trajectories/map_XXXX.json` + `manifest.json`

### Step 2: Format Trajectories (~seconds)

Converts raw trajectories into LLaMA conversational format with observation/action/reward roles, grouped into sets of 20-40 episodes per map, packed into ~4096-token slices.

```bash
python scripts/format_trajectories.py --output data/formatted/mid.jsonl
```

Output: `data/formatted/mid.jsonl` (JSONL, one slice per line)

Quality variants (for the data quality experiment in Section 4.4):
```bash
python scripts/format_trajectories.py --quality high --output data/formatted/high.jsonl
python scripts/format_trajectories.py --quality low --output data/formatted/low.jsonl
```

### Step 3: Tokenize (~minutes)

Tokenizes slices with the LLaMA tokenizer and builds action masks, reward vectors, and Bellman target pointers.

```bash
python scripts/tokenize_dataset.py --input data/formatted/mid.jsonl --output-dir data/tokenized
```

Output: `data/tokenized/dataset.pt`

### Step 4: Train ICRL (~hours on GPU)

Fine-tunes LLaMA 3.1 8B with IA3 adapters using the DQN Bellman loss. Uses 4-bit quantization to fit in 12-16GB VRAM.

```bash
python scripts/train_icrl.py --load-in-4bit --micro-batch-size 1
```

Key flags:
- `--load-in-4bit`: 4-bit quantization (required for <16GB VRAM)
- `--load-in-8bit`: 8-bit quantization (uses more VRAM than 4-bit, may be slightly simpler to debug)
- `--gradient-checkpointing`: saves VRAM, but usually makes training slower
- `--micro-batch-size 1`: process 1 slice at a time (gradient accumulation handles the rest)
- `--batch-size 10`: effective batch size (paper default)
- `--lr 0.01`: learning rate (paper default)
- `--alpha 0.1`: Polyak averaging factor
- `--num-epochs 1`: number of passes over the dataset

Example training commands:
```bash
# Lowest VRAM footprint
python scripts/train_icrl.py --load-in-4bit --micro-batch-size 1

# If you still hit OOM
python scripts/train_icrl.py --load-in-4bit --micro-batch-size 1 --gradient-checkpointing

# Alternative quantized path
python scripts/train_icrl.py --load-in-8bit --micro-batch-size 1
```

Output: `experiments/icrl/checkpoint_final.pt` + `train_log.jsonl`

### Step 5: Evaluate

Test on unseen FrozenLake maps. Reproduces Figures 3 and 4 from the paper.

In-distribution (sizes 3-5, Figure 3):
```bash
python scripts/eval_icrl.py \
    --checkpoint experiments/icrl/checkpoint_final.pt \
    --load-in-4bit \
    --num-maps 50 \
    --map-sizes 3,4,5
```

Out-of-distribution (sizes 6-7, Figure 4):
```bash
python scripts/eval_icrl.py \
    --checkpoint experiments/icrl/checkpoint_final.pt \
    --load-in-4bit \
    --num-maps 50 \
    --map-sizes 6,7 \
    --output-dir experiments/eval_ood
```

Output: `experiments/eval/reward_curve.png` + `eval_results.json`

## Project Structure

```
src/
  config.py          Dataclass configs (DQN, FrozenLake, Format)
  frozen_lake.py     Gymnasium FrozenLake wrapper
  dqn.py             QNetwork, ReplayBuffer, DQNAgent
  train.py           DQN training loop
  collect.py         Train DQN + collect episode trajectories
  format.py          Format trajectories into LLaMA conversational text
  tokenize_data.py   Tokenize formatted text, build action masks
  icrl_model.py      LLaMA + IA3 with DQN loss and Polyak averaging
  icrl_train.py      ICRL training loop
  icrl_eval.py       Evaluation on unseen maps

scripts/
  collect_trajectories.py   Step 1: DQN trajectory collection
  format_trajectories.py    Step 2: Format for LLM
  tokenize_dataset.py       Step 3: Tokenize
  train_icrl.py             Step 4: Train ICRL
  eval_icrl.py              Step 5: Evaluate

data/                       Generated data (gitignored)
  trajectories/             Raw DQN trajectories per map
  formatted/                JSONL slices in LLaMA format
  tokenized/                Tokenized tensors for training

experiments/                Training outputs (gitignored)
```

## Paper Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | LLaMA 3.1 8B Instruct |
| Adapter | IA3 (keys, values, FFN down projection) |
| Learning rate | 1e-2 |
| LR warmup | Linear, first 10 batches |
| Batch size | 10 slices x 4096 tokens |
| Discount (gamma) | 0.9 |
| Reward scale | 30x |
| Polyak alpha | 0.1 |
| Training maps | 250 (sizes 3-5) |
| Episodes per set | 20-40 |
| Eval maps | 50 |
| Eval episodes | 30 per map |
| Eval epsilon warmup | 0 to 1 over first 20 episodes |

## References

- [Paper: arXiv 2501.14176](https://arxiv.org/abs/2501.14176)
- [Gymnasium Frozen Lake](https://gymnasium.farama.org/)
- [IA3: Few-shot Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2205.05638)
- [DQN: Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602)
