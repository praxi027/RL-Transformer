# RL + Transformer: In-Context Reinforcement Learning (ICRL)

Reproduction and exploration of ["RL + Transformer = A General-Purpose Problem Solver"](https://arxiv.org/abs/2501.14176) (Rentschler & Roberts, 2025).

## Overview

Fine-tune a pre-trained LLM (LLaMA 3.1 8B Instruct) with Deep Q-Network (DQN) reinforcement learning to develop **In-Context Reinforcement Learning (ICRL)** — an emergent ability where the transformer meta-learns to solve new problems through in-context experience, without weight updates at inference time.

## Key Components

- **Model**: LLaMA 3.1 8B Instruct with IA3 adapters (parameter-efficient fine-tuning)
- **RL Algorithm**: DQN with Polyak averaging (α=0.1), discount factor γ=0.9, rewards scaled ×30
- **Environment**: Frozen Lake (Gymnasium) — parametric grid world with discrete states/actions
- **Data Format**: Conversational format using `action`, `observation`, `reward` roles (not user/assistant)
- **Training Data**: 250 different Frozen Lake parameterizations, episodes randomly mixed and concatenated into 4096-token sequences

## Experiments to Reproduce

1. **In-distribution generalization** — Unseen maps from the same distribution (3-5 tile grids)
2. **Out-of-distribution generalization** — Larger maps (6-7 tiles) never seen during training
3. **In-context behavior stitching** — Combining learned skills from separate experiences
4. **Robustness to data quality** — Training on high/mid/low quality trajectories
5. **Non-stationary environments** — Adapting when the environment changes mid-evaluation

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── data/               # Training data generation & storage
├── environments/       # Frozen Lake environment wrappers
├── models/             # IA3 adapter & transformer model code
├── training/           # DQN training loop
├── evaluation/         # Evaluation scripts & metrics
├── notebooks/          # Exploration notebooks
├── requirements.txt    # Python dependencies
└── configs/            # Hyperparameter configs
```

## References

- [Paper: arXiv 2501.14176](https://arxiv.org/abs/2501.14176)
- [Gymnasium Frozen Lake](https://gymnasium.farama.org/)
- [IA3: Few-shot Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2205.05638)
- [DQN: Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602)
