import os
import json
import random

import numpy as np
import torch

from src.frozen_lake import make_env
from src.dqn import DQNAgent
from src.train import one_hot, set_seed


def train_and_collect(map_id, dqn_config, env_config, seed=42, device="cpu"):
    set_seed(seed)

    size = random.choice(env_config.map_sizes)
    env, desc = make_env(
        size=size,
        is_slippery=env_config.is_slippery,
        max_episode_steps=env_config.max_episode_steps,
    )
    obs_dim = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, dqn_config, device=device)

    episodes = []
    current_steps = []
    obs_raw, _ = env.reset(seed=seed)
    obs = one_hot(obs_raw, obs_dim)

    for step in range(dqn_config.total_timesteps):
        action = agent.select_action(obs, explore=True)
        next_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        current_steps.append({
            "obs": int(obs_raw),
            "action": int(action),
            "reward": float(reward),
        })

        scaled_reward = reward * env_config.reward_scale
        next_obs = one_hot(next_raw, obs_dim)
        agent.store(obs, action, scaled_reward, next_obs, done)
        agent.update()

        if done:
            success = terminated and reward > 0
            episodes.append({
                "steps": current_steps,
                "success": success,
            })
            current_steps = []
            obs_raw, _ = env.reset()
            obs = one_hot(obs_raw, obs_dim)
        else:
            obs_raw = next_raw
            obs = next_obs

    env.close()

    n_success = sum(ep["success"] for ep in episodes)
    return {
        "map_id": map_id,
        "map_desc": list(desc),
        "map_size": size,
        "num_episodes": len(episodes),
        "num_successes": n_success,
        "episodes": episodes,
    }


def save_map_data(out_dir, data):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"map_{data['map_id']:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path
