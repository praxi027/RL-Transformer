import os
import json
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.frozen_lake import make_env
from src.dqn import DQNAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(obs, n):
    v = np.zeros(n, dtype=np.float32)
    v[obs] = 1.0
    return v


def train_dqn(dqn_config, env_config, seed=42, device="cpu", desc=None):
    set_seed(seed)

    size = random.choice(env_config.map_sizes) if desc is None else len(desc)
    env, desc = make_env(
        size=size,
        is_slippery=env_config.is_slippery,
        max_episode_steps=env_config.max_episode_steps,
        desc=desc,
    )
    obs_dim = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, dqn_config, device=device)

    # Tracking
    ep_rewards, ep_successes = [], []
    ep_reward, ep_len = 0.0, 0

    obs, info = env.reset(seed=seed)
    obs = one_hot(obs, obs_dim)

    for step in range(dqn_config.total_timesteps):
        action = agent.select_action(obs, explore=True)
        next_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        scaled_reward = reward * env_config.reward_scale
        next_obs = one_hot(next_raw, obs_dim)

        agent.store(obs, action, scaled_reward, next_obs, done)
        ep_reward += scaled_reward
        ep_len += 1
        obs = next_obs

        agent.update()

        if done:
            success = terminated and reward > 0
            ep_rewards.append(ep_reward)
            ep_successes.append(success)
            obs_raw, info = env.reset()
            obs = one_hot(obs_raw, obs_dim)
            ep_reward, ep_len = 0.0, 0

        if step > 0 and step % dqn_config.log_interval == 0 and ep_rewards:
            recent_r = ep_rewards[-100:]
            recent_s = ep_successes[-100:]
            print(f"Step {step} | Ep: {len(ep_rewards)} | "
                  f"Reward: {sum(recent_r)/len(recent_r):.1f} | "
                  f"Success: {sum(recent_s)/len(recent_s):.0%} | "
                  f"Eps: {agent.epsilon:.3f}")

    env.close()
    return agent, ep_rewards, ep_successes, desc


def save_results(run_dir, agent, ep_rewards, ep_successes):
    os.makedirs(run_dir, exist_ok=True)
    agent.save(os.path.join(run_dir, "checkpoint.pt"))

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"rewards": ep_rewards, "successes": ep_successes}, f)

    # Reward plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep_rewards, alpha=0.3, label="Episode reward")
    if len(ep_rewards) >= 50:
        smoothed = np.convolve(ep_rewards, np.ones(50) / 50, mode="valid")
        ax.plot(range(49, len(ep_rewards)), smoothed, label="50-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.savefig(os.path.join(run_dir, "rewards.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Success rate plot
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(ep_successes) >= 50:
        rates = np.convolve([float(s) for s in ep_successes], np.ones(50) / 50, mode="valid")
        ax.plot(range(49, len(ep_successes)), rates)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(os.path.join(run_dir, "success_rate.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
