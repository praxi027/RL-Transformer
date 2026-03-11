import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def make_env(size, is_slippery=False, max_episode_steps=200, desc=None):
    if desc is None:
        desc = generate_random_map(size=size)
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=is_slippery,
        max_episode_steps=max_episode_steps,
    )
    return env, desc
