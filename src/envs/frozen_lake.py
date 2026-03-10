import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from src.envs.base import EnvWrapper


class FrozenLakeWrapper(EnvWrapper):
    def __init__(self, config):
        self.config = config
        self.max_obs_dim = config.max_obs_dim
        self._env = None
        self._current_size = None

    def reset(self, seed=None):
        size = random.choice(self.config.map_sizes)
        self._current_size = size
        desc = generate_random_map(size=size)

        if self._env is not None:
            self._env.close()

        self._env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            is_slippery=self.config.is_slippery,
            max_episode_steps=self.config.max_episode_steps,
        )
        obs, info = self._env.reset(seed=seed)
        info["map_size"] = size
        info["map"] = desc
        return self._encode_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        scaled_reward = reward * self.config.reward_scale
        info["success"] = terminated and reward > 0
        return self._encode_obs(obs), scaled_reward, terminated, truncated, info

    def _encode_obs(self, obs):
        one_hot = np.zeros(self.max_obs_dim, dtype=np.float32)
        one_hot[obs] = 1.0
        return torch.from_numpy(one_hot)

    @property
    def observation_dim(self):
        return self.max_obs_dim

    @property
    def num_actions(self):
        return 4

    def close(self):
        if self._env is not None:
            self._env.close()
