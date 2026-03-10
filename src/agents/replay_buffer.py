import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def store(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "obs": torch.from_numpy(self.obs[indices]).to(device),
            "action": torch.from_numpy(self.actions[indices]).to(device),
            "reward": torch.from_numpy(self.rewards[indices]).to(device),
            "next_obs": torch.from_numpy(self.next_obs[indices]).to(device),
            "done": torch.from_numpy(self.dones[indices]).to(device),
        }

    def __len__(self):
        return self.size
