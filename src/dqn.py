import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def store(self, obs, action, reward, next_obs, done):
        transition = (obs, action, reward, next_obs, done)
        if len(self.buf) < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, device="cpu"):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            "obs": torch.as_tensor(np.array(obs), dtype=torch.float32, device=device),
            "action": torch.as_tensor(np.array(actions), dtype=torch.long, device=device),
            "reward": torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(np.array(next_obs), dtype=torch.float32, device=device),
            "done": torch.as_tensor(np.array(dones), dtype=torch.float32, device=device),
        }

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, config, device="cpu"):
        self.config = config
        self.device = device
        self.n_actions = n_actions

        self.q_net = QNetwork(obs_dim, n_actions, config.hidden_dims).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.target_net.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_size)

        self.epsilon = config.epsilon_start
        self._eps_step = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps

    def select_action(self, obs, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(obs_t).argmax(dim=1).item()

    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        q_values = self.q_net(batch["obs"])
        q_selected = q_values.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(batch["next_obs"]).max(dim=1).values
            target = batch["reward"] + self.config.gamma * next_q * (1 - batch["done"])

        loss = F.mse_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        alpha = self.config.target_update_alpha
        with torch.no_grad():
            for p_t, p_o in zip(self.target_net.parameters(), self.q_net.parameters()):
                p_t.data.mul_(1 - alpha).add_(alpha * p_o.data)

        self.epsilon = max(self.config.epsilon_end, self.epsilon - self._eps_step)

        return {"loss": loss.item(), "q_mean": q_values.mean().item(), "epsilon": self.epsilon}

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
