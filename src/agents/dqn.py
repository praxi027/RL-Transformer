import copy
import random
import torch
import torch.nn.functional as F

from src.agents.base import Agent
from src.agents.replay_buffer import ReplayBuffer


class DQNAgent(Agent):
    def __init__(self, q_network, config, obs_dim, device="cpu"):
        self.config = config
        self.device = device

        self.q_network = q_network.to(device)
        self.target_network = copy.deepcopy(q_network).to(device)
        self.target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_size, obs_dim)

        self.epsilon = config.epsilon_start
        self._epsilon_step = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps
        self.num_actions = q_network.net[-1].out_features

    def select_action(self, obs, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q = self.q_network(obs.unsqueeze(0).to(self.device))
            return q.argmax(dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(obs.numpy(), action, reward, next_obs.numpy(), done)

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        q_values = self.q_network(batch["obs"])
        q_selected = q_values.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(batch["next_obs"]).max(dim=1).values
            target = batch["reward"] + self.config.gamma * next_q * (1 - batch["done"])

        loss = F.mse_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Polyak averaging
        alpha = self.config.target_update_alpha
        with torch.no_grad():
            for p_t, p_o in zip(self.target_network.parameters(), self.q_network.parameters()):
                p_t.data.mul_(1 - alpha).add_(alpha * p_o.data)

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon - self._epsilon_step)

        return {
            "loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "epsilon": self.epsilon,
        }

    def save(self, path):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
