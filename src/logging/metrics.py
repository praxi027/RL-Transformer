import json
from collections import defaultdict


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.step_metrics = defaultdict(list)

    def log_episode(self, reward, length, success):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(bool(success))

    def log_step(self, metrics):
        for k, v in metrics.items():
            self.step_metrics[k].append(v)

    def summary(self, last_n=100):
        if not self.episode_rewards:
            return {}
        recent_r = self.episode_rewards[-last_n:]
        recent_s = self.episode_successes[-last_n:]
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": sum(recent_r) / len(recent_r),
            "success_rate": sum(recent_s) / len(recent_s),
            "mean_length": sum(self.episode_lengths[-last_n:]) / len(recent_r),
        }

    def save(self, path):
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "step_metrics": dict(self.step_metrics),
        }
        with open(path, "w") as f:
            json.dump(data, f)
