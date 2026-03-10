import os
from datetime import datetime

from src.logging.metrics import MetricsTracker
from src.logging.plotting import plot_rewards, plot_success_rate


class Trainer:
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.metrics = MetricsTracker()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(config.experiment_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def train(self):
        obs, info = self.env.reset()
        ep_reward = 0.0
        ep_length = 0

        for step in range(self.config.total_timesteps):
            action = self.agent.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.agent.store_transition(obs, action, reward, next_obs, done)
            ep_reward += reward
            ep_length += 1
            obs = next_obs

            step_metrics = self.agent.update()
            if step_metrics:
                self.metrics.log_step(step_metrics)

            if done:
                success = info.get("success", False)
                self.metrics.log_episode(ep_reward, ep_length, success)
                obs, info = self.env.reset()
                ep_reward = 0.0
                ep_length = 0

            if step > 0 and step % self.config.log_interval == 0:
                s = self.metrics.summary()
                if s:
                    print(f"Step {step} | Episodes: {s['episodes']} | "
                          f"Reward: {s['mean_reward']:.2f} | "
                          f"Success: {s['success_rate']:.2%} | "
                          f"Eps: {self.agent.epsilon:.3f}")

            if step > 0 and step % self.config.save_interval == 0:
                self.agent.save(os.path.join(self.run_dir, f"checkpoint_{step}.pt"))

        # Final save
        self.agent.save(os.path.join(self.run_dir, "checkpoint_final.pt"))
        self.metrics.save(os.path.join(self.run_dir, "metrics.json"))
        plot_rewards(self.metrics.episode_rewards,
                     save_path=os.path.join(self.run_dir, "rewards.png"))
        plot_success_rate(self.metrics.episode_successes,
                          save_path=os.path.join(self.run_dir, "success_rate.png"))

        print(f"\nTraining complete. Results saved to {self.run_dir}")
        return self.metrics
