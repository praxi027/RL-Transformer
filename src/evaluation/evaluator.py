from src.logging.metrics import MetricsTracker


def evaluate(agent, env, num_episodes=100):
    metrics = MetricsTracker()

    for _ in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        metrics.log_episode(ep_reward, ep_length, info.get("success", False))

    summary = metrics.summary(last_n=num_episodes)
    return summary, metrics
