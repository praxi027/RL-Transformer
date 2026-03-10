import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(episode_rewards, window=50, title="Training Rewards", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_success_rate(successes, window=50, title="Success Rate", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(successes) >= window:
        rates = np.convolve([float(s) for s in successes], np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(successes)), rates)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
