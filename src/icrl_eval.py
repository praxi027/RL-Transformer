import random

import torch
from transformers import AutoTokenizer

from src.frozen_lake import make_env
from src.icrl_model import ICRLModel, ACTION_IDS_ORDERED, TOKEN_ID_TO_IDX
from src.tokenize_data import MODEL_ID, ACTION_TOKEN_IDS
from src.format import BOT, EOT, SHI, EHI, EID, ACTION_MAP


def build_context_tokens(tokenizer, history):
    """Convert episode history into token IDs for the model."""
    text = BOT
    for step in history:
        text += f"{SHI}observation{EHI}\n{step['obs']}{EID}"
        text += f"{SHI}action{EHI}\n{ACTION_MAP[step['action']]}{EID}"
        if step.get("reward", 0.0) != 0.0:
            text += f"{SHI}reward{EHI}\n{step['reward']}{EID}"
    # End previous episode, start new observation prompt
    if history:
        text += EOT
    return text


def build_prompt_for_action(tokenizer, episodes_text, current_obs):
    """Build the full prompt: past episodes + current observation.

    Returns token IDs ready for the model, ending right before
    the action content token so the model's next-token logits
    give us Q-values for the 4 actions.
    """
    # Current step: observation header + obs number + action header
    # The model will predict the action word as the next token
    current = f"{SHI}observation{EHI}\n{current_obs}{EID}{SHI}action{EHI}\n"
    full_text = episodes_text + current
    return tokenizer.encode(full_text, add_special_tokens=False)


def select_action(model, tokenizer, token_ids, epsilon, rng):
    """Select an action using epsilon-greedy over the model's Q-values."""
    if rng.random() >= epsilon:
        return rng.randint(0, 3)

    input_ids = torch.tensor([token_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model.forward(input_ids, attention_mask)

    # Last position's logits predict the next token (the action word)
    q_values = logits[0, -1, model.action_ids]  # (4,)
    return q_values.argmax().item()


def run_episode(model, tokenizer, env, episodes_text, epsilon, rng, max_steps=200):
    """Run one episode, returning the trajectory and updated context."""
    obs, _ = env.reset()
    history = []

    for _ in range(max_steps):
        token_ids = build_prompt_for_action(tokenizer, episodes_text, obs)
        action = select_action(model, tokenizer, token_ids, epsilon, rng)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        history.append({"obs": int(obs), "action": action, "reward": float(reward)})
        obs = next_obs

        if done:
            break

    # Build text for this completed episode to add to context
    ep_text = BOT
    for step in history:
        ep_text += f"{SHI}observation{EHI}\n{step['obs']}{EID}"
        ep_text += f"{SHI}action{EHI}\n{ACTION_MAP[step['action']]}{EID}"
        if step["reward"] != 0.0:
            ep_text += f"{SHI}reward{EHI}\n{step['reward']}{EID}"
    ep_text += EOT

    total_reward = sum(s["reward"] for s in history)
    return history, ep_text, total_reward


def evaluate_map(
    model, tokenizer, size, num_episodes=30, warmup_episodes=20,
    is_slippery=False, max_episode_steps=200, seed=None,
):
    """Evaluate on a single map for num_episodes with epsilon-greedy warmup."""
    rng = random.Random(seed)
    env, desc = make_env(
        size=size, is_slippery=is_slippery, max_episode_steps=max_episode_steps,
    )

    episodes_text = ""
    rewards = []

    for ep in range(num_episodes):
        # Epsilon warmup: 0 -> 1 over first warmup_episodes, then stays at 1
        if ep < warmup_episodes:
            epsilon = ep / warmup_episodes
        else:
            epsilon = 1.0

        _, ep_text, total_reward = run_episode(
            model, tokenizer, env, episodes_text, epsilon, rng,
            max_steps=max_episode_steps,
        )
        episodes_text += ep_text
        rewards.append(total_reward)

    env.close()
    return {"desc": list(desc), "size": size, "rewards": rewards}


def evaluate(
    model, tokenizer, num_maps=50, num_episodes=30,
    map_sizes=(3, 4, 5), seed=42, is_slippery=False,
):
    """Evaluate on multiple random maps. Returns per-map results."""
    rng = random.Random(seed)
    results = []

    for i in range(num_maps):
        size = rng.choice(map_sizes)
        map_seed = rng.randint(0, 100_000)
        result = evaluate_map(
            model, tokenizer, size=size, num_episodes=num_episodes,
            is_slippery=is_slippery, seed=map_seed,
        )
        result["map_id"] = i

        avg_reward = sum(result["rewards"]) / len(result["rewards"])
        late_reward = sum(result["rewards"][-10:]) / 10
        print(
            f"Map {i:3d} ({size}x{size}): "
            f"avg={avg_reward:.2f}  last10={late_reward:.2f}"
        )
        results.append(result)

    return results
