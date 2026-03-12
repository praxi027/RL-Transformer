import json
import os
import random
import re

ACTION_MAP = {0: "left", 1: "down", 2: "right", 3: "up"}

BOT = "<|begin_of_text|>"
EOT = "<|end_of_text|>"
SHI = "<|start_header_id|>"
EHI = "<|end_header_id|>"
EID = "<|eot_id|>"

SPECIAL_TOKENS = [BOT, EOT, SHI, EHI, EID]

QUALITY_WEIGHTS = {
    "high": (5, 1),  # (success_weight, failure_weight)
    "mid": (1, 1),
    "low": (1, 5),
}


def format_step(obs, action, reward=None):
    text = f"{SHI}observation{EHI}\n{obs}{EID}"
    text += f"{SHI}action{EHI}\n{ACTION_MAP[action]}{EID}"
    if reward is not None and reward != 0.0:
        text += f"{SHI}reward{EHI}\n{reward}{EID}"
    return text


def format_episode(episode):
    steps = episode["steps"]
    text = BOT
    for step in steps:
        text += format_step(step["obs"], step["action"], step["reward"])
    text += EOT
    return text


def count_tokens(text):
    remaining = text
    count = 0
    for tok in SPECIAL_TOKENS:
        n = remaining.count(tok)
        count += n
        remaining = remaining.replace(tok, "")
    for part in remaining.split("\n"):
        part = part.strip()
        if part:
            count += 1
    return count


def sample_set(episodes, set_size, quality, rng):
    successes = [ep for ep in episodes if ep["success"]]
    failures = [ep for ep in episodes if not ep["success"]]

    if not successes or not failures:
        selected = rng.choices(episodes, k=set_size)
    else:
        sw, fw = QUALITY_WEIGHTS[quality]
        weights = [sw if ep["success"] else fw for ep in episodes]
        selected = rng.choices(episodes, weights=weights, k=set_size)

    rng.shuffle(selected)
    return selected


def format_set(episodes):
    return "".join(format_episode(ep) for ep in episodes)


def build_slice(map_pool, max_tokens, set_size_range, quality, rng):
    slice_text = ""
    token_count = 0
    sets_used = 0
    episodes_used = 0

    while True:
        map_data = rng.choice(map_pool)
        set_size = rng.randint(set_size_range[0], set_size_range[1])
        episodes = sample_set(map_data["episodes"], set_size, quality, rng)
        set_text = format_set(episodes)
        set_tokens = count_tokens(set_text)

        if token_count + set_tokens > max_tokens and sets_used > 0:
            break

        slice_text += set_text
        token_count += set_tokens
        sets_used += 1
        episodes_used += set_size

        if token_count >= max_tokens * 0.85:
            break

    return {
        "text": slice_text,
        "num_tokens_approx": token_count,
        "num_sets": sets_used,
        "num_episodes": episodes_used,
        "quality": quality,
    }


def load_map_pool(trajectory_dir):
    manifest_path = os.path.join(trajectory_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    pool = []
    for entry in manifest:
        map_path = os.path.join(trajectory_dir, f"map_{entry['map_id']:04d}.json")
        with open(map_path) as f:
            data = json.load(f)
        pool.append({"map_id": data["map_id"], "episodes": data["episodes"]})

    return pool


def build_dataset(trajectory_dir, num_slices, quality="mid", max_tokens=4096,
                  set_size_range=(20, 40), seed=42):
    pool = load_map_pool(trajectory_dir)
    rng = random.Random(seed)

    for _ in range(num_slices):
        yield build_slice(pool, max_tokens, set_size_range, quality, rng)
