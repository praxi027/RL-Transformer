import json
import os

import torch
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

ACTION_TOKEN_IDS = {
    "left": 2414,
    "down": 2996,
    "right": 1315,
    "up": 455,
}
ALL_ACTION_IDS = set(ACTION_TOKEN_IDS.values())

# observation / action / reward role header token IDs
ROLE_ACTION_ID = 1335  # "action"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def find_action_positions(input_ids):
    """Find positions where the model predicts an action word.

    Pattern: <|start_header_id|> action <|end_header_id|> \n ACTION_WORD
    The action word is the position we care about. We detect it by finding
    the role token 'action' (1335) followed by <|end_header_id|> + newline,
    then the next token is the action content position.
    """
    positions = []
    for i in range(len(input_ids) - 3):
        if (input_ids[i] == 128006  # <|start_header_id|>
                and input_ids[i + 1] == ROLE_ACTION_ID  # action
                and input_ids[i + 2] == 128007  # <|end_header_id|>
                and input_ids[i + 3] == 198):  # \n
            if i + 4 < len(input_ids) and input_ids[i + 4] in ALL_ACTION_IDS:
                positions.append(i + 4)
    return positions


def find_reward_after_action(input_ids, action_pos):
    """Look ahead from an action position to find the reward value.

    After an action token, the sequence is:
      <|eot_id|> <|start_header_id|> observation ... or reward ...
    If a reward header follows (before the next observation or episode end),
    reward is 1.0. Otherwise reward is 0.0.
    """
    # Search forward for the next role header
    for j in range(action_pos + 1, min(action_pos + 10, len(input_ids))):
        if input_ids[j] == 128006:  # <|start_header_id|>
            if j + 1 < len(input_ids) and input_ids[j + 1] == 50107:  # "reward"
                return 1.0
            break
    return 0.0


def is_terminal_action(input_ids, action_pos):
    """Check if this action ends the episode.

    Terminal = the next episode marker <|end_of_text|> (128001) or
    a reward header appears before the next observation header.
    Either way, there is no next state for Bellman backup (or only reward).
    """
    for j in range(action_pos + 1, min(action_pos + 15, len(input_ids))):
        if input_ids[j] == 128001:  # <|end_of_text|>
            return True
        if input_ids[j] == 128006:  # <|start_header_id|>
            if j + 1 < len(input_ids):
                if input_ids[j + 1] == 50107:  # "reward" -> terminal (goal reached)
                    return True
                if input_ids[j + 1] == 79060:  # "observation" -> not terminal
                    return False
    return True  # end of sequence


def tokenize_slice(tokenizer, text, max_length=4096):
    """Tokenize one slice and build action mask, rewards, pointers."""
    encoding = tokenizer.encode(text, add_special_tokens=False)

    # Truncate or pad to max_length
    if len(encoding) > max_length:
        encoding = encoding[:max_length]
    pad_len = max_length - len(encoding)
    attention_mask = [1] * len(encoding) + [0] * pad_len
    encoding = encoding + [tokenizer.pad_token_id or 0] * pad_len

    action_positions = find_action_positions(encoding)

    action_mask = [0] * max_length
    rewards = [0.0] * max_length
    next_action_idx = [0] * max_length
    terminal_mask = [0] * max_length

    for k, pos in enumerate(action_positions):
        action_mask[pos] = 1
        rewards[pos] = find_reward_after_action(encoding, pos) * 30.0  # reward scale
        terminal = is_terminal_action(encoding, pos)
        terminal_mask[pos] = 1 if terminal else 0

        # Point to next action position, or self if terminal/last
        if terminal or k + 1 >= len(action_positions):
            next_action_idx[pos] = pos
        else:
            next_action_idx[pos] = action_positions[k + 1]

    return {
        "input_ids": encoding,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
        "rewards": rewards,
        "next_action_idx": next_action_idx,
        "terminal_mask": terminal_mask,
    }


def tokenize_dataset(input_path, output_dir, max_length=4096):
    """Tokenize all slices from a JSONL file and save as torch tensors."""
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_data = {
        "input_ids": [],
        "attention_mask": [],
        "action_mask": [],
        "rewards": [],
        "next_action_idx": [],
        "terminal_mask": [],
    }

    with open(input_path) as f:
        for line in f:
            sl = json.loads(line)
            row = tokenize_slice(tokenizer, sl["text"], max_length)
            for key in all_data:
                all_data[key].append(row[key])

    os.makedirs(output_dir, exist_ok=True)

    # Save as tensors
    tensors = {}
    for key in all_data:
        if key == "rewards":
            tensors[key] = torch.tensor(all_data[key], dtype=torch.float32)
        else:
            tensors[key] = torch.tensor(all_data[key], dtype=torch.long)

    torch.save(tensors, os.path.join(output_dir, "dataset.pt"))

    n = len(all_data["input_ids"])
    n_actions = sum(sum(row) for row in all_data["action_mask"])
    n_terminal = sum(sum(row) for row in all_data["terminal_mask"])
    return {
        "num_slices": n,
        "total_action_positions": n_actions,
        "total_terminal_positions": n_terminal,
        "avg_actions_per_slice": n_actions / max(n, 1),
    }
