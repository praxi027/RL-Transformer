import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model, IA3Config

from src.tokenize_data import MODEL_ID, ACTION_TOKEN_IDS

# Ordered: left=0, down=1, right=2, up=3
ACTION_IDS_ORDERED = torch.tensor([
    ACTION_TOKEN_IDS["left"],
    ACTION_TOKEN_IDS["down"],
    ACTION_TOKEN_IDS["right"],
    ACTION_TOKEN_IDS["up"],
])

# Reverse mapping: token_id -> index in ACTION_IDS_ORDERED
TOKEN_ID_TO_IDX = {tid: i for i, tid in enumerate(ACTION_IDS_ORDERED.tolist())}


class ICRLModel:
    def __init__(self, model_id=MODEL_ID, device="cuda", alpha=0.1, gamma=0.9):
        self.device = device
        self.alpha = alpha
        self.gamma = gamma

        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        )

        ia3_config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
        self.model = get_peft_model(base, ia3_config)
        self.model.to(device)

        # Target adapter: clone of trainable (IA3) weights
        self.target_state = {
            n: p.data.clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

        self.action_ids = ACTION_IDS_ORDERED.to(device)

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits

    def forward_target(self, input_ids, attention_mask):
        # Swap in target weights
        saved = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                saved[n] = p.data.clone()
                p.data.copy_(self.target_state[n])

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits

        # Restore policy weights
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(saved[n])

        return logits

    def compute_loss(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        action_mask = batch["action_mask"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_action_idx = batch["next_action_idx"].to(self.device)
        terminal_mask = batch["terminal_mask"].to(self.device)

        # Policy forward
        logits = self.forward(input_ids, attention_mask)
        # Target forward (no grad, swaps weights temporarily)
        target_logits = self.forward_target(input_ids, attention_mask)

        # Extract Q-values for the 4 action tokens
        # logits[:, i, :] predicts token at position i+1
        # So for action at position p, Q-values are at logits[:, p-1, :]
        q_all = logits[:, :, self.action_ids]             # (B, seq, 4)
        target_q_all = target_logits[:, :, self.action_ids]  # (B, seq, 4)

        # Get action positions
        batch_idx, pos_idx = action_mask.nonzero(as_tuple=True)
        if len(batch_idx) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        # Shifted positions: logits at p-1 predict the token at p
        logit_pos = pos_idx - 1

        # Policy Q-values at action positions: (N, 4)
        q_at_actions = q_all[batch_idx, logit_pos]

        # Which action was actually taken? Map token ID -> index 0-3
        action_tokens = input_ids[batch_idx, pos_idx]
        action_idx = torch.zeros_like(action_tokens)
        for tid, idx in TOKEN_ID_TO_IDX.items():
            action_idx[action_tokens == tid] = idx

        # Q(s, a_taken)
        q_taken = q_at_actions.gather(1, action_idx.unsqueeze(1)).squeeze(1)

        # Target Q-values at next-action positions
        next_pos = next_action_idx[batch_idx, pos_idx]
        next_logit_pos = next_pos - 1
        target_q_next = target_q_all[batch_idx, next_logit_pos].max(dim=1).values

        # Bellman target: y = r + gamma * max Q_target(s') * (1 - terminal)
        term = terminal_mask[batch_idx, pos_idx].float()
        rew = rewards[batch_idx, pos_idx]
        y = rew + self.gamma * target_q_next * (1.0 - term)

        loss = F.mse_loss(q_taken, y.detach())

        info = {
            "loss": loss.item(),
            "q_mean": q_taken.mean().item(),
            "q_std": q_taken.std().item(),
            "reward_mean": rew.mean().item(),
            "target_mean": y.mean().item(),
            "num_actions": len(batch_idx),
        }
        return loss, info

    def polyak_update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.target_state[n].mul_(1.0 - self.alpha).add_(
                        self.alpha * p.data
                    )

    def save(self, path):
        torch.save({
            "adapter": self.model.state_dict(),
            "target": self.target_state,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["adapter"])
        self.target_state = ckpt["target"]
