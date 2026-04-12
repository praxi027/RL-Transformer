import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, IA3Config, prepare_model_for_kbit_training

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
    def __init__(self, model_id=MODEL_ID, device="cuda", alpha=0.1, gamma=0.9,
                 load_in_4bit=False, load_in_8bit=False,
                 gradient_checkpointing=False):
        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose only one quantization mode: 4-bit or 8-bit")

        self.device = device
        self.alpha = alpha
        self.gamma = gamma

        load_kwargs = {
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": device}
        elif load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["device_map"] = {"": device}

        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        base.config.use_cache = False
        if load_in_4bit or load_in_8bit:
            gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=gc_kwargs,
            )
        elif gradient_checkpointing:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            base.config.use_cache = False

        ia3_config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
        self.model = get_peft_model(base, ia3_config)
        self.model.config.use_cache = False
        # Ensure IA3 adapter weights match the base model dtype
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        if not (load_in_4bit or load_in_8bit):
            self.model.to(device)

        # Target adapter: clone of trainable (IA3) weights
        self.target_state = {
            n: p.data.clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

        # Optional DDP wrapper for the policy forward. Target forward and
        # parameter access always go through the unwrapped PEFT module.
        self.ddp_model = None

        self.action_ids = ACTION_IDS_ORDERED.to(device)

    def wrap_ddp(self, local_rank):
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        return {
            n: p.detach().cpu().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

    def forward(self, input_ids, attention_mask):
        m = self.ddp_model if self.ddp_model is not None else self.model
        return m(
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
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        action_mask = batch["action_mask"].to(self.device, non_blocking=True)
        rewards = batch["rewards"].to(self.device, non_blocking=True)
        next_action_idx = batch["next_action_idx"].to(self.device, non_blocking=True)
        terminal_mask = batch["terminal_mask"].to(self.device, non_blocking=True)

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
            # Graph-connected zero so .backward() is safe under DDP (all ranks
            # must call backward together to keep gradient all-reduce in sync).
            return logits.sum() * 0.0, {}

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

        # Double DQN: policy selects action, target evaluates it
        next_pos = next_action_idx[batch_idx, pos_idx]
        next_logit_pos = next_pos - 1
        policy_q_next = q_all[batch_idx, next_logit_pos]
        best_next_action = policy_q_next.argmax(dim=1)
        target_q_next = target_q_all[batch_idx, next_logit_pos].gather(
            1, best_next_action.unsqueeze(1),
        ).squeeze(1)

        # Bellman target: y = r + gamma * Q_target(s', argmax_a Q_policy(s',a)) * (1 - terminal)
        term = terminal_mask[batch_idx, pos_idx].to(dtype=target_q_next.dtype)
        rew = rewards[batch_idx, pos_idx].to(dtype=target_q_next.dtype)
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

    def save(self, path, **extra_state):
        ckpt = {
            "adapter": self.trainable_state_dict(),
            "target": {
                n: t.detach().cpu().clone()
                for n, t in self.target_state.items()
            },
            "checkpoint_format": "adapter_only_v1",
        }
        ckpt.update(extra_state)
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["adapter"], strict=False)
        self.target_state = {
            n: t.to(self.device)
            for n, t in ckpt["target"].items()
        }
        return ckpt
