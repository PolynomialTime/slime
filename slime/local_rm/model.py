import json
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel


@dataclass
class DemoSample:
    tokens: list[int]
    response_length: int


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str | None = None,
        base_config: PretrainedConfig | None = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        normalization_constant: float = 1.0,
        reward_max: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias
        self.normalization_constant = normalization_constant
        self.reward_max = reward_max
        self.c_coef = getattr(self, "c_coef", 1.0)


def _finite_float_or_default(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def get_reward_normalization_stats(config: ScalarModelConfig) -> tuple[float, float]:
    bias = _finite_float_or_default(getattr(config, "bias", 0.0), 0.0)
    normalization_constant = _finite_float_or_default(
        getattr(config, "normalization_constant", 1.0),
        1.0,
    )
    if normalization_constant < 1e-3:
        normalization_constant = 1e-3
    return bias, normalization_constant


def sanitize_scalar_model_config(config: ScalarModelConfig) -> ScalarModelConfig:
    bias, normalization_constant = get_reward_normalization_stats(config)
    config.bias = bias
    config.normalization_constant = normalization_constant
    config.c_coef = _finite_float_or_default(getattr(config, "c_coef", 1.0), 1.0)
    config.reward_max = _finite_float_or_default(getattr(config, "reward_max", 5.0), 5.0)
    return config


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.scalar_head = nn.Linear(self.config.hidden_size, 1, dtype=torch.bfloat16)
        nn.init.normal_(self.scalar_head.weight, std=1 / (self.config.hidden_size + 1) ** 0.5)
        nn.init.constant_(self.scalar_head.bias, 0.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        rewards = self.scalar_head(output.last_hidden_state)
        bias, normalization_constant = get_reward_normalization_stats(self.config)
        rewards = rewards - bias
        rewards = rewards / normalization_constant
        return rewards

    @classmethod
    def build_on_cpu(cls, cfg, base_config):
        """Build model architecture on CPU without using from_pretrained (avoids meta tensors)."""
        model = cls.__new__(cls)
        PreTrainedModel.__init__(model, cfg)
        model.config = cfg
        model.lm_backbone = AutoModel.from_config(base_config, trust_remote_code=True, torch_dtype=torch.bfloat16)
        # Materialize any meta tensors to CPU
        model.lm_backbone = model.lm_backbone.to_empty(device="cpu")
        model.scalar_head = nn.Linear(cfg.hidden_size, 1, dtype=torch.bfloat16, device="cpu")
        return model


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4, device: str | torch.device = "cpu"):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.tensor(eps, device=device)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + 1e-8)

    @torch.no_grad()
    def update_from_batch(self, x: torch.Tensor) -> bool:
        x = x.float().reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return False
        b_mean = x.mean()
        b_var = x.var(unbiased=False)
        b_count = torch.tensor(float(x.numel()), device=x.device)

        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * b_count / tot

        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta * delta * self.count * b_count / tot
        new_var = m2 / tot

        if not (
            torch.isfinite(new_mean).item()
            and torch.isfinite(new_var).item()
            and torch.isfinite(tot).item()
        ):
            return False
        self.mean, self.var, self.count = new_mean, new_var, tot
        return True


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def init_reward_model(base_model: str, reward_model_path: str | None):
    if reward_model_path:
        # Load saved checkpoint weights directly
        base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        cfg = ScalarModelConfig(
            base_model=base_model,
            base_config=base_config,
            hidden_size=base_config.hidden_size,
        )
        saved_cfg_path = os.path.join(reward_model_path, "config.json")
        if os.path.exists(saved_cfg_path):
            with open(saved_cfg_path) as f:
                saved_cfg = json.load(f)
            for key in ("bias", "normalization_constant", "c_coef", "reward_max"):
                if key in saved_cfg:
                    setattr(cfg, key, saved_cfg[key])
        sanitize_scalar_model_config(cfg)
        # Create architecture and load weights
        # Untie weights to ensure all params are saved/loaded independently
        cfg.tie_word_embeddings = False
        if base_config is not None:
            base_config.tie_word_embeddings = False
        model = ScalarModel(cfg)
        weights_path = os.path.join(reward_model_path, "pytorch_model.bin")
        index_path = os.path.join(reward_model_path, "pytorch_model.bin.index.json")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        elif os.path.exists(index_path):
            with open(index_path) as idx_f:
                index = json.load(idx_f)
            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for shard in shard_files:
                shard_path = os.path.join(reward_model_path, shard)
                state_dict.update(torch.load(shard_path, map_location="cpu", weights_only=False))
        else:
            state_dict = None
        if state_dict is not None:
            # Check if model has meta tensors
            has_meta = any(p.is_meta for p in model.parameters())
            if has_meta:
                # Meta tensors present — use to_empty first, then load
                model = model.to_empty(device="cpu")
                model.load_state_dict(state_dict, strict=False, assign=True)
                # Handle tied weights: if embed_tokens is still meta, copy from saved
                for name, p in list(model.named_parameters()):
                    if p.is_meta:
                        # Try to find weight by checking common tied weight patterns
                        alt_name = name.replace("embed_tokens", "lm_head") if "embed_tokens" in name else name.replace("lm_head", "embed_tokens")
                        if alt_name in state_dict:
                            p.data = state_dict[alt_name].to(p.dtype)
            else:
                model.load_state_dict(state_dict, strict=False)
        return model
    base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    cfg = ScalarModelConfig(
        base_model=base_model,
        base_config=base_config,
        hidden_size=base_config.hidden_size,
    )
    sanitize_scalar_model_config(cfg)
    return ScalarModel(cfg)


def pad_batch(tokens_list: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(t) for t in tokens_list)
    batch = []
    lengths = []
    for t in tokens_list:
        lengths.append(len(t))
        if len(t) < max_len:
            t = t + [pad_id] * (max_len - len(t))
        batch.append(t)
    input_ids = torch.tensor(batch, dtype=torch.long)
    attention_mask = input_ids != pad_id
    lengths = torch.tensor(lengths, dtype=torch.long)
    return input_ids, attention_mask, lengths


def get_sequence_rewards(
    model: ScalarModel,
    tokens_list: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention_mask, lengths = pad_batch(tokens_list, pad_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    rewards_token = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
    last_idx = (lengths - 1).to(device)
    return rewards_token[torch.arange(rewards_token.size(0), device=device), last_idx]


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


@torch.inference_mode()
def get_sequence_rewards_adaptive(
    model: ScalarModel,
    tokens_list: list[list[int]],
    pad_id: int,
    device: torch.device | str,
    max_batch_size: int | None = None,
    max_batch_tokens: int | None = None,
) -> torch.Tensor:
    if not tokens_list:
        return torch.empty(0, dtype=torch.float32)

    device = torch.device(device)
    stable_batch_size = len(tokens_list) if max_batch_size is None or max_batch_size <= 0 else max_batch_size
    max_batch_tokens = None if max_batch_tokens is None or max_batch_tokens <= 0 else max_batch_tokens

    outputs = []
    start = 0
    while start < len(tokens_list):
        batch_size = min(stable_batch_size, len(tokens_list) - start)

        while True:
            end = min(start + batch_size, len(tokens_list))
            if max_batch_tokens is not None:
                total_tokens = 0
                bounded_end = start
                while bounded_end < end:
                    sample_tokens = len(tokens_list[bounded_end])
                    if bounded_end > start and total_tokens + sample_tokens > max_batch_tokens:
                        break
                    total_tokens += sample_tokens
                    bounded_end += 1
                end = max(start + 1, bounded_end)

            current_tokens = tokens_list[start:end]
            try:
                rewards = get_sequence_rewards(model, current_tokens, pad_id, device)
                outputs.append(rewards.detach().cpu())
                stable_batch_size = min(stable_batch_size, len(current_tokens))
                start = end
                break
            except RuntimeError as exc:
                if not is_cuda_oom_error(exc):
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if len(current_tokens) == 1:
                    raise
                batch_size = max(1, len(current_tokens) // 2)

    return torch.cat(outputs, dim=0)


def build_prompt_text(
    tokenizer,
    prompt: str | list[dict],
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None = None,
) -> str:
    if apply_chat_template:
        if isinstance(prompt, str):
            if prompt.lstrip().startswith("Human: "):
                from slime.local_rm.data import parse_hh_rlhf_text

                prompt = parse_hh_rlhf_text(prompt)
            else:
                prompt = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
    return prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)


def tokenize_prompt_answer(
    tokenizer,
    prompt: str | list[dict],
    answer: str,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None = None,
) -> DemoSample:
    prompt_text = build_prompt_text(
        tokenizer,
        prompt,
        apply_chat_template=apply_chat_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    tokens = prompt_ids + answer_ids
    return DemoSample(tokens=tokens, response_length=len(answer_ids))
