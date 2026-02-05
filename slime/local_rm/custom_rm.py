import os
import time

import torch

from .model import get_sequence_rewards, init_reward_model, load_tokenizer

_MODEL = None
_TOKENIZER = None
_MODEL_MTIME = 0.0


def _load_model(args):
    global _MODEL, _TOKENIZER, _MODEL_MTIME
    reward_dir = getattr(args, "reward_model_dir", None) or "reward_model"
    model_path = os.path.join(reward_dir, "latest")
    if not os.path.exists(model_path):
        return None

    mtime = os.path.getmtime(model_path)
    if _MODEL is None or mtime > _MODEL_MTIME:
        base_model = getattr(args, "reward_model_init", None) or args.hf_checkpoint
        _TOKENIZER = load_tokenizer(base_model)
        _MODEL = init_reward_model(base_model, model_path)
        _MODEL.eval()
        _MODEL_MTIME = mtime
    return _MODEL


def custom_rm(args, sample):
    model = _load_model(args)
    if model is None:
        return 0.0

    base_model = getattr(args, "reward_model_init", None) or args.hf_checkpoint
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(base_model)
    pad_id = _TOKENIZER.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokens = sample.tokens
    if not tokens:
        return 0.0
    with torch.no_grad():
        reward = get_sequence_rewards(model, [tokens], pad_id, device)[0].item()
    return reward
