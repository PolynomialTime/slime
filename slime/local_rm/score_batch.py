"""Batch reward scoring subprocess. Called by custom_rm.py with proper CUDA_VISIBLE_DEVICES."""
import argparse
import json
import os
import sys

import torch

from .model import get_sequence_rewards_adaptive, init_reward_model, load_tokenizer

_MODEL = None
_TOKENIZER = None
_MODEL_PATH = None


def _parse_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _load(base_model, model_path):
    global _MODEL, _TOKENIZER, _MODEL_PATH
    if _MODEL is not None and _MODEL_PATH == model_path:
        return
    _TOKENIZER = load_tokenizer(base_model)
    _MODEL = init_reward_model(base_model, model_path)
    _MODEL.eval()
    if torch.cuda.is_available():
        _MODEL.to("cuda")
    _MODEL_PATH = model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokens-file", required=True)
    args = parser.parse_args()

    with open(args.tokens_file) as f:
        tokens_list = json.load(f)

    _load(args.base_model, args.model_path)

    device = next(_MODEL.parameters()).device
    pad_id = _TOKENIZER.pad_token_id
    max_batch_size = _parse_env_int("SLIME_CUSTOM_RM_MAX_BATCH_SIZE", 16)
    max_batch_tokens = _parse_env_int("SLIME_CUSTOM_RM_MAX_BATCH_TOKENS", 2048)

    rewards = []
    if not tokens_list or all(not t for t in tokens_list):
        rewards = [0.0] * len(tokens_list)
    else:
        valid_tokens = [t for t in tokens_list if t]
        empty_indices = {i for i, t in enumerate(tokens_list) if not t}

        with torch.no_grad():
            all_r = get_sequence_rewards_adaptive(
                _MODEL,
                valid_tokens,
                pad_id,
                device,
                max_batch_size=max_batch_size,
                max_batch_tokens=max_batch_tokens,
            ).tolist()

        ridx = 0
        for i in range(len(tokens_list)):
            if i in empty_indices:
                rewards.append(0.0)
            else:
                rewards.append(all_r[ridx])
                ridx += 1

    # Output as JSON on last line
    print(json.dumps(rewards))


if __name__ == "__main__":
    main()
