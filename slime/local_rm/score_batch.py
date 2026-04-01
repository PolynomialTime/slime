"""Batch reward scoring subprocess. Called by custom_rm.py with proper CUDA_VISIBLE_DEVICES."""
import argparse
import json
import sys

import torch

from .model import get_sequence_rewards, init_reward_model, load_tokenizer

_MODEL = None
_TOKENIZER = None
_MODEL_PATH = None


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

    rewards = []
    if not tokens_list or all(not t for t in tokens_list):
        rewards = [0.0] * len(tokens_list)
    else:
        valid_tokens = [t for t in tokens_list if t]
        empty_indices = {i for i, t in enumerate(tokens_list) if not t}

        with torch.no_grad():
            # Process in sub-batches to avoid OOM
            SUB_BATCH = 64
            all_r = []
            for start in range(0, len(valid_tokens), SUB_BATCH):
                sub = valid_tokens[start:start + SUB_BATCH]
                r = get_sequence_rewards(_MODEL, sub, pad_id, device)
                all_r.extend(r.tolist())

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
