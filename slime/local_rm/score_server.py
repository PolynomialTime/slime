"""Persistent reward scoring server. Communicates via stdin/stdout JSON lines."""
import json
import os
import sys
import torch
from .model import get_sequence_rewards_adaptive, init_reward_model, load_tokenizer


def _parse_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def main():
    # Read config from first line
    config = json.loads(sys.stdin.readline())
    base_model = config["base_model"]
    model_path = config["model_path"]
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = load_tokenizer(base_model)
    model = init_reward_model(base_model, model_path)
    model.eval()
    model.to(device)

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    max_batch_size = _parse_env_int("SLIME_CUSTOM_RM_MAX_BATCH_SIZE", 16)
    max_batch_tokens = _parse_env_int("SLIME_CUSTOM_RM_MAX_BATCH_TOKENS", 2048)
    empty_cache_every_request = os.environ.get("SLIME_CUSTOM_RM_EMPTY_CACHE_EVERY_REQUEST", "1").lower() not in {
        "0",
        "false",
        "no",
    }

    # Signal ready
    print(json.dumps({"status": "ready", "device": str(device)}), flush=True)

    # Process requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("cmd") == "reload":
                new_path = request["model_path"]
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = init_reward_model(base_model, new_path)
                model.eval()
                model.to(device)
                device = next(model.parameters()).device
                print(json.dumps({"status": "reloaded"}), flush=True)
                continue

            tokens_list = request["tokens"]
            valid = [(i, t) for i, t in enumerate(tokens_list) if t]

            if not valid:
                print(json.dumps([0.0] * len(tokens_list)), flush=True)
                continue

            valid_tokens = [t for _, t in valid]
            with torch.no_grad():
                rewards = get_sequence_rewards_adaptive(
                    model,
                    valid_tokens,
                    pad_id,
                    device,
                    max_batch_size=max_batch_size,
                    max_batch_tokens=max_batch_tokens,
                )
            r_list = rewards.tolist()

            result = [0.0] * len(tokens_list)
            for idx, (orig_i, _) in enumerate(valid):
                result[orig_i] = r_list[idx]
            print(json.dumps(result), flush=True)
            if empty_cache_every_request and device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
