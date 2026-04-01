"""Persistent reward scoring server. Communicates via stdin/stdout JSON lines."""
import json
import sys
import torch
from .model import get_sequence_rewards, init_reward_model, load_tokenizer


def main():
    # Read config from first line
    config = json.loads(sys.stdin.readline())
    base_model = config["base_model"]
    model_path = config["model_path"]

    tokenizer = load_tokenizer(base_model)
    model = init_reward_model(base_model, model_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

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
                model = init_reward_model(base_model, new_path)
                model.eval()
                if torch.cuda.is_available():
                    model.to("cuda")
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
                rewards = get_sequence_rewards(model, valid_tokens, pad_id, device)
            r_list = rewards.tolist()

            result = [0.0] * len(tokens_list)
            for idx, (orig_i, _) in enumerate(valid):
                result[orig_i] = r_list[idx]
            print(json.dumps(result), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
