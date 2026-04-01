"""
Convert hh-rlhf text+label format to SFT messages format.
Samples 10k random examples from the training set.

Input:  {"text": "Human: ...\n\nAssistant: ...\n\nHuman: ...", "label": "response"}
Output: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
"""
import json
import random
import sys
from pathlib import Path


def parse_hh_rlhf_text(text: str) -> list[dict]:
    """Parse hh-rlhf 'text' field into message turns."""
    messages = []
    # Split on "Human: " and "Assistant: " markers
    parts = text.strip().split("\n\n")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Human: "):
            messages.append({"role": "user", "content": part[len("Human: "):]})
        elif part.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": part[len("Assistant: "):]})
    return messages


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    # Load all data
    all_data = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append(json.loads(line))

    print(f"Loaded {len(all_data)} samples from {input_path}")

    # Sample
    random.seed(seed)
    if n_samples < len(all_data):
        sampled = random.sample(all_data, n_samples)
    else:
        sampled = all_data
    print(f"Sampled {len(sampled)} for SFT")

    # Convert to messages format
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled:
            messages = parse_hh_rlhf_text(item["text"])
            label = item.get("label", "")
            if not messages or not label:
                skipped += 1
                continue
            # Append the label as the final assistant response
            messages.append({"role": "assistant", "content": label})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    written = len(sampled) - skipped
    print(f"Written {written} samples to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
