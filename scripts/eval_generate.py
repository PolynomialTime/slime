"""
Generate model responses for winrate evaluation.
Run on the GPU cluster (no internet required).

Usage:
    python scripts/eval_generate.py \
        --model-path /path/to/checkpoint \
        --prompt-data /path/to/test.jsonl \
        --output /path/to/outputs.jsonl \
        --apply-chat-template
"""

import argparse
import json
import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_hh_rlhf_text(text: str) -> list[dict]:
    """Parse hh-rlhf text field into conversation turns."""
    messages = []
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-key", default="text")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--apply-chat-template-kwargs", type=str, default=None,
                        help='JSON string, e.g. \'{"enable_thinking":false}\'')
    return parser.parse_args()


def load_prompts(path: str, prompt_key: str) -> list:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj[prompt_key])
    return prompts


def build_input(tokenizer, prompt, apply_chat_template: bool, chat_template_kwargs: dict | None = None) -> str:
    if not apply_chat_template:
        return prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
    if isinstance(prompt, str):
        if prompt.lstrip().startswith("Human: "):
            messages = parse_hh_rlhf_text(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )


def main():
    args = parse_args()
    chat_template_kwargs = json.loads(args.apply_chat_template_kwargs) if args.apply_chat_template_kwargs else None

    logger.info("Loading tokenizer and model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'  # decoder-only models need left padding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompts = load_prompts(args.prompt_data, args.prompt_key)
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompt_data)

    inputs = [build_input(tokenizer, p, args.apply_chat_template, chat_template_kwargs) for p in prompts]

    results = []
    for i in tqdm(range(0, len(inputs), args.batch_size), desc="generating"):
        batch_inputs = inputs[i : i + args.batch_size]
        batch_prompts = prompts[i : i + args.batch_size]

        enc = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (prompt, input_ids, output_ids) in enumerate(
            zip(batch_prompts, enc["input_ids"], out)
        ):
            new_tokens = output_ids[input_ids.shape[0]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append({"prompt": prompt, "response": response})

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Saved %d outputs to %s", len(results), args.output)


if __name__ == "__main__":
    main()
