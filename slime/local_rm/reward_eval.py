import json
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from .model import get_sequence_rewards, init_reward_model, load_tokenizer, tokenize_prompt_answer

logger = logging.getLogger(__name__)


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_eval_cfg(args):
    eval_path = args.reward_eval_path or args.reward_demo_path
    prompt_key = args.reward_eval_prompt_key or args.reward_demo_prompt_key
    chosen_key = args.reward_eval_chosen_key
    rejected_key = args.reward_eval_rejected_key
    batch_size = args.reward_eval_batch_size or args.reward_update_batch_size
    max_samples = args.reward_eval_max_samples
    return eval_path, prompt_key, chosen_key, rejected_key, batch_size, max_samples


def reward_eval(args, rollout_id: int) -> None:
    eval_path, prompt_key, chosen_key, rejected_key, batch_size, max_samples = _get_eval_cfg(args)
    if not eval_path:
        logger.info("reward_eval: reward_eval_path not set, skipping")
        return

    reward_dir = Path(args.reward_model_dir)
    model_path = reward_dir / "latest"
    if not model_path.exists():
        logger.info("reward_eval: reward model %s not found, skipping", model_path)
        return

    base_model = args.reward_model_init or args.hf_checkpoint
    tokenizer = load_tokenizer(base_model)
    pad_id = tokenizer.pad_token_id
    model = init_reward_model(base_model, str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chosen_tokens: list[list[int]] = []
    rejected_tokens: list[list[int]] = []

    total = 0
    for item in _iter_jsonl(eval_path):
        if max_samples is not None and total >= max_samples:
            break
        if prompt_key not in item or chosen_key not in item or rejected_key not in item:
            continue
        prompt = item[prompt_key]
        chosen = item[chosen_key]
        rejected = item[rejected_key]

        chosen_sample = tokenize_prompt_answer(
            tokenizer,
            prompt=prompt,
            answer=chosen,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
        rejected_sample = tokenize_prompt_answer(
            tokenizer,
            prompt=prompt,
            answer=rejected,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )

        if chosen_sample.response_length <= 0 or rejected_sample.response_length <= 0:
            continue
        chosen_tokens.append(chosen_sample.tokens)
        rejected_tokens.append(rejected_sample.tokens)
        total += 1

    if total == 0:
        logger.info("reward_eval: no valid samples found, skipping")
        return

    correct = 0
    pad_id = tokenizer.pad_token_id
    
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", "0"))))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    show_tqdm = world_size <= 1 or rank == 0
    with torch.no_grad():
        for i in tqdm(
            range(0, total, batch_size),
            desc="reward_eval",
            leave=False,
            disable=not show_tqdm,
        ):
            c_batch = chosen_tokens[i : i + batch_size]
            r_batch = rejected_tokens[i : i + batch_size]
            c_scores = get_sequence_rewards(model, c_batch, pad_id, device)
            r_scores = get_sequence_rewards(model, r_batch, pad_id, device)
            correct += (c_scores > r_scores).sum().item()

    # 显式释放
    del model
    del tokenizer
    del chosen_tokens
    del rejected_tokens

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = correct / total
    logger.info(
        "reward_eval rollout=%s samples=%s acc=%.4f path=%s",
        rollout_id,
        total,
        acc,
        eval_path,
    )
