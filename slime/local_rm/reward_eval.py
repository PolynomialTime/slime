import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from .data import load_prompt_answer_samples
from .model import get_sequence_rewards, init_reward_model, load_tokenizer

logger = logging.getLogger(__name__)


def _build_prompt_index(samples):
    prompt_to_samples = defaultdict(list)
    for sample in samples:
        if sample.prompt is None:
            continue
        prompt_to_samples[sample.prompt].append(sample)
    return prompt_to_samples


def reward_eval(args, rollout_id: int) -> None:
    eval_path = getattr(args, "reward_eval_path", None) or getattr(args, "reward_demo_path", None)
    target_path = getattr(args, "reward_eval_target_path", None)
    if not eval_path or not target_path:
        logger.info("reward_eval: reward_eval_path/target_path not fully set, skipping")
        return

    reward_dir = Path(args.reward_model_dir)
    model_path_value = getattr(args, "reward_model_path", None)
    model_path = Path(model_path_value) if model_path_value else reward_dir / "latest"
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

    positive_samples = load_prompt_answer_samples(
        eval_path,
        tokenizer=tokenizer,
        prompt_key=getattr(args, "reward_eval_prompt_key", None) or getattr(args, "reward_demo_prompt_key", "text"),
        answer_key=getattr(args, "reward_eval_chosen_key", "chosen"),
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
    )
    max_samples = getattr(args, "reward_eval_max_samples", None)
    if max_samples is not None:
        positive_samples = positive_samples[:max_samples]

    target_samples = load_prompt_answer_samples(
        target_path,
        tokenizer=tokenizer,
        prompt_key=getattr(args, "reward_eval_target_prompt_key", "prompt"),
        answer_key=getattr(args, "reward_eval_target_answer_key", "response"),
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
    )
    target_index = _build_prompt_index(target_samples)

    positive_tokens: list[list[int]] = []
    target_tokens: list[list[int]] = []
    missing = 0
    for sample in positive_samples:
        matches = target_index.get(sample.prompt)
        if not matches:
            missing += 1
            continue
        positive_tokens.append(sample.tokens)
        target_tokens.append(matches[0].tokens)

    total = len(positive_tokens)
    if total == 0:
        logger.info("reward_eval: no valid prompt-matched samples found, skipping")
        return

    correct = 0
    margin_sum = 0.0
    batch_size = args.reward_eval_batch_size or args.reward_update_batch_size

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
            c_batch = positive_tokens[i : i + batch_size]
            t_batch = target_tokens[i : i + batch_size]
            c_scores = get_sequence_rewards(model, c_batch, pad_id, device)
            t_scores = get_sequence_rewards(model, t_batch, pad_id, device)
            correct += (c_scores > t_scores).sum().item()
            margin_sum += (c_scores - t_scores).sum().item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = correct / total
    margin = margin_sum / total
    eval_source = getattr(args, "reward_eval_source", None) or "external"
    logger.info(
        "reward_eval rollout=%s source=%s samples=%s acc=%.4f margin=%.4f positive_path=%s target_path=%s missing=%s model_path=%s",
        rollout_id,
        eval_source,
        total,
        acc,
        margin,
        eval_path,
        target_path,
        missing,
        model_path,
    )

    out_path_value = getattr(args, "reward_eval_output_path", None)
    out_path = Path(out_path_value) if out_path_value else reward_dir / f"reward_eval_rollout_{rollout_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rollout_id": rollout_id,
                "eval_source": eval_source,
                "matched_acc": acc,
                "matched_margin": margin,
                "correct": correct,
                "total": total,
                "positive": len(positive_samples),
                "targets": len(target_samples),
                "missing": missing,
                "model_path": str(model_path),
                "positive_path": eval_path,
                "target_path": target_path,
            },
            f,
            indent=2,
        )
    logger.info("reward_eval: saved results to %s", out_path)
