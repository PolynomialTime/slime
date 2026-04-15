import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.optim as optim

from .data import iter_batches, load_demo_samples, load_rollout_samples
from .model import (
    RunningMeanStd,
    get_reward_normalization_stats,
    get_sequence_rewards,
    init_reward_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


def _atomic_save(model, save_dir: Path) -> None:
    tmp_dir = save_dir.with_name(save_dir.name + "_tmp")
    if tmp_dir.exists():
        for f in tmp_dir.glob("*"):
            f.unlink()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir, safe_serialization=False)
    if save_dir.exists():
        for f in save_dir.glob("*"):
            f.unlink()
    if save_dir.exists():
        save_dir.rmdir()
    tmp_dir.rename(save_dir)


def _collect_rollout_paths(args, rollout_id: int, rollout_path: str) -> list[str]:
    window = int(getattr(args, "reward_update_rollout_window", 1) or 1)
    if window <= 1:
        return [rollout_path]

    template = getattr(args, "save_debug_rollout_data", None)
    if template is None:
        return [rollout_path]

    paths = []
    start = max(0, rollout_id - window + 1)
    for rid in range(start, rollout_id + 1):
        path = template.format(rollout_id=rid)
        if os.path.exists(path):
            paths.append(path)
    if not paths:
        paths = [rollout_path]
    return paths


def _build_prompt_index(samples):
    prompt_to_samples = defaultdict(list)
    for sample in samples:
        if sample.prompt is not None:
            prompt_to_samples[sample.prompt].append(sample)
    return prompt_to_samples


def _holdout_prompt(prompt: str, holdout_ratio: float) -> bool:
    if holdout_ratio <= 0.0:
        return False
    digest = hashlib.sha1(prompt.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < holdout_ratio


def _split_rollout_samples_by_prompt(samples, holdout_ratio: float):
    if holdout_ratio <= 0.0:
        return samples, []

    prompt_groups = defaultdict(list)
    for sample in samples:
        if sample.prompt is not None:
            prompt_groups[sample.prompt].append(sample)
    unique_prompts = sorted(prompt_groups)
    if len(unique_prompts) < 2:
        return samples, []

    holdout_prompts = {prompt for prompt in unique_prompts if _holdout_prompt(prompt, holdout_ratio)}
    if not holdout_prompts:
        holdout_prompts = {unique_prompts[0]}
    if len(holdout_prompts) == len(unique_prompts):
        holdout_prompts.remove(unique_prompts[0])

    train_samples = []
    eval_samples = []
    for sample in samples:
        if sample.prompt is not None and sample.prompt in holdout_prompts:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)

    if not train_samples and eval_samples:
        pivot = eval_samples.pop(0)
        train_samples.append(pivot)
    return train_samples, eval_samples


def _build_eval_pairs(eval_rollout_samples, demo_samples_by_prompt, max_samples=None):
    demo_tokens = []
    roll_tokens = []
    for sample in eval_rollout_samples:
        matches = demo_samples_by_prompt.get(sample.prompt)
        if not matches:
            continue
        demo_tokens.append(matches[0].tokens)
        roll_tokens.append(sample.tokens)
        if max_samples is not None and len(demo_tokens) >= max_samples:
            break
    return demo_tokens, roll_tokens


def _run_inline_eval(model, demo_tokens, roll_tokens, pad_id, device, batch_size: int):
    if not demo_tokens:
        return -1.0, 0.0
    correct = 0
    total = len(demo_tokens)
    margin_sum = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            demo_batch = demo_tokens[start : start + batch_size]
            roll_batch = roll_tokens[start : start + batch_size]
            demo_scores = get_sequence_rewards(model, demo_batch, pad_id, device)
            roll_scores = get_sequence_rewards(model, roll_batch, pad_id, device)
            correct += (demo_scores > roll_scores).sum().item()
            margin_sum += (demo_scores - roll_scores).sum().item()
    model.train()
    return correct / total, margin_sum / total


def update_reward(args, rollout_id: int, rollout_path: str) -> None:
    reward_dir = Path(args.reward_model_dir)
    reward_dir.mkdir(parents=True, exist_ok=True)

    base_model = args.reward_model_init or args.hf_checkpoint
    model_path = reward_dir / "latest"

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = load_tokenizer(base_model)
    if model_path.exists():
        model = init_reward_model(base_model, str(model_path))
    else:
        model = init_reward_model(base_model, None)
    model.config.c_coef = float(getattr(args, "c_coef_init", 1.0))
    model.train()

    old_model = init_reward_model(base_model, str(model_path) if model_path.exists() else None)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    old_model.to(device)

    rollout_paths = _collect_rollout_paths(args, rollout_id, rollout_path)
    all_rollout_samples = []
    for path in rollout_paths:
        all_rollout_samples.extend(load_rollout_samples(path))
    if not all_rollout_samples:
        raise RuntimeError("No rollout samples were loaded for reward update.")

    prompt_filter = {sample.prompt for sample in all_rollout_samples if sample.prompt}
    if not prompt_filter:
        raise RuntimeError("Loaded rollout samples do not contain prompt strings for matching.")

    demo_samples = []
    if args.reward_demo_path:
        demo_samples = load_demo_samples(
            args.reward_demo_path,
            tokenizer=tokenizer,
            prompt_key=args.reward_demo_prompt_key,
            answer_key=args.reward_demo_answer_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            prompt_filter=prompt_filter,
        )
    if not demo_samples:
        raise RuntimeError("No reward demo samples were loaded for reward update.")

    demo_samples_by_prompt = _build_prompt_index(demo_samples)
    missing = [sample.prompt for sample in all_rollout_samples if sample.prompt not in demo_samples_by_prompt]
    if missing:
        raise RuntimeError(
            "Prompt-matched reward update found %d rollout samples without reward demos. Example prompt prefix: %s"
            % (len(missing), missing[0][:200])
        )

    train_rollout_samples, eval_rollout_samples = _split_rollout_samples_by_prompt(
        all_rollout_samples,
        float(getattr(args, "reward_eval_holdout_ratio", 0.1) or 0.0),
    )
    if not train_rollout_samples:
        raise RuntimeError("No training rollout samples remain after prompt-hash holdout split.")

    optimizer = optim.AdamW(model.parameters(), lr=args.reward_update_lr)
    rms = RunningMeanStd(device=device)
    epsilon_stability_eps = float(getattr(args, "reward_epsilon_stability_eps", 1e-12))

    c_coef = float(getattr(model.config, "c_coef", 1.0))
    c_coef_min = getattr(args, "c_coef_min", 0.1)
    c_coef_max = getattr(args, "c_coef_max", 10.0)
    coef_scale_up = getattr(args, "coef_scale_up", 1.2)
    coef_scale_down = getattr(args, "coef_scale_down", 0.8)
    target_reward_l2_norm = getattr(args, "target_reward_l2_norm", 5.0)

    num_training_batches = len(train_rollout_samples) // args.reward_update_batch_size
    if num_training_batches <= 0:
        raise RuntimeError(
            "Reward update has zero full batches after prompt matching. train_samples=%d batch_size=%d"
            % (len(train_rollout_samples), args.reward_update_batch_size)
        )

    eval_demo_tokens, eval_roll_tokens = _build_eval_pairs(
        eval_rollout_samples,
        demo_samples_by_prompt,
        max_samples=getattr(args, "reward_eval_max_samples", None),
    )
    logger.info(
        "Prompt-matched reward update: demos=%d train_rollouts=%d eval_rollouts=%d eval_pairs=%d",
        len(demo_samples),
        len(train_rollout_samples),
        len(eval_rollout_samples),
        len(eval_demo_tokens),
    )

    pad_id = tokenizer.pad_token_id
    eval_interval = max(1, num_training_batches // 10)
    eval_batch_size = getattr(args, "reward_eval_batch_size", args.reward_update_batch_size)
    eval_results = []
    best_acc = -1.0
    best_margin = float("-inf")
    best_state_dict = None
    global_batch_idx = 0

    for _ in range(args.reward_update_epochs):
        random.shuffle(train_rollout_samples)
        for roll_batch in iter_batches(train_rollout_samples, args.reward_update_batch_size, drop_last=True):
            demo_batch = [random.choice(demo_samples_by_prompt[s.prompt]) for s in roll_batch]
            demo_tokens = [sample.tokens for sample in demo_batch]
            roll_tokens = [sample.tokens for sample in roll_batch]

            rewards_demo = get_sequence_rewards(model, demo_tokens, pad_id, device)
            rewards_roll = get_sequence_rewards(model, roll_tokens, pad_id, device)

            with torch.no_grad():
                rewards_demo_old = get_sequence_rewards(old_model, demo_tokens, pad_id, device)
                rewards_roll_old = get_sequence_rewards(old_model, roll_tokens, pad_id, device)
                if not torch.isfinite(rewards_demo_old).all().item() or not torch.isfinite(rewards_roll_old).all().item():
                    raise RuntimeError("Old reward model produced non-finite rewards; checkpoint is corrupted.")

            if not torch.isfinite(rewards_demo).all().item() or not torch.isfinite(rewards_roll).all().item():
                raise RuntimeError("Reward model produced non-finite rewards before optimization.")

            matched_margin = rewards_demo.float().mean() - rewards_roll.float().mean()
            delta = torch.cat([rewards_demo - rewards_demo_old, rewards_roll - rewards_roll_old], dim=0).float()
            epsilon = torch.sqrt(torch.mean(delta ** 2) + epsilon_stability_eps)

            loss = -(matched_margin - c_coef * epsilon)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Reward loss became non-finite before backward.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                rewards_norm = torch.cat([rewards_demo.detach(), rewards_roll.detach()], dim=0)
                bias, normalization_constant = get_reward_normalization_stats(model.config)
                raw = rewards_norm.float() * normalization_constant + bias
                if not rms.update_from_batch(raw):
                    raise RuntimeError("RunningMeanStd received no finite reward values.")
                model.config.bias = float(rms.mean.item())
                model.config.normalization_constant = max(float(rms.std.item()), 1e-3)

                eps_val = epsilon.item()
                if not torch.isfinite(torch.tensor(eps_val, device=device)).item():
                    raise RuntimeError("Reward epsilon became non-finite after optimization.")
                hi = target_reward_l2_norm * 1.2
                lo = target_reward_l2_norm * 0.8
                if eps_val > hi:
                    c_coef *= coef_scale_up
                elif eps_val < lo:
                    c_coef *= coef_scale_down
                c_coef = max(c_coef_min, min(c_coef, c_coef_max))
                model.config.c_coef = float(c_coef)

            global_batch_idx += 1
            if eval_demo_tokens and global_batch_idx % eval_interval == 0:
                acc, margin = _run_inline_eval(
                    model,
                    eval_demo_tokens,
                    eval_roll_tokens,
                    pad_id,
                    device,
                    batch_size=eval_batch_size,
                )
                eval_results.append({"batch": global_batch_idx, "matched_acc": acc, "matched_margin": margin})
                if acc > best_acc or (acc == best_acc and margin > best_margin):
                    best_acc = acc
                    best_margin = margin
                    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                logger.info(
                    "reward_eval batch=%d acc=%.4f margin=%.4f loss=%.4f matched_margin=%.4f epsilon=%.4f c_coef=%.4f",
                    global_batch_idx,
                    acc,
                    margin,
                    loss.item(),
                    matched_margin.item(),
                    epsilon.item(),
                    c_coef,
                )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best checkpoint with acc=%.4f margin=%.4f", best_acc, best_margin)

    restored_acc = -1.0
    restored_margin = 0.0
    if eval_demo_tokens:
        restored_acc, restored_margin = _run_inline_eval(
            model,
            eval_demo_tokens,
            eval_roll_tokens,
            pad_id,
            device,
            batch_size=eval_batch_size,
        )
        logger.info("reward_eval_restored acc=%.4f margin=%.4f", restored_acc, restored_margin)

    step_dir = reward_dir / f"step_{rollout_id}"
    step_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(step_dir, safe_serialization=False)
    _atomic_save(model, model_path)

    out_path = reward_dir / f"reward_eval_rollout_{rollout_id}.json"
    out_path.write_text(
        json.dumps(
            {
                "rollout_id": rollout_id,
                "best_matched_acc": best_acc,
                "best_matched_margin": best_margin,
                "restored_matched_acc": restored_acc,
                "restored_matched_margin": restored_margin,
                "eval_curve": eval_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
