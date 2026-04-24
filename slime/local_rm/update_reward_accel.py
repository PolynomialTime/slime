import argparse
import gc
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ImportError:
    SummaryWriter = None
    _HAS_TB = False

from .data import (
    build_token_sample_index,
    init_alignment_stats,
    iter_batches,
    load_demo_samples,
    load_prompt_answer_samples,
    load_rollout_samples,
    match_token_sample,
    summarize_rollout_samples,
)
from .model import (
    RunningMeanStd,
    get_reward_normalization_stats,
    get_sequence_rewards,
    init_reward_model,
    load_tokenizer,
)
from slime.utils.logging_utils import configure_logger


def _shard_samples(samples: list, process_index: int, num_processes: int) -> list:
    if num_processes <= 1:
        return samples
    return samples[process_index::num_processes]


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


def _filter_rollout_samples_with_demos(
    rollout_samples,
    demo_sample_index,
    *,
    max_missing_frac: float,
):
    matched_samples = []
    missing_prompts = []
    alignment_stats = init_alignment_stats()
    for sample in rollout_samples:
        matched_demo = match_token_sample(
            sample,
            demo_sample_index,
            strict_row_id=True,
            random_prompt_fallback=False,
            stats=alignment_stats,
        )
        if matched_demo is not None:
            matched_samples.append(sample)
        else:
            missing_prompts.append(sample.prompt)

    total = len(rollout_samples)
    missing_count = len(missing_prompts)
    missing_frac = missing_count / max(total, 1)
    stats = {
        "total": total,
        "matched": len(matched_samples),
        "missing": missing_count,
        "missing_frac": missing_frac,
        "missing_unique_prompts": len({p for p in missing_prompts if p}),
        "example_prompt": (missing_prompts[0][:200] if missing_prompts else None),
    }
    stats.update(alignment_stats)

    if missing_count > 0 and missing_frac > max_missing_frac:
        raise RuntimeError(
            "Reward update found %d rollout samples without reward demos "
            "(unique_prompts=%d, frac=%.4f, threshold=%.4f, matched_by_row_id=%d, prompt_fallback=%d, missing_row_id_match=%d, legacy=%d). Example prompt prefix: %s"
            % (
                missing_count,
                stats["missing_unique_prompts"],
                missing_frac,
                max_missing_frac,
                stats["matched_by_row_id"],
                stats["matched_by_prompt_fallback"],
                stats["missing_row_id_match"],
                stats["legacy_samples_without_source_row_id"],
                stats["example_prompt"] or "<none>",
            )
        )

    return matched_samples, stats


def _holdout_prompt(prompt: str, holdout_ratio: float) -> bool:
    if holdout_ratio <= 0.0:
        return False
    if holdout_ratio >= 1.0:
        return True
    digest = hashlib.sha1(prompt.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < holdout_ratio


def _split_rollout_samples_by_prompt(samples, holdout_ratio: float):
    if holdout_ratio <= 0.0:
        return samples, [], {
            "total_samples": len(samples),
            "train_samples": len(samples),
            "eval_samples": 0,
            "total_prompts": len({s.prompt for s in samples if s.prompt}),
            "train_prompts": len({s.prompt for s in samples if s.prompt}),
            "eval_prompts": 0,
            "holdout_ratio": holdout_ratio,
        }

    prompt_groups = defaultdict(list)
    for sample in samples:
        if sample.prompt is not None:
            prompt_groups[sample.prompt].append(sample)

    unique_prompts = sorted(prompt_groups)
    if len(unique_prompts) < 2:
        return samples, [], {
            "total_samples": len(samples),
            "train_samples": len(samples),
            "eval_samples": 0,
            "total_prompts": len(unique_prompts),
            "train_prompts": len(unique_prompts),
            "eval_prompts": 0,
            "holdout_ratio": holdout_ratio,
        }

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
        fallback_prompt = next(iter(holdout_prompts))
        keep_eval = []
        for sample in eval_samples:
            if sample.prompt == fallback_prompt:
                train_samples.append(sample)
            else:
                keep_eval.append(sample)
        eval_samples = keep_eval
        holdout_prompts.discard(fallback_prompt)

    stats = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "total_prompts": len(unique_prompts),
        "train_prompts": len(unique_prompts) - len(holdout_prompts),
        "eval_prompts": len(holdout_prompts),
        "holdout_ratio": holdout_ratio,
    }
    return train_samples, eval_samples, stats


def _build_holdout_eval_data(eval_rollout_samples, demo_sample_index, max_samples=None):
    chosen_tokens = []
    target_tokens = []
    missing = 0
    alignment_stats = init_alignment_stats()
    for sample in eval_rollout_samples:
        matched_demo = match_token_sample(
            sample,
            demo_sample_index,
            strict_row_id=True,
            random_prompt_fallback=False,
            stats=alignment_stats,
        )
        if matched_demo is None:
            missing += 1
            continue
        chosen_tokens.append(matched_demo.tokens)
        target_tokens.append(sample.tokens)
        if max_samples is not None and len(chosen_tokens) >= max_samples:
            break

    stats = {
        "positive": len(chosen_tokens),
        "targets": len(eval_rollout_samples),
        "matched": len(chosen_tokens),
        "missing": missing,
        "source": "holdout",
    }
    stats.update(alignment_stats)
    return chosen_tokens, target_tokens, stats


def _load_external_eval_data(args, tokenizer):
    eval_path = getattr(args, "reward_eval_path", None) or getattr(args, "reward_demo_path", None)
    target_path = getattr(args, "reward_eval_target_path", None)
    rejected_key = getattr(args, "reward_eval_rejected_key", None)
    if not eval_path or (not target_path and not rejected_key):
        return [], [], {"positive": 0, "targets": 0, "matched": 0, "missing": 0, "source": "none"}

    apply_ct = getattr(args, "apply_chat_template", False)
    apply_ct_kwargs = getattr(args, "apply_chat_template_kwargs", None)
    max_samples = getattr(args, "reward_eval_max_samples", None)
    shuffle_seed = int(getattr(args, "reward_eval_shuffle_seed", 42))
    eval_prompt_key = getattr(args, "reward_eval_prompt_key", None) or getattr(args, "reward_demo_prompt_key", "text")

    positive_samples = load_prompt_answer_samples(
        eval_path,
        tokenizer=tokenizer,
        prompt_key=eval_prompt_key,
        answer_key=getattr(args, "reward_eval_chosen_key", "chosen"),
        apply_chat_template=apply_ct,
        apply_chat_template_kwargs=apply_ct_kwargs,
    )
    if max_samples is not None and len(positive_samples) > max_samples:
        shuffled_positive_samples = list(positive_samples)
        random.Random(shuffle_seed).shuffle(shuffled_positive_samples)
        positive_samples = shuffled_positive_samples[:max_samples]

    if target_path:
        target_samples = load_prompt_answer_samples(
            target_path,
            tokenizer=tokenizer,
            prompt_key=getattr(args, "reward_eval_target_prompt_key", "prompt"),
            answer_key=getattr(args, "reward_eval_target_answer_key", "response"),
            apply_chat_template=apply_ct,
            apply_chat_template_kwargs=apply_ct_kwargs,
        )
    else:
        target_samples = load_prompt_answer_samples(
            eval_path,
            tokenizer=tokenizer,
            prompt_key=eval_prompt_key,
            answer_key=rejected_key,
            apply_chat_template=apply_ct,
            apply_chat_template_kwargs=apply_ct_kwargs,
        )
    target_index = build_token_sample_index(target_samples)

    chosen_tokens = []
    target_tokens = []
    missing = 0
    alignment_stats = init_alignment_stats()
    strict_row_id = not bool(target_path)
    for sample in positive_samples:
        matched_target = match_token_sample(
            sample,
            target_index,
            strict_row_id=strict_row_id,
            random_prompt_fallback=False,
            stats=alignment_stats,
        )
        if matched_target is None:
            missing += 1
            continue
        chosen_tokens.append(sample.tokens)
        target_tokens.append(matched_target.tokens)

    stats = {
        "positive": len(positive_samples),
        "targets": len(target_samples),
        "matched": len(chosen_tokens),
        "missing": missing,
        "source": "external",
    }
    stats.update(alignment_stats)
    return chosen_tokens, target_tokens, stats


def _run_inline_eval(model, chosen_tokens, target_tokens, pad_id, device, batch_size=8):
    if not chosen_tokens:
        return -1.0, 0.0
    correct = 0
    total = len(chosen_tokens)
    margin_sum = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            c_batch = chosen_tokens[i : i + batch_size]
            r_batch = target_tokens[i : i + batch_size]
            c_scores = get_sequence_rewards(model, c_batch, pad_id, device)
            r_scores = get_sequence_rewards(model, r_batch, pad_id, device)
            correct += (c_scores > r_scores).sum().item()
            margin_sum += (c_scores - r_scores).sum().item()
    model.train()
    return correct / total, margin_sum / total


def _reload_eval_tolerances(eval_size: int) -> tuple[float, float]:
    # One flipped pair changes accuracy by 1 / eval_size; anything above that is a real mismatch.
    acc_tol = 1.1 / max(eval_size, 1)
    margin_tol = 1e-3
    return acc_tol, margin_tol


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=str, required=True)
    parser.add_argument("--rollout-id", type=int, required=True)
    parser.add_argument("--rollout-path", type=str, required=True)
    return parser.parse_args()


def _reward_tb_dir(reward_dir: Path, rollout_id: int) -> Path:
    slime_root = reward_dir.parents[1] if len(reward_dir.parents) >= 2 else reward_dir.parent
    round_id = os.environ.get("ROUND_ID", str(rollout_id))
    return slime_root / "tensorboard_log" / "slime-reward" / f"round{round_id}"


def main():
    cli = parse_args()
    with open(cli.args_json, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    args = SimpleNamespace(**cfg_dict)

    configure_logger()
    accelerator = Accelerator()
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    base_model = args.reward_model_init or args.hf_checkpoint
    reward_dir = Path(args.reward_model_dir)
    reward_dir.mkdir(parents=True, exist_ok=True)
    model_path = reward_dir / "latest"

    tokenizer = load_tokenizer(base_model)
    if model_path.exists():
        model = init_reward_model(base_model, str(model_path))
        accelerator.print(
            "[reward] Loaded checkpoint from %s with carried c_coef=%.4f"
            % (model_path, float(getattr(model.config, "c_coef", getattr(args, "c_coef_init", 1.0))))
        )
    else:
        model = init_reward_model(base_model, None)
        model.config.c_coef = float(getattr(args, "c_coef_init", 1.0))
        accelerator.print(
            "[reward] Initializing fresh model with c_coef_init=%.4f"
            % float(getattr(args, "c_coef_init", 1.0))
        )
    if hasattr(model, "lm_backbone") and hasattr(model.lm_backbone, "gradient_checkpointing_enable"):
        model.lm_backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        accelerator.print("[reward] Non-reentrant gradient checkpointing enabled for lm_backbone")
    model.train()

    old_model = init_reward_model(base_model, str(model_path) if model_path.exists() else None)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=args.reward_update_lr)
    model, optimizer = accelerator.prepare(model, optimizer)
    old_model.to(accelerator.device)

    all_rollout_samples = []
    rollout_summary = {
        "total": 0,
        "empty": 0,
        "eos_only": 0,
        "truncated": 0,
        "response_chars": 0,
        "non_printing_chars": 0,
        "non_printing_samples": 0,
    }
    rollout_paths = [cli.rollout_path]
    window = int(getattr(args, "reward_update_rollout_window", 1) or 1)
    if window > 1 and getattr(args, "save_debug_rollout_data", None):
        start = max(0, cli.rollout_id - window + 1)
        rollout_paths = [
            args.save_debug_rollout_data.format(rollout_id=i)
            for i in range(start, cli.rollout_id + 1)
        ]
    for path in rollout_paths:
        if Path(path).exists():
            summary = summarize_rollout_samples(path, stop_token_ids=getattr(args, "rollout_stop_token_ids", None))
            for key, value in summary.items():
                rollout_summary[key] += value
            all_rollout_samples.extend(load_rollout_samples(path))

    if not all_rollout_samples:
        raise RuntimeError("No rollout samples were loaded for reward update.")

    source_row_id_filter = {sample.source_row_id for sample in all_rollout_samples if sample.source_row_id is not None}
    legacy_prompt_filter = {sample.prompt for sample in all_rollout_samples if sample.prompt and sample.source_row_id is None}
    if not source_row_id_filter and not legacy_prompt_filter:
        raise RuntimeError("Loaded rollout samples do not contain source_row_id or prompt strings for matching.")

    demo_samples = []
    if args.reward_demo_path:
        demo_samples = load_demo_samples(
            args.reward_demo_path,
            tokenizer=tokenizer,
            prompt_key=args.reward_demo_prompt_key,
            answer_key=args.reward_demo_answer_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            source_row_id_filter=source_row_id_filter or None,
            prompt_filter=legacy_prompt_filter or None,
        )
    if not demo_samples:
        raise RuntimeError("No reward demo samples were loaded for reward update.")

    demo_sample_index = build_token_sample_index(demo_samples)
    max_missing_demo_frac = float(getattr(args, "reward_max_missing_demo_frac", 0.10) or 0.0)
    all_rollout_samples, demo_match_stats = _filter_rollout_samples_with_demos(
        all_rollout_samples,
        demo_sample_index,
        max_missing_frac=max_missing_demo_frac,
    )
    if not all_rollout_samples:
        raise RuntimeError("All rollout samples were dropped because no reward demos matched their prompts.")

    holdout_ratio = float(getattr(args, "reward_eval_holdout_ratio", 0.1) or 0.0)
    train_rollout_samples_all, eval_rollout_samples, split_stats = _split_rollout_samples_by_prompt(
        all_rollout_samples,
        holdout_ratio=holdout_ratio,
    )
    if not train_rollout_samples_all:
        raise RuntimeError("No training rollout samples remain after prompt-hash holdout split.")

    rollout_samples = _shard_samples(train_rollout_samples_all, accelerator.process_index, accelerator.num_processes)
    if not rollout_samples:
        raise RuntimeError(
            "Reward update left rank %d without training rollout samples after sharding."
            % accelerator.process_index
        )

    external_eval_chosen, external_eval_targets, external_eval_stats = _load_external_eval_data(args, tokenizer)
    if external_eval_chosen:
        eval_chosen = external_eval_chosen
        eval_targets = external_eval_targets
        eval_stats = external_eval_stats
    else:
        eval_chosen, eval_targets, eval_stats = _build_holdout_eval_data(
            eval_rollout_samples,
            demo_sample_index,
            max_samples=getattr(args, "reward_eval_max_samples", None),
        )

    tb_writer = None
    if accelerator.is_main_process and _HAS_TB:
        tb_dir = _reward_tb_dir(reward_dir, cli.rollout_id)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
        accelerator.print(f"[reward_tb] Logging to {tb_dir}")

    if accelerator.is_main_process:
        accelerator.print(
            "Loaded %d reward demos, %d total rollouts, %d train rollouts, %d eval rollouts"
            % (
                len(demo_samples),
                len(all_rollout_samples),
                len(train_rollout_samples_all),
                len(eval_rollout_samples),
            )
        )
        if demo_match_stats["missing"] > 0:
            accelerator.print(
                "Reward demo matching dropped %d/%d rollout samples (unique_prompts=%d frac=%.4f row_id=%d prompt_fallback=%d missing_row_id=%d legacy=%d) due to missing/empty demos; example=%r"
                % (
                    demo_match_stats["missing"],
                    demo_match_stats["total"],
                    demo_match_stats["missing_unique_prompts"],
                    demo_match_stats["missing_frac"],
                    demo_match_stats["matched_by_row_id"],
                    demo_match_stats["matched_by_prompt_fallback"],
                    demo_match_stats["missing_row_id_match"],
                    demo_match_stats["legacy_samples_without_source_row_id"],
                    demo_match_stats["example_prompt"],
                )
            )
        accelerator.print(
            "Prompt split: total=%d train=%d eval=%d holdout_ratio=%.3f"
            % (
                split_stats["total_prompts"],
                split_stats["train_prompts"],
                split_stats["eval_prompts"],
                split_stats["holdout_ratio"],
            )
        )
        if rollout_summary["total"] > 0:
            accelerator.print(
                "Rollout diagnostics before filtering: total=%d empty=%d eos_only=%d truncated=%d"
                % (
                    rollout_summary["total"],
                    rollout_summary["empty"],
                    rollout_summary["eos_only"],
                    rollout_summary["truncated"],
                )
            )
            if tb_writer is not None:
                total = rollout_summary["total"]
                tb_writer.add_scalar("reward/empty_rollout_frac", rollout_summary["empty"] / total, cli.rollout_id)
                tb_writer.add_scalar("reward/eos_only_rollout_frac", rollout_summary["eos_only"] / total, cli.rollout_id)
                tb_writer.add_scalar("reward/truncated_rollout_frac", rollout_summary["truncated"] / total, cli.rollout_id)
                tb_writer.add_scalar(
                    "reward/non_printing_rollout_frac",
                    rollout_summary["non_printing_samples"] / total,
                    cli.rollout_id,
                )
                tb_writer.add_scalar(
                    "reward/non_printing_rollout_char_frac",
                    rollout_summary["non_printing_chars"] / max(rollout_summary["response_chars"], 1),
                    cli.rollout_id,
                )
            accelerator.print(
                "Rollout hygiene before filtering: response_chars=%d non_printing_samples=%d non_printing_chars=%d non_printing_char_frac=%.6f"
                % (
                    rollout_summary["response_chars"],
                    rollout_summary["non_printing_samples"],
                    rollout_summary["non_printing_chars"],
                    rollout_summary["non_printing_chars"] / max(rollout_summary["response_chars"], 1),
                )
            )
        accelerator.print(
            "Loaded %d eval matched pairs for inline eval (source=%s positive=%d targets=%d missing=%d row_id=%d prompt_fallback=%d missing_row_id=%d legacy=%d)"
            % (
                len(eval_chosen),
                eval_stats.get("source", "holdout"),
                eval_stats["positive"],
                eval_stats["targets"],
                eval_stats["missing"],
                eval_stats.get("matched_by_row_id", 0),
                eval_stats.get("matched_by_prompt_fallback", 0),
                eval_stats.get("missing_row_id_match", 0),
                eval_stats.get("legacy_samples_without_source_row_id", 0),
            )
        )
        accelerator.print(
            "After sharding (process %d/%d): %d training rollout samples"
            % (
                accelerator.process_index,
                accelerator.num_processes,
                len(rollout_samples),
            )
        )

    def _cfg(m):
        return accelerator.unwrap_model(m).config

    pad_id = tokenizer.pad_token_id
    rms = RunningMeanStd(device=accelerator.device)
    epsilon_stability_eps = float(getattr(args, "reward_epsilon_stability_eps", 1e-12))
    c_coef = float(getattr(_cfg(model), "c_coef", 1.0))
    c_coef_min = getattr(args, "c_coef_min", 0.1)
    c_coef_max = getattr(args, "c_coef_max", 10.0)
    coef_scale_up = getattr(args, "coef_scale_up", 1.2)
    coef_scale_down = getattr(args, "coef_scale_down", 0.8)
    target_reward_l2_norm = getattr(args, "target_reward_l2_norm", 5.0)
    online_pref_weight = float(getattr(args, "reward_online_pref_weight", 1.0) or 0.0)
    if online_pref_weight <= 0.0:
        raise RuntimeError("Reward update requires at least one positive preference weight.")

    num_training_batches_local = len(rollout_samples) // args.reward_update_batch_size
    local_batch_tensor = torch.tensor(num_training_batches_local, device=accelerator.device)
    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_batch_tensor, op=dist.ReduceOp.MIN)
    global_min_batches = int(local_batch_tensor.item())
    if global_min_batches <= 0:
        raise RuntimeError(
            "Reward update has zero full batches on at least one rank after sharding. "
            "local_batches=%d batch_size=%d"
            % (num_training_batches_local, args.reward_update_batch_size)
        )

    if accelerator.is_main_process:
        accelerator.print(f"Batch size: {args.reward_update_batch_size}")
        accelerator.print(f"Reward demo pool: {len(demo_samples)}")
        accelerator.print(f"Local rollout batches: {num_training_batches_local}")
        accelerator.print(f"Training iterations per epoch: {global_min_batches}")

    eval_interval = max(1, global_min_batches // 10)
    eval_batch_size = getattr(args, "reward_eval_batch_size", 8) or 8
    eval_results = []
    best_acc = -1.0
    best_margin = float("-inf")
    best_state_dict = None
    best_config_snapshot = None
    global_batch_idx = 0

    for _epoch in tqdm(
        range(args.reward_update_epochs),
        desc="reward_update_epoch",
        leave=False,
        disable=not accelerator.is_main_process,
    ):
        random.shuffle(rollout_samples)
        batch_iter = iter_batches(rollout_samples, args.reward_update_batch_size, drop_last=True)
        for batch_idx, roll_batch in enumerate(
            tqdm(
                batch_iter,
                desc="reward_update_batch",
                leave=False,
                disable=not accelerator.is_main_process,
            )
        ):
            if batch_idx >= global_min_batches:
                break

            demo_batch = []
            for rollout_sample in roll_batch:
                matched_demo = match_token_sample(
                    rollout_sample,
                    demo_sample_index,
                    strict_row_id=True,
                    random_prompt_fallback=True,
                )
                if matched_demo is None:
                    raise RuntimeError(
                        "Reward training batch lost its demo match after upfront filtering; "
                        f"source_row_id={rollout_sample.source_row_id} prompt_prefix={repr((rollout_sample.prompt or '')[:200])}"
                    )
                demo_batch.append(matched_demo)
            demo_tokens = [sample.tokens for sample in demo_batch]
            roll_tokens = [sample.tokens for sample in roll_batch]

            rewards_demo = get_sequence_rewards(model, demo_tokens, pad_id, accelerator.device)
            rewards_roll = get_sequence_rewards(model, roll_tokens, pad_id, accelerator.device)

            with torch.no_grad():
                rewards_demo_old = get_sequence_rewards(old_model, demo_tokens, pad_id, accelerator.device)
                rewards_roll_old = get_sequence_rewards(old_model, roll_tokens, pad_id, accelerator.device)
                if not torch.isfinite(rewards_demo_old).all().item() or not torch.isfinite(rewards_roll_old).all().item():
                    raise RuntimeError("Old reward model produced non-finite rewards; checkpoint is corrupted.")

            if not torch.isfinite(rewards_demo).all().item() or not torch.isfinite(rewards_roll).all().item():
                raise RuntimeError("Reward model produced non-finite rewards before optimization.")

            online_margin = rewards_demo.float().mean() - rewards_roll.float().mean()

            preference_objective = online_pref_weight * online_margin
            delta_terms = [rewards_demo - rewards_demo_old, rewards_roll - rewards_roll_old]

            delta = torch.cat(delta_terms, dim=0).float()
            epsilon = torch.sqrt(torch.mean(delta ** 2) + epsilon_stability_eps)

            loss = -(preference_objective - c_coef * epsilon)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Reward loss became non-finite before backward.")
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                rewards_norm = torch.cat([rewards_demo.detach(), rewards_roll.detach()], dim=0)
                cfg = _cfg(model)
                bias, normalization_constant = get_reward_normalization_stats(cfg)
                raw = rewards_norm.float() * normalization_constant + bias
                raw_all = accelerator.gather(raw)
                if not rms.update_from_batch(raw_all):
                    raise RuntimeError("RunningMeanStd received no finite reward values.")
                cfg.bias = float(rms.mean.item())
                cfg.normalization_constant = max(float(rms.std.item()), 1e-3)

            with torch.no_grad():
                eps_all = accelerator.gather(epsilon.detach())
                epsilon_global = eps_all.mean().item()
                if not torch.isfinite(torch.tensor(epsilon_global, device=accelerator.device)).item():
                    raise RuntimeError("Reward epsilon became non-finite after optimization.")
                hi = target_reward_l2_norm * 1.2
                lo = target_reward_l2_norm * 0.8
                # Skip c_coef updates for the first few batches. The very first
                # batch(es) can emit a pseudo-large epsilon because the
                # accelerator-wrapped `model` and the plain `old_model` have
                # slightly different numerical paths (mixed precision, etc.),
                # which otherwise drives c_coef straight to the ceiling.
                if global_batch_idx >= 5:
                    if epsilon_global > hi:
                        c_coef *= coef_scale_up
                    elif epsilon_global < lo:
                        c_coef *= coef_scale_down
                c_coef = max(c_coef_min, min(c_coef, c_coef_max))
                _cfg(model).c_coef = float(c_coef)
                online_acc = accelerator.gather((rewards_demo > rewards_roll).float()).mean().item()
                online_margin_global = accelerator.gather((rewards_demo - rewards_roll).detach().float()).mean().item()
                r_demo_global = accelerator.gather(rewards_demo.detach().float()).mean().item()
                r_roll_global = accelerator.gather(rewards_roll.detach().float()).mean().item()

            global_batch_idx += 1

            if tb_writer is not None:
                tb_writer.add_scalar("reward/loss", loss.item(), global_batch_idx)
                tb_writer.add_scalar("reward/online_margin", online_margin_global, global_batch_idx)
                tb_writer.add_scalar("reward/online_acc", online_acc, global_batch_idx)
                tb_writer.add_scalar("reward/irl_margin", online_margin_global, global_batch_idx)
                tb_writer.add_scalar("reward/matched_margin", online_margin_global, global_batch_idx)
                tb_writer.add_scalar("reward/matched_acc", online_acc, global_batch_idx)
                tb_writer.add_scalar("reward/r_demo", r_demo_global, global_batch_idx)
                tb_writer.add_scalar("reward/r_roll", r_roll_global, global_batch_idx)
                tb_writer.add_scalar("reward/epsilon", epsilon_global, global_batch_idx)
                tb_writer.add_scalar("reward/c_coef", c_coef, global_batch_idx)

            if eval_chosen and global_batch_idx % eval_interval == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    acc, margin = _run_inline_eval(
                        unwrapped,
                        eval_chosen,
                        eval_targets,
                        pad_id,
                        accelerator.device,
                        batch_size=eval_batch_size,
                    )
                    eval_results.append({"batch": global_batch_idx, "matched_acc": acc, "matched_margin": margin})
                    if tb_writer is not None:
                        tb_writer.add_scalar("reward/accuracy", acc, global_batch_idx)
                        tb_writer.add_scalar("reward/eval_matched_margin", margin, global_batch_idx)
                    if acc > best_acc or (acc == best_acc and margin > best_margin):
                        best_acc = acc
                        best_margin = margin
                        best_state_dict = {
                            key: value.detach().cpu().clone()
                            for key, value in unwrapped.state_dict().items()
                        }
                        best_cfg = _cfg(model)
                        best_config_snapshot = {
                            "bias": float(getattr(best_cfg, "bias", 0.0)),
                            "normalization_constant": float(getattr(best_cfg, "normalization_constant", 1.0)),
                            "c_coef": float(getattr(best_cfg, "c_coef", 1.0)),
                        }
                    accelerator.print(
                        f"[reward_eval] batch={global_batch_idx}"
                        f" acc={acc:.4f}"
                        f" margin={margin:.4f}"
                        f" loss={loss.item():.4f}"
                        f" online_margin={online_margin_global:.4f}"
                        f" online_acc={online_acc:.4f}"
                        f" epsilon={epsilon_global:.4f}"
                        f" c_coef={c_coef:.4f}"
                    )
                accelerator.wait_for_everyone()
            elif accelerator.is_main_process and global_batch_idx % 100 == 0:
                accelerator.print(
                    f"[reward_update] rollout={cli.rollout_id}"
                    f" batch={global_batch_idx}"
                    f" loss={loss.item():.4f}"
                    f" online_margin={online_margin_global:.4f}"
                    f" online_acc={online_acc:.4f}"
                    f" r_demo={r_demo_global:.4f}"
                    f" r_roll={r_roll_global:.4f}"
                    f" epsilon={epsilon_global:.4f}"
                    f" c_coef={c_coef:.4f}"
                )

    if accelerator.is_main_process and eval_chosen:
        unwrapped = accelerator.unwrap_model(model)
        acc, margin = _run_inline_eval(
            unwrapped,
            eval_chosen,
            eval_targets,
            pad_id,
            accelerator.device,
            batch_size=eval_batch_size,
        )
        eval_results.append({"batch": global_batch_idx, "matched_acc": acc, "matched_margin": margin})
        accelerator.print(f"[reward_eval_final] batch={global_batch_idx} acc={acc:.4f} margin={margin:.4f}")

    if accelerator.is_main_process:
        accelerator.print(f"Reward update completed for rollout {cli.rollout_id}")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        round_id = os.environ.get("ROUND_ID", str(cli.rollout_id))

        if best_state_dict is not None:
            unwrapped.load_state_dict(best_state_dict)
            if best_config_snapshot is not None:
                cfg = _cfg(model)
                for k, v in best_config_snapshot.items():
                    setattr(cfg, k, v)
            accelerator.print(f"[reward] Restored best checkpoint with acc={best_acc:.4f} margin={best_margin:.4f}")

        restored_acc = -1.0
        restored_margin = 0.0
        if eval_chosen:
            restored_acc, restored_margin = _run_inline_eval(
                unwrapped,
                eval_chosen,
                eval_targets,
                pad_id,
                accelerator.device,
                batch_size=eval_batch_size,
            )
            accelerator.print(f"[reward_eval_restored] acc={restored_acc:.4f} margin={restored_margin:.4f}")

        eval_out = reward_dir / f"reward_eval_round_{round_id}.json"
        final_acc = eval_results[-1]["matched_acc"] if eval_results else -1.0
        final_margin = eval_results[-1]["matched_margin"] if eval_results else 0.0
        step_dir = reward_dir / f"step_round{round_id}"
        step_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(step_dir, safe_serialization=False)
        _atomic_save(unwrapped, model_path)

        reloaded_acc = -1.0
        reloaded_margin = 0.0
        reload_acc_gap = None
        reload_margin_gap = None
        reload_verified = False
        if eval_chosen:
            # Validate that the on-disk checkpoint reloads to the same model we just evaluated in memory.
            unwrapped.to("cpu")
            old_model.to("cpu")
            del model
            del old_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reloaded_model = init_reward_model(base_model, str(step_dir))
            reloaded_model.eval()
            reloaded_model.to(accelerator.device)
            reloaded_acc, reloaded_margin = _run_inline_eval(
                reloaded_model,
                eval_chosen,
                eval_targets,
                pad_id,
                accelerator.device,
                batch_size=eval_batch_size,
            )
            reload_acc_gap = abs(reloaded_acc - restored_acc)
            reload_margin_gap = abs(reloaded_margin - restored_margin)
            acc_tol, margin_tol = _reload_eval_tolerances(len(eval_chosen))
            reload_verified = reload_acc_gap <= acc_tol and reload_margin_gap <= margin_tol
            accelerator.print(
                "[reward_eval_reloaded] acc=%.4f margin=%.4f acc_gap=%.6f margin_gap=%.6f"
                % (reloaded_acc, reloaded_margin, reload_acc_gap, reload_margin_gap)
            )
            if tb_writer is not None:
                tb_writer.add_scalar("reward/reloaded_acc", reloaded_acc, global_batch_idx)
                tb_writer.add_scalar("reward/reloaded_margin", reloaded_margin, global_batch_idx)
                tb_writer.add_scalar("reward/save_reload_acc_gap", reload_acc_gap, global_batch_idx)
                tb_writer.add_scalar("reward/save_reload_margin_gap", reload_margin_gap, global_batch_idx)
            if not reload_verified:
                raise RuntimeError(
                    "Reloaded reward checkpoint mismatch: restored_acc=%.6f reloaded_acc=%.6f "
                    "restored_margin=%.6f reloaded_margin=%.6f"
                    % (restored_acc, reloaded_acc, restored_margin, reloaded_margin)
                )

        eval_out.write_text(
            json.dumps(
                {
                    "round_id": round_id,
                    "rollout_id": cli.rollout_id,
                    "best_acc": best_acc,
                    "final_acc": final_acc,
                    "restored_acc": restored_acc,
                    "best_matched_acc": best_acc,
                    "best_matched_margin": best_margin,
                    "final_matched_acc": final_acc,
                    "final_matched_margin": final_margin,
                    "restored_matched_acc": restored_acc,
                    "restored_matched_margin": restored_margin,
                    "reloaded_acc": reloaded_acc,
                    "reloaded_margin": reloaded_margin,
                    "save_reload_acc_gap": reload_acc_gap,
                    "save_reload_margin_gap": reload_margin_gap,
                    "save_reload_verified": reload_verified,
                    "eval_curve": eval_results,
                    "eval_source": eval_stats.get("source", "holdout"),
                    "eval_stats": eval_stats,
                    "demo_match_stats": demo_match_stats,
                    "train_eval_split": split_stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if accelerator.is_main_process and tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
