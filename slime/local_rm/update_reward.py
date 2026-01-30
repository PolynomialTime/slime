import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .data import iter_batches, load_demo_samples, load_rollout_samples
from .model import RunningMeanStd, get_sequence_rewards, init_reward_model, load_tokenizer


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
        p = template.format(rollout_id=rid)
        if os.path.exists(p):
            paths.append(p)
    if not paths:
        paths = [rollout_path]
    return paths


def update_reward(args, rollout_id: int, rollout_path: str) -> None:
    reward_dir = Path(args.reward_model_dir)
    reward_dir.mkdir(parents=True, exist_ok=True)

    base_model = args.reward_model_init or args.hf_checkpoint
    model_path = reward_dir / "latest"

    tokenizer = load_tokenizer(base_model)
    model = init_reward_model(base_model, str(model_path) if model_path.exists() else None)
    if not hasattr(model.config, "c_coef"):
        model.config.c_coef = float(getattr(args, "c_coef_init", 1.0))
    else:
        if not model_path.exists():
            model.config.c_coef = float(getattr(args, "c_coef_init", 1.0))
    model.train()

    old_model = init_reward_model(base_model, str(model_path) if model_path.exists() else None)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    old_model.to(device)

    demo_samples = []
    if args.reward_demo_path:
        demo_samples = load_demo_samples(
            args.reward_demo_path,
            tokenizer=tokenizer,
            prompt_key=args.reward_demo_prompt_key,
            answer_key=args.reward_demo_answer_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )

    rollout_paths = _collect_rollout_paths(args, rollout_id, rollout_path)
    rollout_samples = []
    for p in rollout_paths:
        rollout_samples.extend(load_rollout_samples(p))
    if not demo_samples or not rollout_samples:
        return

    optimizer = optim.AdamW(model.parameters(), lr=args.reward_update_lr)
    rms = RunningMeanStd(device=device)

    c_coef = float(getattr(model.config, "c_coef", 1.0))
    c_coef_min = getattr(args, "c_coef_min", 0.1)
    c_coef_max = getattr(args, "c_coef_max", 10.0)
    coef_scale_up = getattr(args, "coef_scale_up", 1.2)
    coef_scale_down = getattr(args, "coef_scale_down", 0.8)
    target_reward_l2_norm = getattr(args, "target_reward_l2_norm", 5.0)

    pad_id = tokenizer.pad_token_id
    for _ in range(args.reward_update_epochs):
        for demo_batch, roll_batch in zip(
            iter_batches(demo_samples, args.reward_update_batch_size),
            iter_batches(rollout_samples, args.reward_update_batch_size),
        ):
            demo_tokens = [s.tokens for s in demo_batch]
            roll_tokens = [s.tokens for s in roll_batch]

            rewards_demo = get_sequence_rewards(model, demo_tokens, pad_id, device)
            rewards_roll = get_sequence_rewards(model, roll_tokens, pad_id, device)

            with torch.no_grad():
                rewards_demo_old = get_sequence_rewards(old_model, demo_tokens, pad_id, device)
                rewards_roll_old = get_sequence_rewards(old_model, roll_tokens, pad_id, device)

            l_old = rewards_demo.mean() - rewards_roll.mean()

            delta = torch.cat(
                [rewards_demo - rewards_demo_old, rewards_roll - rewards_roll_old], dim=0
            )
            epsilon = torch.sqrt(torch.mean(delta ** 2))

            loss = -(l_old - c_coef * epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rewards_norm = torch.cat([rewards_demo.detach(), rewards_roll.detach()], dim=0)
                raw = rewards_norm * float(model.config.normalization_constant) + float(model.config.bias)
                rms.update_from_batch(raw)
                new_bias = float(rms.mean.item())
                new_std = float(rms.std.item())
                if new_std < 1e-3:
                    new_std = 1e-3
                model.config.bias = new_bias
                model.config.normalization_constant = new_std

            with torch.no_grad():
                hi = target_reward_l2_norm * 1.2
                lo = target_reward_l2_norm * 0.8
                eps_val = epsilon.item()
                if eps_val > hi:
                    c_coef *= coef_scale_up
                elif eps_val < lo:
                    c_coef *= coef_scale_down
                c_coef = max(c_coef_min, min(c_coef, c_coef_max))
                model.config.c_coef = float(c_coef)

    step_dir = reward_dir / f"step_{rollout_id}"
    step_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(step_dir, safe_serialization=False)
    _atomic_save(model, model_path)
