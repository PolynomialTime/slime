import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm

from .data import iter_batches, load_demo_samples, load_rollout_samples
from .model import RunningMeanStd, get_sequence_rewards, init_reward_model, load_tokenizer
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=str, required=True)
    parser.add_argument("--rollout-id", type=int, required=True)
    parser.add_argument("--rollout-path", type=str, required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.args_json, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    args = SimpleNamespace(**cfg_dict)

    configure_logger()
    accelerator = Accelerator()

    base_model = args.reward_model_init or args.hf_checkpoint
    reward_dir = Path(args.reward_model_dir)
    reward_dir.mkdir(parents=True, exist_ok=True)
    model_path = reward_dir / "latest"

    tokenizer = load_tokenizer(base_model)
    # Always start reward model from base weights each update
    model = init_reward_model(base_model, None)
    model.config.c_coef = float(getattr(args, "c_coef_init", 1.0))

    # Old model is the previous reward checkpoint (if exists)
    old_model = init_reward_model(base_model, str(model_path) if model_path.exists() else None)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=args.reward_update_lr)
    model, optimizer = accelerator.prepare(model, optimizer)
    old_model.to(accelerator.device)

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

    rollout_samples = []
    rollout_paths = [cli.rollout_path]
    window = int(getattr(args, "reward_update_rollout_window", 1) or 1)
    if window > 1 and getattr(args, "save_debug_rollout_data", None):
        start = max(0, cli.rollout_id - window + 1)
        rollout_paths = [
            args.save_debug_rollout_data.format(rollout_id=i)
            for i in range(start, cli.rollout_id + 1)
        ]
    for p in rollout_paths:
        if Path(p).exists():
            rollout_samples.extend(load_rollout_samples(p))

    if not demo_samples or not rollout_samples:
        return

    demo_samples = _shard_samples(demo_samples, accelerator.process_index, accelerator.num_processes)
    rollout_samples = _shard_samples(rollout_samples, accelerator.process_index, accelerator.num_processes)

    def _cfg(m):
        return accelerator.unwrap_model(m).config

    pad_id = tokenizer.pad_token_id
    rms = RunningMeanStd(device=accelerator.device)
    c_coef = float(getattr(_cfg(model), "c_coef", 1.0))
    c_coef_min = getattr(args, "c_coef_min", 0.1)
    c_coef_max = getattr(args, "c_coef_max", 10.0)
    coef_scale_up = getattr(args, "coef_scale_up", 1.2)
    coef_scale_down = getattr(args, "coef_scale_down", 0.8)
    target_reward_l2_norm = getattr(args, "target_reward_l2_norm", 5.0)

    for _ in tqdm(
        range(args.reward_update_epochs),
        desc="reward_update_epoch",
        leave=False,
        disable=not accelerator.is_main_process,
    ):
        for demo_batch, roll_batch in tqdm(
            zip(
                iter_batches(demo_samples, args.reward_update_batch_size),
                iter_batches(rollout_samples, args.reward_update_batch_size),
            ),
            desc="reward_update_batch",
            leave=False,
            disable=not accelerator.is_main_process,
        ):
            demo_tokens = [s.tokens for s in demo_batch]
            roll_tokens = [s.tokens for s in roll_batch]

            rewards_demo = get_sequence_rewards(model, demo_tokens, pad_id, accelerator.device)
            rewards_roll = get_sequence_rewards(model, roll_tokens, pad_id, accelerator.device)

            with torch.no_grad():
                rewards_demo_old = get_sequence_rewards(old_model, demo_tokens, pad_id, accelerator.device)
                rewards_roll_old = get_sequence_rewards(old_model, roll_tokens, pad_id, accelerator.device)

            l_old = rewards_demo.mean() - rewards_roll.mean()
            delta = torch.cat(
                [rewards_demo - rewards_demo_old, rewards_roll - rewards_roll_old], dim=0
            )
            epsilon = torch.sqrt(torch.mean(delta ** 2))

            loss = -(l_old - c_coef * epsilon)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                rewards_norm = torch.cat([rewards_demo.detach(), rewards_roll.detach()], dim=0)
                cfg = _cfg(model)
                raw = rewards_norm * float(cfg.normalization_constant) + float(cfg.bias)
                raw_all = accelerator.gather(raw)
                rms.update_from_batch(raw_all)
                new_bias = float(rms.mean.item())
                new_std = float(rms.std.item())
                if new_std < 1e-3:
                    new_std = 1e-3
                cfg.bias = new_bias
                cfg.normalization_constant = new_std

            with torch.no_grad():
                eps_all = accelerator.gather(epsilon.detach())
                epsilon_global = eps_all.mean().item()
                hi = target_reward_l2_norm * 1.2
                lo = target_reward_l2_norm * 0.8
                if epsilon_global > hi:
                    c_coef *= coef_scale_up
                elif epsilon_global < lo:
                    c_coef *= coef_scale_down
                c_coef = max(c_coef_min, min(c_coef, c_coef_max))
                _cfg(model).c_coef = float(c_coef)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        step_dir = reward_dir / f"step_{cli.rollout_id}"
        step_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(step_dir, safe_serialization=False)
        _atomic_save(unwrapped, model_path)


if __name__ == "__main__":
    main()
