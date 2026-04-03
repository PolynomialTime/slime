import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    SummaryWriter = None
    _HAS_TB = False

from .data import iter_batches, load_demo_samples, load_rollout_samples
from .model import RunningMeanStd, get_sequence_rewards, init_reward_model, load_tokenizer, tokenize_prompt_answer
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


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_eval_data(args, tokenizer):
    """Preload eval chosen/rejected token pairs."""
    eval_path = getattr(args, "reward_eval_path", None) or getattr(args, "reward_demo_path", None)
    if not eval_path:
        return [], []
    prompt_key = getattr(args, "reward_eval_prompt_key", None) or getattr(args, "reward_demo_prompt_key", "text")
    chosen_key = getattr(args, "reward_eval_chosen_key", "chosen")
    rejected_key = getattr(args, "reward_eval_rejected_key", "rejected")
    max_samples = getattr(args, "reward_eval_max_samples", None)
    apply_ct = getattr(args, "apply_chat_template", False)
    apply_ct_kwargs = getattr(args, "apply_chat_template_kwargs", None)

    chosen_tokens, rejected_tokens = [], []
    total = 0
    for item in _iter_jsonl(eval_path):
        if max_samples is not None and total >= max_samples:
            break
        if prompt_key not in item or chosen_key not in item or rejected_key not in item:
            continue
        c = tokenize_prompt_answer(tokenizer, item[prompt_key], item[chosen_key], apply_ct, apply_ct_kwargs)
        r = tokenize_prompt_answer(tokenizer, item[prompt_key], item[rejected_key], apply_ct, apply_ct_kwargs)
        if c.response_length <= 0 or r.response_length <= 0:
            continue
        chosen_tokens.append(c.tokens)
        rejected_tokens.append(r.tokens)
        total += 1
    return chosen_tokens, rejected_tokens


def _run_inline_eval(model, chosen_tokens, rejected_tokens, pad_id, device, batch_size=8):
    """Run chosen vs rejected eval on current model, return accuracy."""
    if not chosen_tokens:
        return -1.0
    correct = 0
    total = len(chosen_tokens)
    model.eval()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            c_batch = chosen_tokens[i : i + batch_size]
            r_batch = rejected_tokens[i : i + batch_size]
            c_scores = get_sequence_rewards(model, c_batch, pad_id, device)
            r_scores = get_sequence_rewards(model, r_batch, pad_id, device)
            correct += (c_scores > r_scores).sum().item()
    model.train()
    return correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=str, required=True)
    parser.add_argument("--rollout-id", type=int, required=True)
    parser.add_argument("--rollout-path", type=str, required=True)
    return parser.parse_args()


def _reward_tb_dir(reward_dir: Path, rollout_id: int) -> Path:
    slime_root = reward_dir.parents[1] if len(reward_dir.parents) >= 2 else reward_dir.parent
    round_id = os.environ.get('ROUND_ID', str(rollout_id))
    return slime_root / 'tensorboard_log' / 'slime-reward' / f'round{round_id}'


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
    # Warm-start reward model from previous checkpoint when available
    if model_path.exists():
        model = init_reward_model(base_model, str(model_path))
    else:
        model = init_reward_model(base_model, None)
    model.config.c_coef = float(getattr(args, "c_coef_init", 1.0))
    model.train()

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

    tb_writer = None
    if accelerator.is_main_process and _HAS_TB:
        tb_dir = _reward_tb_dir(reward_dir, cli.rollout_id)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
        accelerator.print(f'[reward_tb] Logging to {tb_dir}')

    if accelerator.is_main_process:
        accelerator.print(f"Loaded {len(demo_samples)} demo samples, {len(rollout_samples)} rollout samples")

    # Preload eval data
    eval_chosen, eval_rejected = _load_eval_data(args, tokenizer)
    if accelerator.is_main_process:
        accelerator.print(f"Loaded {len(eval_chosen)} eval pairs for inline eval")

    demo_samples = _shard_samples(demo_samples, accelerator.process_index, accelerator.num_processes)
    rollout_samples = _shard_samples(rollout_samples, accelerator.process_index, accelerator.num_processes)

    if accelerator.is_main_process:
        accelerator.print(f"After sharding (process {accelerator.process_index}/{accelerator.num_processes}): {len(demo_samples)} demo samples, {len(rollout_samples)} rollout samples")

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

    num_demo_batches = len(demo_samples) // args.reward_update_batch_size
    num_roll_batches = len(rollout_samples) // args.reward_update_batch_size
    num_training_batches_local = min(num_demo_batches, num_roll_batches)
    # Sync min batch count across all ranks to prevent NCCL timeout
    import torch as _torch
    _local = _torch.tensor(num_training_batches_local, device=accelerator.device)
    _global_min = accelerator.reduce(_local, reduction="min")
    num_training_batches = int(_global_min.item())

    if accelerator.is_main_process:
        accelerator.print(f"Batch size: {args.reward_update_batch_size}")
        accelerator.print(f"Demo batches: {num_demo_batches}, Rollout batches: {num_roll_batches}")
        accelerator.print(f"Training iterations per epoch: {num_training_batches}")
        if num_demo_batches != num_roll_batches:
            demo_used = num_training_batches * args.reward_update_batch_size
            roll_used = num_training_batches * args.reward_update_batch_size
            accelerator.print(f"⚠️  WARNING: Batch count mismatch detected!")
            accelerator.print(f"   - Demo samples used: ~{demo_used}/{len(demo_samples)} ({100*demo_used/len(demo_samples):.1f}%)")
            accelerator.print(f"   - Rollout samples used: ~{roll_used}/{len(rollout_samples)} ({100*roll_used/len(rollout_samples):.1f}%)")

    # Eval every N batches
    eval_interval = max(1, num_training_batches // 10)  # ~10 evals per epoch
    eval_results = []
    global_batch_idx = 0

    for epoch in tqdm(
        range(args.reward_update_epochs),
        desc="reward_update_epoch",
        leave=False,
        disable=not accelerator.is_main_process,
    ):
        for demo_batch, roll_batch in tqdm(
            zip(
                iter_batches(demo_samples, args.reward_update_batch_size, drop_last=True),
                iter_batches(rollout_samples, args.reward_update_batch_size, drop_last=True),
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
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

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

            global_batch_idx += 1

            if tb_writer is not None:
                tb_writer.add_scalar('reward/loss', loss.item(), global_batch_idx)
                tb_writer.add_scalar('reward/irl_margin', l_old.item(), global_batch_idx)
                tb_writer.add_scalar('reward/r_demo', rewards_demo.mean().item(), global_batch_idx)
                tb_writer.add_scalar('reward/r_roll', rewards_roll.mean().item(), global_batch_idx)
                tb_writer.add_scalar('reward/epsilon', epsilon_global, global_batch_idx)
                tb_writer.add_scalar('reward/c_coef', c_coef, global_batch_idx)

            # Inline eval every eval_interval batches
            if accelerator.is_main_process and eval_chosen and global_batch_idx % eval_interval == 0:
                unwrapped = accelerator.unwrap_model(model)
                acc = _run_inline_eval(unwrapped, eval_chosen, eval_rejected, pad_id, accelerator.device)
                eval_results.append({"batch": global_batch_idx, "accuracy": acc})
                if tb_writer is not None:
                    tb_writer.add_scalar('reward/accuracy', acc, global_batch_idx)
                accelerator.print(
                    f"[reward_eval] batch={global_batch_idx}"
                    f"  acc={acc:.4f}"
                    f"  loss={loss.item():.4f}"
                    f"  irl_margin={l_old.item():.4f}"
                    f"  epsilon={epsilon_global:.4f}"
                    f"  c_coef={c_coef:.4f}"
                )

            elif accelerator.is_main_process and global_batch_idx % 100 == 0:
                accelerator.print(
                    f"[reward_update] rollout={cli.rollout_id}"
                    f"  batch={global_batch_idx}"
                    f"  loss={loss.item():.4f}"
                    f"  irl_margin={l_old.item():.4f}"
                    f"  r_demo={rewards_demo.mean().item():.4f}"
                    f"  r_roll={rewards_roll.mean().item():.4f}"
                    f"  epsilon={epsilon_global:.4f}"
                    f"  c_coef={c_coef:.4f}"
                )

    # Final eval
    if accelerator.is_main_process and eval_chosen:
        unwrapped = accelerator.unwrap_model(model)
        acc = _run_inline_eval(unwrapped, eval_chosen, eval_rejected, pad_id, accelerator.device)
        eval_results.append({"batch": global_batch_idx, "accuracy": acc})
        accelerator.print(f"[reward_eval_final] batch={global_batch_idx} acc={acc:.4f}")

    if accelerator.is_main_process:
        accelerator.print(f"Reward update completed for rollout {cli.rollout_id}")

    if accelerator.is_main_process and tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        step_dir = reward_dir / f"step_{cli.rollout_id}"
        step_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(step_dir, safe_serialization=False)
        _atomic_save(unwrapped, model_path)

        # Save eval results
        eval_out = reward_dir / f"reward_eval_rollout_{cli.rollout_id}.json"
        final_acc = eval_results[-1]["accuracy"] if eval_results else -1
        eval_out.write_text(json.dumps({
            "rollout_id": cli.rollout_id,
            "accuracy": final_acc,
            "eval_curve": eval_results,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
