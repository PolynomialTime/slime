import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ImportError:
    SummaryWriter = None
    _HAS_TB = False

from slime.utils.data import read_file_with_source_row_ids
from slime.utils.logging_utils import configure_logger

from .data import iter_batches
from .model import (
    RunningMeanStd,
    build_prompt_text,
    get_reward_normalization_stats,
    get_sequence_rewards,
    init_reward_model,
    load_tokenizer,
    sanitize_scalar_model_config,
    tokenize_prompt_answer,
)


@dataclass
class PreferencePair:
    prompt: str
    chosen_tokens: list[int]
    rejected_tokens: list[int]


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _cfg(model, accelerator: Accelerator):
    return accelerator.unwrap_model(model).config


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


def _holdout_prompt(prompt: str, holdout_ratio: float) -> bool:
    if holdout_ratio <= 0.0:
        return False
    if holdout_ratio >= 1.0:
        return True
    digest = hashlib.sha1(prompt.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < holdout_ratio


def _split_pairs_by_prompt(
    pairs: list[PreferencePair],
    holdout_ratio: float,
) -> tuple[list[PreferencePair], list[PreferencePair], dict]:
    unique_prompts = sorted({pair.prompt for pair in pairs})
    if holdout_ratio <= 0.0 or len(unique_prompts) < 2:
        return pairs, [], {
            "total_pairs": len(pairs),
            "train_pairs": len(pairs),
            "eval_pairs": 0,
            "total_prompts": len(unique_prompts),
            "train_prompts": len(unique_prompts),
            "eval_prompts": 0,
            "holdout_ratio": holdout_ratio,
        }

    holdout_prompts = {p for p in unique_prompts if _holdout_prompt(p, holdout_ratio)}
    if not holdout_prompts:
        holdout_prompts = {unique_prompts[0]}
    elif len(holdout_prompts) == len(unique_prompts):
        holdout_prompts.discard(unique_prompts[0])

    train_pairs = [p for p in pairs if p.prompt not in holdout_prompts]
    eval_pairs = [p for p in pairs if p.prompt in holdout_prompts]

    if not train_pairs and eval_pairs:
        salvage = eval_pairs[0].prompt
        train_pairs = [p for p in eval_pairs if p.prompt == salvage]
        eval_pairs = [p for p in eval_pairs if p.prompt != salvage]
        holdout_prompts.discard(salvage)

    return train_pairs, eval_pairs, {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "total_prompts": len(unique_prompts),
        "train_prompts": len(unique_prompts) - len(holdout_prompts),
        "eval_prompts": len(holdout_prompts),
        "holdout_ratio": holdout_ratio,
    }


def _load_preference_pairs(
    path: str,
    tokenizer,
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
    progress_desc: str | None = None,
    progress: bool = False,
) -> tuple[list[PreferencePair], dict]:
    pairs: list[PreferencePair] = []
    total_rows = 0
    skipped_empty = 0
    iterator = read_file_with_source_row_ids(path)
    if progress:
        iterator = tqdm(iterator, desc=progress_desc or f"loading {Path(path).name}")
    for _, item in iterator:
        total_rows += 1
        prompt = item[prompt_key]
        chosen_demo = tokenize_prompt_answer(
            tokenizer,
            prompt=prompt,
            answer=item[chosen_key],
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        rejected_demo = tokenize_prompt_answer(
            tokenizer,
            prompt=prompt,
            answer=item[rejected_key],
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        if chosen_demo.response_length <= 0 or rejected_demo.response_length <= 0:
            skipped_empty += 1
            continue
        prompt_text = build_prompt_text(
            tokenizer,
            prompt,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        pairs.append(
            PreferencePair(
                prompt=prompt_text,
                chosen_tokens=chosen_demo.tokens,
                rejected_tokens=rejected_demo.tokens,
            )
        )
    return pairs, {
        "path": path,
        "rows": total_rows,
        "pairs": len(pairs),
        "skipped_empty": skipped_empty,
    }


def _gather_mean(value: torch.Tensor, accelerator: Accelerator) -> float:
    gathered = accelerator.gather(value.reshape(1))
    return float(gathered.float().mean().item())


def _evaluate_pairs_distributed(
    model,
    pairs: list[PreferencePair],
    pad_id: int,
    accelerator: Accelerator,
    batch_size: int,
) -> tuple[float, float]:
    if not pairs:
        return -1.0, 0.0

    local_pairs = _shard_samples(pairs, accelerator.process_index, accelerator.num_processes)
    was_training = model.training
    model.eval()
    metrics = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
    with torch.no_grad():
        for batch in iter_batches(local_pairs, batch_size, drop_last=False):
            chosen_tokens = [p.chosen_tokens for p in batch]
            rejected_tokens = [p.rejected_tokens for p in batch]
            r_chosen = get_sequence_rewards(model, chosen_tokens, pad_id, accelerator.device)
            r_rejected = get_sequence_rewards(model, rejected_tokens, pad_id, accelerator.device)
            diff = (r_chosen - r_rejected).float()
            metrics[0] += (diff > 0).float().sum().to(metrics.dtype)
            metrics[1] += diff.sum().to(metrics.dtype)
            metrics[2] += diff.numel()
    if _dist_ready():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    total = int(metrics[2].item())
    if total <= 0:
        return -1.0, 0.0
    return metrics[0].item() / total, metrics[1].item() / total


def _evaluate_pairs_local(
    unwrapped_model,
    pairs: list[PreferencePair],
    pad_id: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    if not pairs:
        return -1.0, 0.0

    was_training = unwrapped_model.training
    unwrapped_model.eval()
    correct = 0
    margin_sum = 0.0
    with torch.no_grad():
        for batch in iter_batches(pairs, batch_size, drop_last=False):
            chosen_tokens = [p.chosen_tokens for p in batch]
            rejected_tokens = [p.rejected_tokens for p in batch]
            r_chosen = get_sequence_rewards(unwrapped_model, chosen_tokens, pad_id, device)
            r_rejected = get_sequence_rewards(unwrapped_model, rejected_tokens, pad_id, device)
            diff = (r_chosen - r_rejected).float()
            correct += int((diff > 0).sum().item())
            margin_sum += float(diff.sum().item())
    if was_training:
        unwrapped_model.train()
    return correct / len(pairs), margin_sum / len(pairs)


def _capture_state(unwrapped_model) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    cfg = unwrapped_model.config
    sanitize_scalar_model_config(cfg)
    state = {k: v.detach().cpu().clone() for k, v in unwrapped_model.state_dict().items()}
    snapshot = {
        "bias": float(getattr(cfg, "bias", 0.0)),
        "normalization_constant": float(getattr(cfg, "normalization_constant", 1.0)),
        "c_coef": float(getattr(cfg, "c_coef", 1.0)),
        "reward_max": float(getattr(cfg, "reward_max", 5.0)),
    }
    return state, snapshot


def _restore_state(unwrapped_model, state: dict[str, torch.Tensor], snapshot: dict[str, float]) -> None:
    unwrapped_model.load_state_dict(state)
    for key, value in snapshot.items():
        setattr(unwrapped_model.config, key, value)
    sanitize_scalar_model_config(unwrapped_model.config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=str, required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.args_json, encoding="utf-8") as f:
        args = SimpleNamespace(**json.load(f))

    configure_logger()
    accelerator = Accelerator()

    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_model = args.base_model
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    tb_writer = None
    if accelerator.is_main_process and _HAS_TB:
        tb_dir_override = getattr(args, "tb_dir", None)
        if tb_dir_override:
            tb_dir = Path(tb_dir_override)
        else:
            tb_dir = Path(getattr(args, "tb_root", str(output_dir.parent.parent / "tensorboard_log" / "slime-reward"))) / "bt_pretrain"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
        accelerator.print(f"[bt_tb] Logging to {tb_dir}")

    tokenizer = load_tokenizer(base_model)
    pad_id = tokenizer.pad_token_id

    prompt_key = getattr(args, "prompt_key", "text")
    chosen_key = getattr(args, "chosen_key", "chosen")
    rejected_key = getattr(args, "rejected_key", "rejected")
    apply_chat_template = bool(getattr(args, "apply_chat_template", False))
    apply_chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", None)

    all_pairs, pref_stats = _load_preference_pairs(
        args.pref_path,
        tokenizer,
        prompt_key=prompt_key,
        chosen_key=chosen_key,
        rejected_key=rejected_key,
        apply_chat_template=apply_chat_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        progress_desc="load-train",
        progress=accelerator.is_main_process,
    )
    if not all_pairs:
        raise RuntimeError("No valid preference pairs were loaded for BT pretraining.")

    holdout_ratio = float(getattr(args, "holdout_ratio", 0.1) or 0.0)
    train_pairs_global, eval_pairs, split_stats = _split_pairs_by_prompt(all_pairs, holdout_ratio)
    if not train_pairs_global:
        raise RuntimeError("No training pairs remain after prompt-hash holdout split.")

    external_eval_pairs: list[PreferencePair] = []
    external_eval_stats = {"path": None, "rows": 0, "pairs": 0, "skipped_empty": 0}
    if getattr(args, "eval_path", None):
        external_eval_pairs, external_eval_stats = _load_preference_pairs(
            args.eval_path,
            tokenizer,
            prompt_key=getattr(args, "eval_prompt_key", prompt_key),
            chosen_key=getattr(args, "eval_chosen_key", chosen_key),
            rejected_key=getattr(args, "eval_rejected_key", rejected_key),
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            progress_desc="load-external-eval",
            progress=accelerator.is_main_process,
        )

    train_pairs = _shard_samples(train_pairs_global, accelerator.process_index, accelerator.num_processes)
    if not train_pairs:
        raise RuntimeError(
            f"BT pretraining left rank {accelerator.process_index} without training pairs after sharding."
        )

    batch_size = int(args.batch_size)
    eval_batch_size = int(getattr(args, "eval_batch_size", batch_size) or batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    grad_clip_norm = float(getattr(args, "grad_clip_norm", 1.0) or 1.0)
    weight_decay = float(getattr(args, "weight_decay", 0.0) or 0.0)

    local_num_batches = len(train_pairs) // batch_size
    batch_tensor = torch.tensor(local_num_batches, device=accelerator.device, dtype=torch.long)
    if _dist_ready():
        dist.all_reduce(batch_tensor, op=dist.ReduceOp.MIN)
    global_min_batches = int(batch_tensor.item())
    if global_min_batches <= 0:
        raise RuntimeError(
            f"BT pretraining has zero full batches on at least one rank after sharding. "
            f"local_batches={local_num_batches} batch_size={batch_size}"
        )

    eval_interval = int(getattr(args, "eval_interval", 0) or 0)
    if eval_interval <= 0:
        eval_interval = max(1, global_min_batches // 10)

    if accelerator.is_main_process:
        accelerator.print(
            f"Loaded BT pairs: total={len(all_pairs)} train={len(train_pairs_global)} "
            f"eval={len(eval_pairs)} external_eval={len(external_eval_pairs)} "
            f"skipped_empty={pref_stats['skipped_empty']} holdout_ratio={holdout_ratio}"
        )
        accelerator.print(
            f"Prompt split: total={split_stats['total_prompts']} "
            f"train={split_stats['train_prompts']} eval={split_stats['eval_prompts']}"
        )
        accelerator.print(
            f"After sharding: rank={accelerator.process_index}/{accelerator.num_processes} "
            f"train_pairs={len(train_pairs)} min_batches_per_rank={global_min_batches} "
            f"eval_interval={eval_interval}"
        )

    model = init_reward_model(base_model, None)
    if bool(getattr(args, "gradient_checkpointing", True)):
        if hasattr(model, "lm_backbone") and hasattr(model.lm_backbone, "gradient_checkpointing_enable"):
            model.lm_backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model, optimizer = accelerator.prepare(model, optimizer)

    rms = RunningMeanStd(device=accelerator.device)
    best_acc = -1.0
    best_margin = float("-inf")
    best_step = 0
    best_state_dict = None
    best_config_snapshot = None
    global_step = 0

    for epoch_idx in range(epochs):
        random.shuffle(train_pairs)
        epoch_iter = iter_batches(train_pairs, batch_size, drop_last=True)
        if accelerator.is_main_process:
            epoch_iter = tqdm(
                epoch_iter,
                total=global_min_batches,
                desc=f"bt_pretrain_epoch_{epoch_idx + 1}/{epochs}",
                leave=False,
            )

        for batch_idx, batch in enumerate(epoch_iter):
            if batch_idx >= global_min_batches:
                break

            chosen_tokens = [p.chosen_tokens for p in batch]
            rejected_tokens = [p.rejected_tokens for p in batch]
            r_chosen = get_sequence_rewards(model, chosen_tokens, pad_id, accelerator.device)
            r_rejected = get_sequence_rewards(model, rejected_tokens, pad_id, accelerator.device)
            if not torch.isfinite(r_chosen).all().item() or not torch.isfinite(r_rejected).all().item():
                raise RuntimeError("BT pretraining produced non-finite rewards before optimization.")

            diff = (r_chosen - r_rejected).float()
            loss = -F.logsigmoid(diff).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("BT pretraining loss became non-finite before backward.")

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                cfg = _cfg(model, accelerator)
                bias, norm_c = get_reward_normalization_stats(cfg)
                raw_local = torch.cat([r_chosen.detach(), r_rejected.detach()], dim=0).float() * norm_c + bias
                raw_global = accelerator.gather(raw_local)
                if not rms.update_from_batch(raw_global):
                    raise RuntimeError("RunningMeanStd received no finite reward values during BT pretraining.")
                cfg.bias = float(rms.mean.item())
                cfg.normalization_constant = max(float(rms.std.item()), 1e-3)
                sanitize_scalar_model_config(cfg)

            loss_global = _gather_mean(loss.detach().float(), accelerator)
            online_acc = _gather_mean((diff > 0).float().mean(), accelerator)
            online_margin = _gather_mean(diff.mean(), accelerator)
            r_chosen_mean = _gather_mean(r_chosen.detach().float().mean(), accelerator)
            r_rejected_mean = _gather_mean(r_rejected.detach().float().mean(), accelerator)
            global_step += 1

            if accelerator.is_main_process:
                if tb_writer is not None:
                    tb_writer.add_scalar("reward/loss", loss_global, global_step)
                    tb_writer.add_scalar("reward/online_acc", online_acc, global_step)
                    tb_writer.add_scalar("reward/online_margin", online_margin, global_step)
                    tb_writer.add_scalar("reward/r_chosen", r_chosen_mean, global_step)
                    tb_writer.add_scalar("reward/r_rejected", r_rejected_mean, global_step)
                    tb_writer.add_scalar("reward/bias", float(cfg.bias), global_step)
                    tb_writer.add_scalar(
                        "reward/normalization_constant",
                        float(cfg.normalization_constant),
                        global_step,
                    )
                    tb_writer.add_scalar("reward/lr", lr, global_step)
                if global_step == 1 or global_step % 100 == 0:
                    accelerator.print(
                        f"[bt_train] epoch={epoch_idx + 1}/{epochs} step={global_step} "
                        f"loss={loss_global:.4f} online_acc={online_acc:.4f} "
                        f"online_margin={online_margin:.4f}"
                    )

            if eval_pairs and global_step % eval_interval == 0:
                accelerator.wait_for_everyone()
                eval_acc, eval_margin = _evaluate_pairs_distributed(
                    model, eval_pairs, pad_id, accelerator, eval_batch_size
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if eval_acc > best_acc or (eval_acc == best_acc and eval_margin > best_margin):
                        best_acc = eval_acc
                        best_margin = eval_margin
                        best_step = global_step
                        unwrapped = accelerator.unwrap_model(model)
                        best_state_dict, best_config_snapshot = _capture_state(unwrapped)
                    if tb_writer is not None:
                        tb_writer.add_scalar("reward/eval_acc", eval_acc, global_step)
                        tb_writer.add_scalar("reward/eval_margin", eval_margin, global_step)
                        tb_writer.add_scalar("reward/best_acc", best_acc, global_step)
                    accelerator.print(
                        f"[bt_eval] step={global_step} acc={eval_acc:.4f} margin={eval_margin:.4f} "
                        f"(best_acc={best_acc:.4f}@step{best_step})"
                    )
                accelerator.wait_for_everyone()

    final_eval_acc = -1.0
    final_eval_margin = 0.0
    if eval_pairs:
        accelerator.wait_for_everyone()
        final_eval_acc, final_eval_margin = _evaluate_pairs_distributed(
            model, eval_pairs, pad_id, accelerator, eval_batch_size
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if best_state_dict is None or final_eval_acc > best_acc or (
                final_eval_acc == best_acc and final_eval_margin > best_margin
            ):
                best_acc = final_eval_acc
                best_margin = final_eval_margin
                best_step = global_step
                unwrapped = accelerator.unwrap_model(model)
                best_state_dict, best_config_snapshot = _capture_state(unwrapped)
            accelerator.print(
                f"[bt_eval_end_of_training] acc={final_eval_acc:.4f} margin={final_eval_margin:.4f}"
            )
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if best_state_dict is not None and best_config_snapshot is not None:
            _restore_state(unwrapped, best_state_dict, best_config_snapshot)
            accelerator.print(
                f"[bt_restore] restored best step={best_step} acc={best_acc:.4f} margin={best_margin:.4f}"
            )
        else:
            sanitize_scalar_model_config(unwrapped.config)

        unwrapped.config.c_coef = float(getattr(args, "c_coef_init", 0.5))
        sanitize_scalar_model_config(unwrapped.config)

        external_eval_acc = -1.0
        external_eval_margin = 0.0
        if external_eval_pairs:
            external_eval_acc, external_eval_margin = _evaluate_pairs_local(
                unwrapped, external_eval_pairs, pad_id, accelerator.device, eval_batch_size
            )
            accelerator.print(
                f"[bt_external_eval] acc={external_eval_acc:.4f} margin={external_eval_margin:.4f}"
            )

        _atomic_save(unwrapped, output_dir)
        report = {
            "base_model": base_model,
            "output_dir": str(output_dir),
            "pref_path": args.pref_path,
            "eval_path": getattr(args, "eval_path", None),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "holdout_ratio": holdout_ratio,
            "best_acc": best_acc,
            "best_margin": best_margin,
            "best_step": best_step,
            "final_holdout_acc": final_eval_acc,
            "final_holdout_margin": final_eval_margin,
            "external_eval_acc": external_eval_acc,
            "external_eval_margin": external_eval_margin,
            "bias": float(getattr(unwrapped.config, "bias", 0.0)),
            "normalization_constant": float(getattr(unwrapped.config, "normalization_constant", 1.0)),
            "c_coef": float(getattr(unwrapped.config, "c_coef", 1.0)),
            "total_pairs": len(all_pairs),
            "train_pairs": len(train_pairs_global),
            "eval_pairs": len(eval_pairs),
            "external_eval_pairs": len(external_eval_pairs),
            "pref_stats": pref_stats,
            "external_eval_stats": external_eval_stats,
            "train_eval_split": split_stats,
        }
        (output_dir.parent / "bt_pretrain_eval.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        accelerator.print(f"[bt_save] saved checkpoint to {output_dir}, report to {output_dir.parent}/bt_pretrain_eval.json")

        if tb_writer is not None:
            tb_writer.add_scalar("reward/final_holdout_acc", final_eval_acc, global_step)
            tb_writer.add_scalar("reward/final_holdout_margin", final_eval_margin, global_step)
            if external_eval_pairs:
                tb_writer.add_scalar("reward/external_eval_acc", external_eval_acc, global_step)
                tb_writer.add_scalar("reward/external_eval_margin", external_eval_margin, global_step)
            tb_writer.flush()
            tb_writer.close()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
