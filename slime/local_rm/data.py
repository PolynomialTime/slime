import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch

from slime.utils.data import read_file_with_source_row_ids
from slime.utils.types import Sample
from slime.utils.text_hygiene import count_non_printing_chars

from .model import build_prompt_text, tokenize_prompt_answer


@dataclass
class TokenSample:
    tokens: list[int]
    response_length: int
    prompt: str | None = None
    raw_prompt: str | None = None
    source_row_id: int | None = None
    # Optional fields populated by load_rollout_samples so downstream pair
    # builders (e.g. in-rollout correctness pairing) can grade responses
    # against ground-truth labels. Demo loaders leave these as None.
    response: str | None = None
    label: str | None = None
    answer_score: float | None = None


@dataclass
class TokenSampleIndex:
    by_source_row_id: dict[int, list[TokenSample]]
    by_prompt: dict[str, list[TokenSample]]


def coerce_source_row_id(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_sample_source_row_id(sample: Sample) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return coerce_source_row_id(metadata.get("source_row_id"))


def init_alignment_stats() -> dict[str, int]:
    return {
        "matched_by_row_id": 0,
        "matched_by_prompt_fallback": 0,
        "missing_row_id_match": 0,
        "legacy_samples_without_source_row_id": 0,
    }


def build_token_sample_index(samples: list[TokenSample]) -> TokenSampleIndex:
    by_source_row_id = defaultdict(list)
    by_prompt = defaultdict(list)
    for sample in samples:
        if sample.source_row_id is not None:
            by_source_row_id[sample.source_row_id].append(sample)
        if sample.prompt is not None:
            by_prompt[sample.prompt].append(sample)
    return TokenSampleIndex(
        by_source_row_id=dict(by_source_row_id),
        by_prompt=dict(by_prompt),
    )


def match_token_sample(
    sample: TokenSample,
    sample_index: TokenSampleIndex,
    *,
    strict_row_id: bool,
    random_prompt_fallback: bool = False,
    stats: dict[str, int] | None = None,
) -> TokenSample | None:
    row_id = sample.source_row_id
    if row_id is not None:
        matches = sample_index.by_source_row_id.get(row_id)
        if matches:
            if stats is not None:
                stats["matched_by_row_id"] = stats.get("matched_by_row_id", 0) + 1
            return matches[0]
        if stats is not None:
            stats["missing_row_id_match"] = stats.get("missing_row_id_match", 0) + 1
        if strict_row_id:
            return None
    else:
        if stats is not None:
            stats["legacy_samples_without_source_row_id"] = stats.get("legacy_samples_without_source_row_id", 0) + 1

    if sample.prompt is None:
        return None

    prompt_matches = sample_index.by_prompt.get(sample.prompt)
    if not prompt_matches:
        return None

    if stats is not None:
        stats["matched_by_prompt_fallback"] = stats.get("matched_by_prompt_fallback", 0) + 1

    if random_prompt_fallback and len(prompt_matches) > 1:
        return random.choice(prompt_matches)
    return prompt_matches[0]


def _sample_is_empty(sample: Sample) -> bool:
    return (sample.response or "").strip() == ""


def _sample_is_eos_only(sample: Sample, stop_token_ids: set[int]) -> bool:
    if sample.response_length <= 0 or not _sample_is_empty(sample) or len(sample.tokens) < sample.response_length:
        return False
    if not stop_token_ids:
        return False
    response_tokens = sample.tokens[-sample.response_length :]
    return bool(response_tokens) and all(token in stop_token_ids for token in response_tokens)


def parse_hh_rlhf_text(text: str) -> list[dict]:
    """Parse hh-rlhf text field ('Human: ...\\n\\nAssistant: ...') into conversation turns."""
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


def load_demo_samples(
    path: str,
    tokenizer,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    apply_chat_template: bool = False,
    apply_chat_template_kwargs: dict | None = None,
    source_row_id_filter: set[int] | None = None,
    prompt_filter: set[str] | None = None,
) -> list[TokenSample]:
    return load_prompt_answer_samples(
        path=path,
        tokenizer=tokenizer,
        prompt_key=prompt_key,
        answer_key=answer_key,
        apply_chat_template=apply_chat_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        source_row_id_filter=source_row_id_filter,
        prompt_filter=prompt_filter,
    )


def load_prompt_answer_samples(
    path: str,
    tokenizer,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    apply_chat_template: bool = False,
    apply_chat_template_kwargs: dict | None = None,
    source_row_id_filter: set[int] | None = None,
    prompt_filter: set[str] | None = None,
) -> list[TokenSample]:
    samples: list[TokenSample] = []
    for fallback_row_id, item in read_file_with_source_row_ids(path):
        prompt = item[prompt_key]
        answer = item[answer_key]
        source_row_id = coerce_source_row_id(item.get("source_row_id"))
        if source_row_id is None:
            source_row_id = fallback_row_id
        prompt_text = build_prompt_text(
            tokenizer,
            prompt,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        if source_row_id_filter is not None or prompt_filter is not None:
            matched_filter = False
            if source_row_id_filter is not None and source_row_id in source_row_id_filter:
                matched_filter = True
            if prompt_filter is not None and prompt_text in prompt_filter:
                matched_filter = True
            if not matched_filter:
                continue
        demo = tokenize_prompt_answer(
            tokenizer,
            prompt=prompt,
            answer=answer,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        if demo.response_length <= 0:
            continue
        samples.append(
            TokenSample(
                tokens=demo.tokens,
                response_length=demo.response_length,
                prompt=prompt_text,
                raw_prompt=prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False),
                source_row_id=source_row_id,
            )
        )
    return samples


def _tokenizer_size(tokenizer) -> int:
    try:
        return len(tokenizer)
    except Exception:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        return int(vocab_size) if vocab_size is not None else 0


def _needs_retokenize_for_reward(tokens: list[int], tokenizer) -> bool:
    vocab_size = _tokenizer_size(tokenizer)
    if vocab_size <= 0:
        return False
    return any((token_id < 0 or token_id >= vocab_size) for token_id in tokens)


def _chat_terminator_id(tokenizer) -> int | None:
    terminator_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if terminator_id is not None and tokenizer.convert_ids_to_tokens(terminator_id) == "<|im_end|>":
        return int(terminator_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return int(eos_id) if eos_id is not None else None


def _retokenize_rollout_sample_for_reward(sample: Sample, tokenizer) -> tuple[list[int], int]:
    prompt = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt, ensure_ascii=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(sample.response or "", add_special_tokens=False)["input_ids"]

    # Completed SGLang rollouts carry the assistant-turn terminator in tokens,
    # but not in decoded response text. Preserve that boundary after retokenizing
    # for reward models that use a different tokenizer from the policy.
    if sample.status == Sample.Status.COMPLETED:
        terminator_id = _chat_terminator_id(tokenizer)
        if terminator_id is not None and (not response_ids or response_ids[-1] != terminator_id):
            response_ids = response_ids + [terminator_id]

    return prompt_ids + response_ids, len(response_ids)


def load_rollout_samples(rollout_path: str, tokenizer=None) -> list[TokenSample]:
    data = torch.load(rollout_path, weights_only=False)
    samples_dict = data.get("samples", [])
    samples: list[TokenSample] = []
    for s in samples_dict:
        sample = Sample.from_dict(s)
        if not sample.tokens or sample.response_length <= 0:
            continue
        prompt = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt, ensure_ascii=False)
        tokens = sample.tokens
        response_length = sample.response_length
        if tokenizer is not None and _needs_retokenize_for_reward(tokens, tokenizer):
            tokens, response_length = _retokenize_rollout_sample_for_reward(sample, tokenizer)
        if not tokens or response_length <= 0:
            continue
        samples.append(
            TokenSample(
                tokens=tokens,
                response_length=response_length,
                prompt=prompt,
                source_row_id=get_sample_source_row_id(sample),
                response=sample.response if sample.response else None,
                label=sample.label if sample.label else None,
            )
        )
    return samples


def summarize_rollout_samples(rollout_path: str, stop_token_ids: list[int] | None = None) -> dict[str, int]:
    data = torch.load(rollout_path, weights_only=False)
    samples_dict = data.get("samples", [])
    stop_token_ids_set = {int(token_id) for token_id in (stop_token_ids or [])}
    summary = {
        "total": 0,
        "empty": 0,
        "eos_only": 0,
        "truncated": 0,
        "response_chars": 0,
        "non_printing_chars": 0,
        "non_printing_samples": 0,
    }
    for s in samples_dict:
        sample = Sample.from_dict(s)
        response_text = sample.response or ""
        non_printing_chars = count_non_printing_chars(response_text)
        summary["total"] += 1
        if sample.status == Sample.Status.TRUNCATED:
            summary["truncated"] += 1
        if _sample_is_empty(sample):
            summary["empty"] += 1
        if _sample_is_eos_only(sample, stop_token_ids_set):
            summary["eos_only"] += 1
        summary["response_chars"] += len(response_text)
        summary["non_printing_chars"] += non_printing_chars
        if non_printing_chars > 0:
            summary["non_printing_samples"] += 1
    return summary


def iter_batches(
    samples: list[TokenSample], batch_size: int, drop_last: bool = False
) -> Iterable[list[TokenSample]]:
    limit = len(samples)
    if drop_last:
        limit = (len(samples) // batch_size) * batch_size
    for i in range(0, limit, batch_size):
        yield samples[i : i + batch_size]
