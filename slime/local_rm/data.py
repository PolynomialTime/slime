import json
from dataclasses import dataclass
from typing import Iterable

import torch

from slime.utils.types import Sample

from .model import build_prompt_text, tokenize_prompt_answer


@dataclass
class TokenSample:
    tokens: list[int]
    response_length: int
    prompt: str | None = None
    raw_prompt: str | None = None


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
    prompt_filter: set[str] | None = None,
) -> list[TokenSample]:
    return load_prompt_answer_samples(
        path=path,
        tokenizer=tokenizer,
        prompt_key=prompt_key,
        answer_key=answer_key,
        apply_chat_template=apply_chat_template,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        prompt_filter=prompt_filter,
    )


def load_prompt_answer_samples(
    path: str,
    tokenizer,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    apply_chat_template: bool = False,
    apply_chat_template_kwargs: dict | None = None,
    prompt_filter: set[str] | None = None,
) -> list[TokenSample]:
    samples: list[TokenSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item[prompt_key]
            answer = item[answer_key]
            prompt_text = build_prompt_text(
                tokenizer,
                prompt,
                apply_chat_template=apply_chat_template,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )
            if prompt_filter is not None and prompt_text not in prompt_filter:
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
                )
            )
    return samples


def load_rollout_samples(rollout_path: str) -> list[TokenSample]:
    data = torch.load(rollout_path, weights_only=False)
    samples_dict = data.get("samples", [])
    samples: list[TokenSample] = []
    for s in samples_dict:
        sample = Sample.from_dict(s)
        if not sample.tokens or sample.response_length <= 0:
            continue
        prompt = sample.prompt if isinstance(sample.prompt, str) else json.dumps(sample.prompt, ensure_ascii=False)
        samples.append(TokenSample(tokens=sample.tokens, response_length=sample.response_length, prompt=prompt))
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
    }
    for s in samples_dict:
        sample = Sample.from_dict(s)
        summary["total"] += 1
        if sample.status == Sample.Status.TRUNCATED:
            summary["truncated"] += 1
        if _sample_is_empty(sample):
            summary["empty"] += 1
        if _sample_is_eos_only(sample, stop_token_ids_set):
            summary["eos_only"] += 1
    return summary


def iter_batches(
    samples: list[TokenSample], batch_size: int, drop_last: bool = False
) -> Iterable[list[TokenSample]]:
    limit = len(samples)
    if drop_last:
        limit = (len(samples) // batch_size) * batch_size
    for i in range(0, limit, batch_size):
        yield samples[i : i + batch_size]
