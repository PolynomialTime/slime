import json
from dataclasses import dataclass
from typing import Iterable

import torch

from slime.utils.types import Sample

from .model import DemoSample, tokenize_prompt_answer


@dataclass
class TokenSample:
    tokens: list[int]
    response_length: int


def load_demo_samples(
    path: str,
    tokenizer,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    apply_chat_template: bool = False,
    apply_chat_template_kwargs: dict | None = None,
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
            demo = tokenize_prompt_answer(
                tokenizer,
                prompt=prompt,
                answer=answer,
                apply_chat_template=apply_chat_template,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )
            if demo.response_length <= 0:
                continue
            samples.append(TokenSample(tokens=demo.tokens, response_length=demo.response_length))
    return samples


def load_rollout_samples(rollout_path: str) -> list[TokenSample]:
    data = torch.load(rollout_path, weights_only=False)
    samples_dict = data.get("samples", [])
    samples: list[TokenSample] = []
    for s in samples_dict:
        sample = Sample.from_dict(s)
        if not sample.tokens or sample.response_length <= 0:
            continue
        samples.append(TokenSample(tokens=sample.tokens, response_length=sample.response_length))
    return samples


def iter_batches(samples: list[TokenSample], batch_size: int) -> Iterable[list[TokenSample]]:
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]
