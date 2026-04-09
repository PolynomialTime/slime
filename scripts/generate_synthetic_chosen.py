#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from synthetic_teacher_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    append_jsonl,
    build_async_client,
    configure_logging,
    count_jsonl_lines,
    generate_text,
    logger,
    read_jsonl,
    resolve_api_key,
    resolve_base_url,
)

SYSTEM_PROMPTS = {
    "v1": (
        "You are writing a high-quality reference answer for assistant training. "
        "Answer the user task directly and stay on task. "
        "Do not add meta-commentary, apologies, refusal boilerplate, or statements about being an AI unless the prompt explicitly requires that. "
        "Do not claim personal real-world experiences, memories, or actions; if the prompt asks for a personal story, provide a clean example answer instead of talking about yourself as an AI. "
        "Do not ask the user to check websites, contact others, or provide more context when a best-effort direct answer is possible. "
        "Do not invent unsupported specific facts; if the prompt is underspecified, give a brief direct answer based on what is known from the prompt or answer that the information is not provided. "
        "For concrete real-world questions with missing facts, prefer a brief 'not enough information provided' style answer over guessing. "
        "For classification, extraction, translation, labeling, multiple-choice, or other short-form tasks, return the minimal task-native answer. "
        "When the prompt provides answer options or label names, choose the best option directly and return just that task-native answer. "
        "For open-ended tasks, provide a substantive answer with the level of detail the task asks for."
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic chosen answers with a teacher model.")
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--prompt-key", type=str, default="text", help="Input JSON key containing the prompt.")
    parser.add_argument("--output-key", type=str, default="chosen", help="Output JSON key for the synthetic answer.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Teacher model name.")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the teacher endpoint.")
    parser.add_argument("--prompt-version", type=str, default=DEFAULT_PROMPT_VERSION, choices=sorted(SYSTEM_PROMPTS))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128, help="How many pending rows to schedule at once.")
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--num-shards", type=int, default=1, help="Split the input into this many contiguous shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to process.")
    parser.add_argument("--resume", action="store_true", help="Resume by skipping the number of rows already written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on input rows for debugging.")
    return parser.parse_args()


def _normalize_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    return json.dumps(prompt, ensure_ascii=False)


def _build_messages(prompt_text: str, prompt_version: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": prompt_text},
    ]


def _validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise SystemExit(f"--num-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(
            f"--shard-index must be in [0, {num_shards - 1}] when --num-shards={num_shards}, got {shard_index}"
        )


def _shard_bounds(total_rows: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    base = total_rows // num_shards
    extra = total_rows % num_shards
    start = shard_index * base + min(shard_index, extra)
    size = base + (1 if shard_index < extra else 0)
    end = start + size
    return start, end


async def _generate_row(
    client: AsyncOpenAI,
    row: dict,
    *,
    prompt_key: str,
    output_key: str,
    model: str,
    prompt_version: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
) -> dict:
    prompt_value = row[prompt_key]
    prompt_text = _normalize_prompt(prompt_value)
    if not prompt_text:
        raise ValueError("encountered empty prompt")
    chosen = await asyncio.wait_for(
        generate_text(
            client,
            model,
            _build_messages(prompt_text, prompt_version),
            semaphore=semaphore,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        timeout=request_timeout + 30,
    )
    return {
        prompt_key: prompt_value,
        output_key: chosen,
    }


async def main_async(args: argparse.Namespace) -> None:
    configure_logging()
    _validate_shard_args(args.num_shards, args.shard_index)

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set OPENAI_API_KEY / WINRATE_API_KEY / ANTHROPIC_AUTH_TOKEN.")
    base_url = resolve_base_url(args.base_url)

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    shard_start, shard_end = _shard_bounds(len(rows), args.num_shards, args.shard_index)
    rows = rows[shard_start:shard_end]

    completed = count_jsonl_lines(args.output)
    if completed:
        if not args.resume:
            raise SystemExit(f"Output already exists with {completed} rows: {args.output}. Re-run with --resume.")
        if completed > len(rows):
            raise SystemExit(
                f"Output has {completed} rows but input only has {len(rows)} rows: {args.output}"
            )

    pending_rows = rows[completed:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        args.output.touch(exist_ok=True)
        logger.info(
            "Nothing to do: shard %d/%d is empty for %s",
            args.shard_index,
            args.num_shards,
            args.input,
        )
        return
    if not pending_rows:
        logger.info(
            "Nothing to do: shard %d/%d output %s already has %d rows",
            args.shard_index,
            args.num_shards,
            args.output,
            completed,
        )
        return

    client = build_async_client(api_key=api_key, base_url=base_url, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    batch_size = max(1, args.batch_size)

    started = time.time()
    total = len(rows)
    logger.info(
        "Generating synthetic chosen: input=%s output=%s model=%s base_url=%s resume=%s remaining=%s",
        args.input,
        args.output,
        args.model,
        base_url,
        args.resume,
        len(pending_rows),
    )
    logger.info(
        "Shard selection: shard=%d/%d start=%d end=%d shard_rows=%d",
        args.shard_index,
        args.num_shards,
        shard_start,
        shard_end,
        len(rows),
    )

    for offset in range(0, len(pending_rows), batch_size):
        batch = pending_rows[offset : offset + batch_size]
        tasks = [
            _generate_row(
                client,
                row,
                prompt_key=args.prompt_key,
                output_key=args.output_key,
                model=args.model,
                prompt_version=args.prompt_version,
                semaphore=semaphore,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
            )
            for row in batch
        ]
        out_rows = await asyncio.gather(*tasks)
        append_jsonl(args.output, out_rows)

        done = completed + offset + len(out_rows)
        written_now = offset + len(out_rows)
        elapsed = max(time.time() - started, 1e-6)
        rate = written_now / elapsed
        logger.info(
            "Progress: %d/%d rows written (%.2f%%, %.2f rows/s) shard=%d/%d",
            done,
            total,
            100.0 * done / total,
            rate,
            args.shard_index,
            args.num_shards,
        )


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
