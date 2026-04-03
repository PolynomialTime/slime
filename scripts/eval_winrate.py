"""
Compute GPT-4o winrate between two models.
Run on the dev machine (requires internet + OpenAI API key).

Usage:
    python scripts/eval_winrate.py \
        --outputs-a /path/to/outputs_irl.jsonl \
        --outputs-b /path/to/outputs_base.jsonl \
        --output /path/to/winrate.json \
        [--api-key sk-...] \
        [--model gpt-4o] \
        [--max-samples 100]
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are an impartial judge evaluating AI assistant responses."

JUDGE_PROMPT = """You are evaluating two AI assistant responses to the same conversation.

Conversation:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is more helpful and appropriate? Reply with exactly one of: "A", "B", or "Tie"."""


async def judge_pair(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, bool]:
    """Returns (parsed_verdict, raw_verdict_text, is_parse_error)."""
    content = JUDGE_PROMPT.format(prompt=prompt, response_a=response_a, response_b=response_b)
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": content},
                    ],
                    max_completion_tokens=10,
                    temperature=0.0,
                )
                verdict = resp.choices[0].message.content.strip()
                verdict_text = verdict.upper()
                m = re.match(r'^\s*\**\s*([AB])\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                if re.search(r'\bTIE\b|\bSAME\b|\bNEITHER\b', verdict_text):
                    return "Tie", verdict, False
                m = re.search(r'\bRESPONSE\s+([AB])\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                m = re.search(r'\b(?:MY\s+(?:ANSWER|CHOICE)|I\s+(?:WOULD\s+)?CHOOSE)[:\s]+([AB])\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                m = re.search(r'\b([AB])\s+IS\s+(?:BETTER|BEST|MORE\s+HELPFUL|PREFERRED)\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                logger.debug("unparseable verdict %r, treating as Tie", verdict)
                return "Tie", verdict, True
            except Exception as e:
                if attempt == 2:
                    logger.warning("judge failed after 3 attempts: %s", e)
                    return "Tie", "", False
                await asyncio.sleep(2 ** attempt)
    return "Tie", "", False


def load_outputs(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def run(args):
    rows_a = load_outputs(args.outputs_a)
    rows_b = load_outputs(args.outputs_b)

    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"Length mismatch: outputs-a has {len(rows_a)} rows, outputs-b has {len(rows_b)}"
        )

    n = len(rows_a)
    if args.max_samples and n > args.max_samples:
        rows_a = rows_a[: args.max_samples]
        rows_b = rows_b[: args.max_samples]
        n = args.max_samples

    logger.info("Evaluating %d pairs with model=%s", n, args.model)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY/OPENROUTER_API_KEY env var")

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or None
    client = AsyncOpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def eval_one(i):
        prompt = rows_a[i]["prompt"]
        resp_a = rows_a[i]["response"]
        resp_b = rows_b[i]["response"]

        swap = random.random() < 0.5
        if swap:
            judge_a, judge_b = resp_b, resp_a
        else:
            judge_a, judge_b = resp_a, resp_b

        verdict, raw_verdict, parse_error = await judge_pair(client, args.model, prompt, judge_a, judge_b, semaphore)

        if verdict == "Tie":
            winner = "tie"
        elif swap:
            winner = "b" if verdict == "A" else "a"
        else:
            winner = "a" if verdict == "A" else "b"

        return {
            "index": i,
            "prompt": prompt[:200],
            "response_a": resp_a[:200],
            "response_b": resp_b[:200],
            "swapped": swap,
            "verdict": verdict,
            "raw_verdict": raw_verdict,
            "parse_error": parse_error,
            "winner": winner,
        }

    tasks = [eval_one(i) for i in range(n)]
    results = await async_tqdm.gather(*tasks, desc="judging")

    a_wins = sum(1 for r in results if r["winner"] == "a")
    b_wins = sum(1 for r in results if r["winner"] == "b")
    ties = sum(1 for r in results if r["winner"] == "tie")
    parse_errors = sum(1 for r in results if r["parse_error"])
    total = len(results)
    winrate_a = (a_wins + 0.5 * ties) / total if total > 0 else 0.0

    summary = {
        "total": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "parse_errors": parse_errors,
        "winrate_a": winrate_a,
        "model": args.model,
        "outputs_a": args.outputs_a,
        "outputs_b": args.outputs_b,
        "samples": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done: total=%d a_wins=%d b_wins=%d ties=%d winrate_a=%.4f",
        total, a_wins, b_wins, ties, winrate_a,
    )
    logger.info("Results saved to %s", args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-a", required=True, help="JSONL from model A")
    parser.add_argument("--outputs-b", required=True, help="JSONL from model B")
    parser.add_argument("--output", required=True, help="Output winrate JSON path")
    parser.add_argument("--api-key", default=None, help="OpenAI/OpenRouter API key")
    parser.add_argument("--base-url", default=None, help="API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API calls")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
