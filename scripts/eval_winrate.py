"""
Compute GPT-4o winrate between two models.
Run on the dev machine (requires internet + OpenAI API key).

Usage:
    python scripts/eval_winrate.py \
        --outputs-a /path/to/outputs_irl.jsonl \
        --outputs-b /path/to/outputs_base.jsonl \
        --output /path/to/winrate.json \
        [--mode blind|reference] \
        [--reference /path/to/reference.jsonl] \
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
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_JUDGE_ATTEMPTS = 24
MAX_UNPARSEABLE_VERDICTS = 8


def extract_verdict_text(resp: Any) -> str:
    if hasattr(resp, "choices"):
        return resp.choices[0].message.content.strip()
    if isinstance(resp, str):
        snippet = resp.strip()[:200]
        if snippet.lower().startswith("<!doctype html") or snippet.lower().startswith("<html"):
            raise RuntimeError(
                "received HTML instead of OpenAI JSON response; check that base_url points to an OpenAI-compatible /v1 endpoint"
            )
        raise RuntimeError(f"unexpected string response from API: {snippet!r}")
    raise RuntimeError(f"unexpected response type from API: {type(resp).__name__}")


def is_content_filter_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if err.get("code") in ("content_filter", "data_inspection_failed"):
            return True
        err_type = err.get("type")
        if err_type == "data_inspection_failed":
            return True
    code = getattr(exc, "code", None)
    if code in ("content_filter", "data_inspection_failed"):
        return True
    message = str(exc).lower()
    return (
        "content_filter" in message
        or "content management policy" in message
        or "data_inspection_failed" in message
        or "inappropriate content" in message
    )


def get_error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        message = err.get("message")
        if message:
            return str(message)
    return str(exc)


def is_repeat_failed_request_error(exc: Exception) -> bool:
    message = get_error_message(exc)
    lowered = message.lower()
    return (
        "same request has failed before" in lowered
        or "相同的请求之前已经失败" in message
    )


def add_retry_nonce(content: str, nonce: int) -> str:
    if nonce <= 0:
        return content
    return (
        f"{content}\n\n"
        f"[Retry nonce {nonce}. This line is metadata for transport retries only; ignore it when judging.]"
    )

JUDGE_SYSTEM = (
    "You are an impartial judge evaluating AI assistant responses. "
    "You must output exactly one token: A, B, or Tie. "
    "Do not include any explanation, reasoning, punctuation, or extra words."
)

BLIND_JUDGE_PROMPT = """You are evaluating two AI assistant responses to the same conversation.

Conversation:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is more helpful and appropriate?

Rules:
- Return exactly one token: A, B, or Tie
- Do not explain your answer
- Do not output any other text

Answer:"""

REFERENCE_JUDGE_PROMPT = """You are evaluating two AI assistant responses to the same conversation using a reference answer.

Conversation:
{prompt}

Reference Answer:
{reference}

Response A:
{response_a}

Response B:
{response_b}

Which response better matches the reference answer's task completion, correctness, and required answer form?

Rules:
- Prefer the response that is more faithful to the reference answer's correctness, constraints, and answer form.
- Do not reward extra verbosity, generic helpfulness, hedging, or added explanation unless it clearly improves fidelity to the reference answer.
- If the task expects a constrained answer form (for example Yes/No, True/False, a multiple-choice option, a numbered option, or a short label), prefer the response that follows that form more faithfully.
- If one response is longer but drifts away from the reference answer's format or intent, prefer the shorter on-target response.
- Return exactly one token: A, B, or Tie
- Do not explain your answer
- Do not output any other text

Answer:"""

FAST_PATH_STOP_MARKERS = (
    "\n",
    "stream of consciousness:",
    "reasoning:",
    "analysis:",
    "explanation:",
)


async def judge_pair(
    client: AsyncOpenAI,
    model: str,
    content: str,
    semaphore: asyncio.Semaphore,
    fallback_client: AsyncOpenAI | None = None,
    fallback_model: str | None = None,
) -> tuple[str, str, bool]:
    """Returns (parsed_verdict, raw_verdict_text, is_parse_error)."""
    async with semaphore:
        active_client = client
        active_model = model
        active_content = content
        using_fallback = False
        attempt = 0
        unparseable_count = 0
        retry_nonce = 0
        while True:
            attempt += 1
            try:
                resp = await active_client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": active_content},
                    ],
                    max_completion_tokens=4,
                    temperature=0.0,
                )
                verdict = extract_verdict_text(resp)
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
                m = re.search(r'\bBETTER\s+(?:RESPONSE|ANSWER)\s+IS\s+([AB])\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                m = re.search(r'\bI\s+PREFER\s+([AB])\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                m = re.search(r'\b(?:ANSWER|VERDICT)[:\s]+([AB])\b', verdict_text)
                if m:
                    return m.group(1), verdict, False
                unparseable_count += 1
                if unparseable_count >= MAX_UNPARSEABLE_VERDICTS:
                    logger.error(
                        "giving up after %d unparseable verdicts; marking sample as parse_error: %r",
                        unparseable_count,
                        verdict,
                    )
                    return "Tie", verdict, True
                delay_s = min(30, max(1, attempt // 5))
                logger.warning(
                    "unparseable verdict on attempt %d, retrying in %ss: %r",
                    attempt,
                    delay_s,
                    verdict,
                )
                await asyncio.sleep(delay_s)
            except Exception as e:
                if is_content_filter_error(e):
                    if fallback_client is not None and fallback_model and not using_fallback:
                        logger.warning(
                            "primary judge blocked by content filter; switching to fallback model=%s",
                            fallback_model,
                        )
                        active_client = fallback_client
                        active_model = fallback_model
                        using_fallback = True
                        attempt = 0
                        await asyncio.sleep(1)
                        continue
                    message = get_error_message(e)
                    scope = "fallback model" if using_fallback else "primary judge with no fallback configured"
                    logger.warning(
                        "judge request blocked by content filter on %s; marking sample as parse_error: %s",
                        scope,
                        message,
                    )
                    return "Tie", f"CONTENT_FILTER: {message}", True
                if is_repeat_failed_request_error(e):
                    retry_nonce += 1
                    active_content = add_retry_nonce(content, retry_nonce)
                    delay_s = min(5, retry_nonce)
                    logger.warning(
                        "judge request hit repeat-failed cache on attempt %d; varying payload and retrying in %ss: %s",
                        attempt,
                        delay_s,
                        get_error_message(e),
                    )
                    await asyncio.sleep(delay_s)
                    continue
                if attempt >= MAX_JUDGE_ATTEMPTS:
                    message = get_error_message(e)
                    logger.error(
                        "giving up after %d failed judge attempts; marking sample as parse_error: %s",
                        attempt,
                        message,
                    )
                    return "Tie", f"ERROR: {message}", True
                delay_s = min(30, 2 ** min(attempt - 1, 4))
                logger.warning(
                    "judge request failed on attempt %d, retrying in %ss: %s",
                    attempt,
                    delay_s,
                    e,
                )
                await asyncio.sleep(delay_s)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_required_str(row: dict, key: str, path: str, index: int) -> str:
    if key not in row:
        raise KeyError(f"Missing key {key!r} in {path} at row {index}")
    value = row[key]
    if value is None:
        return ""
    return str(value)


def build_blind_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return BLIND_JUDGE_PROMPT.format(prompt=prompt, response_a=response_a, response_b=response_b)


def build_reference_prompt(prompt: str, reference: str, response_a: str, response_b: str) -> str:
    return REFERENCE_JUDGE_PROMPT.format(
        prompt=prompt,
        reference=reference,
        response_a=response_a,
        response_b=response_b,
    )


def build_output_sample(
    *,
    index: int,
    prompt: str,
    response_a: str,
    response_b: str,
    swapped: bool,
    verdict: str,
    raw_verdict: str,
    parse_error: bool,
    winner: str,
    judge_source: str,
    reference: str | None = None,
    fast_path_signature: str | None = None,
) -> dict:
    sample = {
        "index": index,
        "prompt": prompt,
        "prompt_len": len(prompt),
        "response_a": response_a,
        "response_a_len": len(response_a),
        "response_b": response_b,
        "response_b_len": len(response_b),
        "responses_exact_match": response_a == response_b,
        "responses_strip_match": response_a.strip() == response_b.strip(),
        "swapped": swapped,
        "verdict": verdict,
        "raw_verdict": raw_verdict,
        "parse_error": parse_error,
        "winner": winner,
        "judge_source": judge_source,
    }
    if reference is not None:
        sample["reference"] = reference
        sample["reference_len"] = len(reference)
    if fast_path_signature is not None:
        sample["fast_path_signature"] = fast_path_signature
    return sample


def leading_answer_segment(text: str) -> str:
    segment = str(text or "").strip()
    lowered = segment.lower()
    cut = len(segment)
    for marker in FAST_PATH_STOP_MARKERS:
        idx = lowered.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    segment = segment[:cut].strip()
    if not segment:
        return ""
    first_line = segment.splitlines()[0].strip()
    return normalize_space(first_line)


def extract_constraint_signature(text: str) -> tuple[str, str] | None:
    segment = leading_answer_segment(text)
    if not segment:
        return None

    upper = segment.upper()
    lower = segment.lower()

    bool_match = re.match(r"^(yes|no|true|false)\b", lower)
    if bool_match:
        return ("bool", bool_match.group(1))

    letter_patterns = (
        r"^\(\s*([A-H])\s*\)",
        r"^(?:OPTION\s+)?([A-H])(?=\s*[\).:\-]|(?:\s+\d)|\s*$)",
    )
    for pattern in letter_patterns:
        match = re.match(pattern, upper)
        if match:
            return ("letter", match.group(1))

    number_patterns = (
        r"^\(\s*(\d{1,3})\s*\)",
        r"^(?:OPTION\s+)?(\d{1,3})(?=\s*[\).:\-]|\s*$)",
    )
    for pattern in number_patterns:
        match = re.match(pattern, upper)
        if match:
            return ("number", match.group(1))

    if (
        len(segment) <= 40
        and not re.search(r"[.!?;,]", segment)
        and 1 <= len(segment.split()) <= 4
    ):
        return ("short_label", lower)

    return None


def signature_matches(text: str, signature: tuple[str, str]) -> bool:
    candidate = extract_constraint_signature(text)
    return candidate == signature


def maybe_fast_path_winner(
    prompt: str,
    response_a: str,
    response_b: str,
    reference: str,
    enabled: bool,
) -> dict | None:
    if not enabled:
        return None

    signature = extract_constraint_signature(reference)
    if signature is None:
        return None

    match_a = signature_matches(response_a, signature)
    match_b = signature_matches(response_b, signature)
    if match_a == match_b:
        return None

    winner = "a" if match_a else "b"
    verdict = "A" if winner == "a" else "B"
    return {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "reference": reference,
        "swapped": False,
        "verdict": verdict,
        "raw_verdict": f"FAST_PATH:{verdict}",
        "parse_error": False,
        "winner": winner,
        "judge_source": "fast_path",
        "fast_path_signature": {"type": signature[0], "value": signature[1]},
    }


def validate_pair_alignment(
    rows_a: list[dict],
    rows_b: list[dict],
    rows_ref: list[dict] | None,
    reference_path: str | None,
    reference_prompt_key: str,
) -> None:
    for i, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        prompt_a = normalize_space(get_required_str(row_a, "prompt", "outputs_a", i))
        prompt_b = normalize_space(get_required_str(row_b, "prompt", "outputs_b", i))
        if prompt_a != prompt_b:
            raise ValueError(f"Prompt mismatch between outputs-a and outputs-b at row {i}")
        if rows_ref is not None and reference_path is not None:
            prompt_ref = normalize_space(get_required_str(rows_ref[i], reference_prompt_key, reference_path, i))
            if prompt_a != prompt_ref:
                raise ValueError(
                    f"Prompt mismatch between outputs and reference at row {i}: {reference_path}"
                )


def init_clients(args) -> tuple[AsyncOpenAI, AsyncOpenAI | None]:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY/OPENROUTER_API_KEY env var")

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or None
    client = AsyncOpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    fallback_model = args.fallback_model or os.environ.get("WINRATE_FALLBACK_MODEL") or None
    fallback_api_key = (
        args.fallback_api_key
        or os.environ.get("WINRATE_FALLBACK_API_KEY")
        or (api_key if fallback_model else None)
    )
    fallback_base_url = (
        args.fallback_base_url
        or os.environ.get("WINRATE_FALLBACK_BASE_URL")
        or (base_url if fallback_model else None)
    )
    fallback_client = None
    if fallback_model:
        fallback_client = AsyncOpenAI(
            api_key=fallback_api_key,
            **({"base_url": fallback_base_url} if fallback_base_url else {}),
        )
    return client, fallback_client


def build_model_job(
    index: int,
    prompt: str,
    response_a: str,
    response_b: str,
    reference: str | None,
    mode: str,
) -> dict:
    swap = random.random() < 0.5
    judge_a, judge_b = (response_b, response_a) if swap else (response_a, response_b)
    if mode == "reference":
        content = build_reference_prompt(prompt, reference or "", judge_a, judge_b)
    else:
        content = build_blind_prompt(prompt, judge_a, judge_b)
    return {
        "index": index,
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "reference": reference,
        "swapped": swap,
        "content": content,
    }


async def run(args):
    rows_a = load_jsonl(args.outputs_a)
    rows_b = load_jsonl(args.outputs_b)

    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"Length mismatch: outputs-a has {len(rows_a)} rows, outputs-b has {len(rows_b)}"
        )

    rows_ref = None
    if args.mode == "reference":
        if not args.reference:
            raise ValueError("--reference is required when --mode=reference")
        rows_ref = load_jsonl(args.reference)
        if len(rows_ref) != len(rows_a):
            raise ValueError(
                f"Length mismatch: reference has {len(rows_ref)} rows while outputs have {len(rows_a)} rows"
            )

    n = len(rows_a)
    if args.max_samples and n > args.max_samples:
        rows_a = rows_a[: args.max_samples]
        rows_b = rows_b[: args.max_samples]
        if rows_ref is not None:
            rows_ref = rows_ref[: args.max_samples]
        n = args.max_samples

    validate_pair_alignment(
        rows_a,
        rows_b,
        rows_ref,
        args.reference,
        args.reference_prompt_key,
    )
    logger.info("Evaluating %d pairs with mode=%s model=%s", n, args.mode, args.model)

    results_by_index: dict[int, dict] = {}
    model_jobs = []
    for i in range(n):
        prompt = get_required_str(rows_a[i], "prompt", args.outputs_a, i)
        resp_a = get_required_str(rows_a[i], "response", args.outputs_a, i)
        resp_b = get_required_str(rows_b[i], "response", args.outputs_b, i)
        reference = None
        if rows_ref is not None:
            reference = get_required_str(rows_ref[i], args.reference_key, args.reference, i)
            fast_result = maybe_fast_path_winner(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
                reference=reference,
                enabled=not args.disable_fast_path,
            )
            if fast_result is not None:
                fast_result["index"] = i
                results_by_index[i] = build_output_sample(
                    index=i,
                    prompt=prompt,
                    response_a=resp_a,
                    response_b=resp_b,
                    reference=reference,
                    swapped=fast_result["swapped"],
                    verdict=fast_result["verdict"],
                    raw_verdict=fast_result["raw_verdict"],
                    parse_error=fast_result["parse_error"],
                    winner=fast_result["winner"],
                    judge_source=fast_result["judge_source"],
                    fast_path_signature=fast_result["fast_path_signature"],
                )
                continue

        model_jobs.append(build_model_job(i, prompt, resp_a, resp_b, reference, args.mode))

    if model_jobs:
        client, fallback_client = init_clients(args)
        semaphore = asyncio.Semaphore(args.concurrency)
        fallback_model = args.fallback_model or os.environ.get("WINRATE_FALLBACK_MODEL") or None

        async def eval_one(job: dict) -> dict:
            verdict, raw_verdict, parse_error = await judge_pair(
                client,
                args.model,
                job["content"],
                semaphore,
                fallback_client=fallback_client,
                fallback_model=fallback_model,
            )

            if verdict == "Tie":
                winner = "tie"
            elif job["swapped"]:
                winner = "b" if verdict == "A" else "a"
            else:
                winner = "a" if verdict == "A" else "b"

            return build_output_sample(
                index=job["index"],
                prompt=job["prompt"],
                response_a=job["response_a"],
                response_b=job["response_b"],
                reference=job["reference"],
                swapped=job["swapped"],
                verdict=verdict,
                raw_verdict=raw_verdict,
                parse_error=parse_error,
                winner=winner,
                judge_source="model",
            )

        model_results = await async_tqdm.gather(*(eval_one(job) for job in model_jobs), desc="judging")
        for sample in model_results:
            results_by_index[sample["index"]] = sample

    results = [results_by_index[i] for i in range(n)]

    a_wins = sum(1 for r in results if r["winner"] == "a")
    b_wins = sum(1 for r in results if r["winner"] == "b")
    ties = sum(1 for r in results if r["winner"] == "tie")
    parse_errors = sum(1 for r in results if r["parse_error"])
    fast_path_count = sum(1 for r in results if r["judge_source"] == "fast_path")
    model_judge_count = sum(1 for r in results if r["judge_source"] == "model")
    total = len(results)
    winrate_a = (a_wins + 0.5 * ties) / total if total > 0 else 0.0

    summary = {
        "total": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "parse_errors": parse_errors,
        "winrate_a": winrate_a,
        "mode": args.mode,
        "model": args.model,
        "outputs_a": args.outputs_a,
        "outputs_b": args.outputs_b,
        "reference_path": args.reference if args.mode == "reference" else None,
        "reference_key": args.reference_key if args.mode == "reference" else None,
        "reference_prompt_key": args.reference_prompt_key if args.mode == "reference" else None,
        "fast_path_count": fast_path_count,
        "model_judge_count": model_judge_count,
        "judge_source_counts": {
            "fast_path": fast_path_count,
            "model": model_judge_count,
        },
        "samples": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done: mode=%s total=%d a_wins=%d b_wins=%d ties=%d winrate_a=%.4f fast_path=%d model=%d",
        args.mode,
        total,
        a_wins,
        b_wins,
        ties,
        winrate_a,
        fast_path_count,
        model_judge_count,
    )
    logger.info("Results saved to %s", args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-a", required=True, help="JSONL from model A")
    parser.add_argument("--outputs-b", required=True, help="JSONL from model B")
    parser.add_argument("--output", required=True, help="Output winrate JSON path")
    parser.add_argument("--mode", choices=["blind", "reference"], default="blind")
    parser.add_argument("--reference", default=None, help="Reference JSONL for reference-mode judging")
    parser.add_argument("--reference-key", default="chosen", help="Reference answer key in --reference")
    parser.add_argument(
        "--reference-prompt-key",
        default="text",
        help="Prompt key in --reference used for strict row alignment",
    )
    parser.add_argument(
        "--disable-fast-path",
        action="store_true",
        help="Disable constrained-answer fast-path in reference mode",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI/OpenRouter API key")
    parser.add_argument("--base-url", default=None, help="API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--fallback-api-key", default=None, help="Fallback judge API key")
    parser.add_argument("--fallback-base-url", default=None, help="Fallback judge API base URL")
    parser.add_argument("--fallback-model", default=None, help="Fallback judge model for filtered prompts")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API calls")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
