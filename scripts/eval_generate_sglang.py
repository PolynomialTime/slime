"""
Fast eval generation using SGLang (much faster than HF transformers).
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import aiohttp

# Support direct execution via `python3 scripts/*.py` by making the repo root
# importable before loading sibling package modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.text_hygiene import count_non_printing_chars, strip_non_printing_chars, summarize_non_printing_chars


def parse_hh_rlhf_text(text: str) -> list[dict]:
    """Parse hh-rlhf text field into conversation turns."""
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=600)
DEFAULT_QWEN_STOP_TOKEN_IDS = [151643, 151644, 151645]


def response_is_empty(text: str) -> bool:
    return (text or "").strip() == ""


def response_has_user_prefix(text: str) -> bool:
    prefix = (text or "").lstrip().lower()
    return prefix.startswith("user\n") or prefix.startswith("user:") or prefix.startswith("<|im_start|>user")


def response_has_assistant_prefix(text: str) -> bool:
    prefix = (text or "").lstrip().lower()
    return prefix.startswith("assistant") or prefix.startswith("<|im_start|>assistant")


def response_is_eos_only(text: str, response_tokens: list[int], stop_token_ids: set[int]) -> bool:
    return bool(response_tokens) and response_is_empty(text) and all(token in stop_token_ids for token in response_tokens)


def resolve_stop_token_ids(tokenizer, override: str | None = None) -> list[int]:
    if override:
        return [int(x) for x in override.replace(",", " ").split() if x.strip()]

    token_ids = []
    for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids or DEFAULT_QWEN_STOP_TOKEN_IDS


def load_prompts(path: str, prompt_key: str, apply_chat_template: bool, tokenizer=None, chat_template_kwargs=None) -> list[str]:
    prompts = []
    raw = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            raw.append(obj[prompt_key])

    if apply_chat_template and tokenizer:
        ct_kwargs = chat_template_kwargs or {}
        for p in raw:
            if isinstance(p, str):
                if p.lstrip().startswith("Human: "):
                    msgs = parse_hh_rlhf_text(p)
                else:
                    msgs = [{"role": "user", "content": p}]
            else:
                msgs = p
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **ct_kwargs))
    else:
        prompts = raw
    return raw, prompts


async def wait_for_sglang_ready(session, url: str, timeout_s: int = 180):
    """Wait until SGLang server is ready, raise if not ready within timeout."""
    deadline = time.time() + timeout_s
    last_error = "no response yet"
    while time.time() < deadline:
        for endpoint in ("/health_generate", "/health"):
            try:
                async with session.get(f"{url}{endpoint}") as resp:
                    if resp.status == 404 and endpoint == "/health_generate":
                        continue
                    if resp.status < 400:
                        logger.info("SGLang ready via %s", endpoint)
                        return
                    body = await resp.text()
                    last_error = f"{endpoint} returned HTTP {resp.status}: {body[:200]!r}"
            except Exception as e:
                last_error = f"{endpoint} failed: {type(e).__name__}: {e}"
        await asyncio.sleep(2)
    raise RuntimeError(f"SGLang at {url} was not ready after {timeout_s}s ({last_error})")


async def generate_one(session, url, prompt, max_tokens, temperature, semaphore, stop_token_ids):
    async with semaphore:
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "stop_token_ids": stop_token_ids,
                "skip_special_tokens": True,
            },
            "return_logprob": True,
        }
        last_error = None
        for attempt in range(1, 6):
            try:
                async with session.post(f"{url}/generate", json=payload) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
                    data = json.loads(body)
                    text = data.get("text", "")
                    if not isinstance(text, str):
                        raise RuntimeError(f"Unexpected response structure; keys={sorted(data.keys())}")
                    meta_info = data.get("meta_info") or {}
                    response_tokens = []
                    if "output_token_logprobs" in meta_info:
                        response_tokens = [item[1] for item in meta_info["output_token_logprobs"]]
                    # Defensive: strip prompt prefix if server returns full_text semantics
                    if text.startswith(prompt):
                        text = text[len(prompt):]
                    if text == "":
                        logger.warning("Empty completion from SGLang (meta_info=%s)", meta_info)
                    return {
                        "text": text,
                        "response_tokens": response_tokens,
                        "finish_reason": meta_info.get("finish_reason"),
                    }
            except Exception as e:
                last_error = e
                logger.warning("SGLang attempt %d/5 failed: %s", attempt, e)
                if attempt < 5:
                    await asyncio.sleep(min(attempt, 5))
        raise RuntimeError(f"SGLang /generate failed after 5 attempts: {last_error}")


async def run(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    stop_token_ids = resolve_stop_token_ids(tokenizer, args.stop_token_ids)
    stop_token_id_set = set(stop_token_ids)

    chat_template_kwargs = None
    if args.apply_chat_template_kwargs:
        chat_template_kwargs = json.loads(args.apply_chat_template_kwargs)

    raw_prompts, formatted_prompts = load_prompts(
        args.prompt_data, args.prompt_key, args.apply_chat_template, tokenizer, chat_template_kwargs
    )

    logger.info(
        "Generating %d responses via SGLang at %s (temperature=%.3f stop_token_ids=%s)",
        len(formatted_prompts),
        args.sglang_url,
        args.temperature,
        stop_token_ids,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        await wait_for_sglang_ready(session, args.sglang_url)
        tasks = [generate_one(session, args.sglang_url, p, args.max_new_tokens, args.temperature, semaphore, stop_token_ids)
                 for p in formatted_prompts]
        from tqdm.asyncio import tqdm as async_tqdm
        response_records = await async_tqdm.gather(*tasks, desc="generating")

    elapsed = time.time() - t0
    responses = [record["text"] for record in response_records]
    non_printing_rows_raw = 0
    non_printing_chars_raw = 0
    sanitized_records = []
    sanitize_budget = 3
    for raw_prompt, record in zip(raw_prompts, response_records, strict=True):
        raw_text = record["text"]
        non_printing_count = count_non_printing_chars(raw_text)
        if non_printing_count:
            non_printing_rows_raw += 1
            non_printing_chars_raw += non_printing_count
            if sanitize_budget > 0:
                logger.warning(
                    "Sanitizing non-printing characters from eval output prompt_tail=%r summary=%s",
                    str(raw_prompt)[-160:],
                    summarize_non_printing_chars(raw_text),
                )
                sanitize_budget -= 1
        record = dict(record)
        record["text_raw"] = raw_text
        record["text"] = strip_non_printing_chars(raw_text)
        sanitized_records.append(record)
    response_records = sanitized_records
    responses = [record["text"] for record in response_records]
    empty_count = sum(response_is_empty(r) for r in responses)
    eos_only_count = sum(response_is_eos_only(record["text"], record["response_tokens"], stop_token_id_set) for record in response_records)
    user_prefix_count = sum(response_has_user_prefix(r) for r in responses)
    assistant_prefix_count = sum(response_has_assistant_prefix(r) for r in responses)
    if non_printing_rows_raw:
        logger.warning(
            "Sanitized non-printing characters from %d/%d eval rows (%d chars total)",
            non_printing_rows_raw,
            len(responses),
            non_printing_chars_raw,
        )
    if empty_count:
        logger.warning("Received %d empty completions out of %d", empty_count, len(responses))
    if eos_only_count or user_prefix_count or assistant_prefix_count:
        logger.warning(
            "Eval pathologies: eos_only=%d user_prefix=%d assistant_prefix=%d out of %d",
            eos_only_count,
            user_prefix_count,
            assistant_prefix_count,
            len(responses),
        )
        anomaly_budget = 3
        for raw_prompt, record in zip(raw_prompts, response_records, strict=True):
            text = record["text"]
            if anomaly_budget <= 0:
                break
            if not (
                response_is_eos_only(text, record["response_tokens"], stop_token_id_set)
                or response_has_user_prefix(text)
                or response_has_assistant_prefix(text)
                or response_is_empty(text)
            ):
                continue
            logger.warning(
                "Eval pathology prompt_tail=%r response_prefix=%r response_tokens=%s finish_reason=%s",
                str(raw_prompt)[-160:],
                text[:160],
                record["response_tokens"][:16],
                record["finish_reason"],
            )
            anomaly_budget -= 1
    logger.info("Generated %d responses in %.1fs (%.1f/s)", len(responses), elapsed, len(responses)/elapsed)

    results = [{"prompt": raw_prompts[i], "response": responses[i]} for i in range(len(responses))]

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d outputs to %s", len(results), args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--prompt-key", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--apply-chat-template-kwargs", type=str, default=None,
                        help='JSON string, e.g. \'{"enable_thinking":false}\'')
    parser.add_argument("--stop-token-ids", type=str, default=None,
                        help="Optional whitespace/comma-separated token ids. Defaults to tokenizer special Qwen ids.")
    parser.add_argument("--concurrency", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
