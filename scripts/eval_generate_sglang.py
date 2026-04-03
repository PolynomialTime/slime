"""
Fast eval generation using SGLang (much faster than HF transformers).
"""
import argparse
import asyncio
import json
import logging
import aiohttp
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=600)


def load_prompts(path: str, prompt_key: str, apply_chat_template: bool, tokenizer=None) -> list[str]:
    prompts = []
    raw = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            raw.append(obj[prompt_key])

    if apply_chat_template and tokenizer:
        for p in raw:
            if isinstance(p, str):
                msgs = [{"role": "user", "content": p}]
            else:
                msgs = p
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
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


async def generate_one(session, url, prompt, max_tokens, semaphore):
    async with semaphore:
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": 0,
                "stop": ["\nHuman:", "\n\nHuman:", "<|im_end|>"],
            }
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
                    # Defensive: strip prompt prefix if server returns full_text semantics
                    if text.startswith(prompt):
                        text = text[len(prompt):]
                    if text == "":
                        logger.warning("Empty completion from SGLang (meta_info=%s)", data.get("meta_info"))
                    return text
            except Exception as e:
                last_error = e
                logger.warning("SGLang attempt %d/5 failed: %s", attempt, e)
                if attempt < 5:
                    await asyncio.sleep(min(attempt, 5))
        raise RuntimeError(f"SGLang /generate failed after 5 attempts: {last_error}")


async def run(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    raw_prompts, formatted_prompts = load_prompts(
        args.prompt_data, args.prompt_key, args.apply_chat_template, tokenizer
    )

    logger.info("Generating %d responses via SGLang at %s", len(formatted_prompts), args.sglang_url)

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        await wait_for_sglang_ready(session, args.sglang_url)
        tasks = [generate_one(session, args.sglang_url, p, args.max_new_tokens, semaphore)
                 for p in formatted_prompts]
        from tqdm.asyncio import tqdm as async_tqdm
        responses = await async_tqdm.gather(*tasks, desc="generating")

    elapsed = time.time() - t0
    empty_count = sum(r == "" for r in responses)
    if empty_count:
        logger.warning("Received %d empty completions out of %d", empty_count, len(responses))
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
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--concurrency", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
