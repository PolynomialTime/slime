"""
Generate GPT-4o responses for winrate comparison.
Run on dev machine (needs API access).
"""
import argparse
import asyncio
import json
import logging

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_prompts(path: str, prompt_key: str) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)[prompt_key])
    return prompts


async def generate_one(client, model, prompt, semaphore):
    messages = [{"role": "user", "content": prompt}]
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.7,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    logger.warning("Failed after 3 attempts: %s", e)
                    return ""
                await asyncio.sleep(2 ** attempt)
    return ""


async def run(args):
    prompts = load_prompts(args.prompt_data, args.prompt_key)
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    logger.info("Generating %d responses with %s", len(prompts), args.model)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def gen(i):
        response = await generate_one(client, args.model, prompts[i], semaphore)
        return {"prompt": prompts[i], "response": response}

    tasks = [gen(i) for i in range(len(prompts))]
    results = await async_tqdm.gather(*tasks, desc="generating")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d outputs to %s", len(results), args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--prompt-key", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
