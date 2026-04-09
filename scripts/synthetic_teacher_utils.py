import asyncio
import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

DEFAULT_MODEL = "gpt-4o"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("SYNTH_TEACHER_REQUEST_TIMEOUT", "180"))

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def resolve_api_key(explicit: str | None) -> str:
    return (
        explicit
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("WINRATE_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or ""
    )


def resolve_base_url(explicit: str | None) -> str | None:
    return (
        explicit
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("WINRATE_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or None
    )


def build_async_client(
    *,
    api_key: str,
    base_url: str | None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> AsyncOpenAI:
    client_kwargs = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": 0,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return AsyncOpenAI(**client_kwargs)


def count_jsonl_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_message_text(message_content) -> str:
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(message_content).strip()


async def generate_text(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    semaphore: asyncio.Semaphore,
    temperature: float,
    max_tokens: int,
    retries: int = 8,
) -> str:
    for attempt in range(retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
            text = extract_message_text(resp.choices[0].message.content)
            if not text:
                raise ValueError("teacher returned empty content")
            return text
        except Exception as exc:
            if attempt == retries - 1:
                raise
            delay = min(30, 2 ** attempt)
            logger.warning("teacher request failed on attempt %d, retrying in %ss: %s", attempt + 1, delay, exc)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")
