import json
import logging
import random
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are an impartial judge evaluating AI assistant responses to a conversation."

JUDGE_PROMPT = """You will compare two AI assistant responses to the same conversation.

Conversation:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Reply with exactly one word: A, B, or Tie.
Consider helpfulness, accuracy, and quality. Do not favor a response just because it is longer."""


def _openrouter_generate(prompt: str, model: str, api_key: str, max_tokens: int, retries: int = 3) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning("openrouter_generate failed after %d retries: %s", retries, e)
                return ""
            time.sleep(2 ** attempt)
    return ""


def _sglang_generate(prompt: str, url: str, max_tokens: int, retries: int = 3) -> str:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{url.rstrip('/')}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning("sglang_generate failed after %d retries: %s", retries, e)
                return ""
            time.sleep(2 ** attempt)
    return ""


def _judge(prompt: str, response_a: str, response_b: str, model: str, api_key: str) -> str:
    """Returns 'A', 'B', or 'Tie'."""
    judge_prompt = JUDGE_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_prompt},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            verdict = resp.json()["choices"][0]["message"]["content"].strip()
            # 解析 verdict：只取第一个词，大写比较
            first_word = verdict.split()[0].upper() if verdict.split() else ""
            if first_word in ("A", "B"):
                return first_word
            if "TIE" in first_word or first_word == "SAME":
                return "Tie"
            # 如果没匹配到，返回 Tie 作为保守回退
            logger.debug("judge returned unexpected verdict: %r, treating as Tie", verdict)
            return "Tie"
        except Exception as e:
            if attempt == 2:
                logger.warning("judge failed: %s", e)
                return "Tie"
            time.sleep(2 ** attempt)
    return "Tie"


def win_rate_eval(args, rollout_id: int) -> None:
    eval_path = getattr(args, "win_rate_eval_path", None)
    if not eval_path:
        return

    api_key = getattr(args, "win_rate_eval_openrouter_key", None)
    if not api_key:
        logger.info("win_rate_eval: win_rate_eval_openrouter_key not set, skipping")
        return

    prompt_key = getattr(args, "win_rate_eval_prompt_key", "text")
    max_samples = getattr(args, "win_rate_eval_max_samples", 50)
    gpt4o_model = getattr(args, "win_rate_eval_gpt4o_model", "openai/gpt-4o")
    max_tokens = getattr(args, "win_rate_eval_max_tokens", 512)

    # 构建 SGLang URL 从现有参数
    sglang_ip = getattr(args, "sglang_router_ip", None) or "127.0.0.1"
    sglang_port = getattr(args, "sglang_router_port", None) or 30000
    sglang_url = f"http://{sglang_ip}:{sglang_port}"

    # 加载测试集
    prompts = []
    try:
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if prompt_key in obj:
                    prompts.append(obj[prompt_key])
                if max_samples and len(prompts) >= max_samples:
                    break
    except Exception as e:
        logger.warning("win_rate_eval: failed to load %s: %s", eval_path, e)
        return

    if not prompts:
        logger.info("win_rate_eval: no prompts found in %s, skipping", eval_path)
        return

    logger.info(
        "win_rate_eval rollout=%d: evaluating %d prompts, gpt4o=%s, sglang=%s",
        rollout_id, len(prompts), gpt4o_model, sglang_url,
    )

    win = tie = lose = skip = 0
    results = []

    for i, prompt in enumerate(prompts):
        # 并行生成两侧回复
        gpt4o_response = _openrouter_generate(prompt, gpt4o_model, api_key, max_tokens)
        model_response = _sglang_generate(prompt, sglang_url, max_tokens)

        if not gpt4o_response or not model_response:
            logger.debug("win_rate_eval sample %d: skipped (empty response)", i)
            skip += 1
            results.append({
                "index": i,
                "prompt": prompt[:200],
                "gpt4o_response": gpt4o_response[:200],
                "model_response": model_response[:200],
                "verdict": "skip",
                "model_win": None,
            })
            continue

        # 随机交换 A/B 顺序避免位置偏差
        swap = random.random() < 0.5
        if swap:
            response_a, response_b = model_response, gpt4o_response
        else:
            response_a, response_b = gpt4o_response, model_response

        raw_verdict = _judge(prompt, response_a, response_b, gpt4o_model, api_key)

        # 将 judge 的 A/B 结果映射回「model vs gpt4o」的胜负
        if raw_verdict == "Tie":
            model_win = None
            tie += 1
        elif swap:
            # A=model, B=gpt4o
            model_win = raw_verdict == "A"
            if model_win:
                win += 1
            else:
                lose += 1
        else:
            # A=gpt4o, B=model
            model_win = raw_verdict == "B"
            if model_win:
                win += 1
            else:
                lose += 1

        results.append({
            "index": i,
            "prompt": prompt[:200],
            "gpt4o_response": gpt4o_response[:200],
            "model_response": model_response[:200],
            "verdict": raw_verdict,
            "swapped": swap,
            "model_win": model_win,
        })

    judged = win + tie + lose
    win_rate = win / judged if judged > 0 else 0.0

    logger.info(
        "win_rate_eval rollout=%d samples=%d win=%d tie=%d lose=%d skip=%d win_rate=%.4f gpt4o_model=%s sglang_url=%s",
        rollout_id, len(prompts), win, tie, lose, skip, win_rate, gpt4o_model, sglang_url,
    )

    # 写入详细结果 JSON
    reward_dir = Path(getattr(args, "reward_model_dir", "reward_model"))
    reward_dir.mkdir(parents=True, exist_ok=True)
    out_path = reward_dir / f"win_rate_rollout_{rollout_id}.json"
    try:
        out_path.write_text(
            json.dumps(
                {
                    "rollout_id": rollout_id,
                    "win": win,
                    "tie": tie,
                    "lose": lose,
                    "skip": skip,
                    "win_rate": win_rate,
                    "gpt4o_model": gpt4o_model,
                    "sglang_url": sglang_url,
                    "samples": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("win_rate_eval: results saved to %s", out_path)
    except Exception as e:
        logger.warning("win_rate_eval: failed to save results: %s", e)
