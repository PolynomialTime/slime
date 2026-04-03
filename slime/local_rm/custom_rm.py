import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import queue

logger = logging.getLogger(__name__)

_QUEUE = queue.Queue()
_WORKER = None
_LOCK = threading.Lock()
_SERVER_PROC = None
_MODEL_MTIME = 0.0
_MAX_RESPONSE_LEN = 512
_SHORT_RESPONSE_THRESHOLD = 50
_TRUNCATION_THRESHOLD = int(0.9 * _MAX_RESPONSE_LEN)
_SHORT_PENALTY = 1.0
_TRUNCATION_PENALTY = 10.0
_HUMAN_CONTINUATION_PENALTY = 5.0
_ASSISTANT_PREFIX_PENALTY = 2.0
_REPETITION_PENALTY_MAX = 3.0


class _Request:
    __slots__ = ("tokens", "result", "done")
    def __init__(self, tokens):
        self.tokens = tokens
        self.result = 0.0
        self.done = threading.Event()


def _start_server(base_model, model_path):
    """Start persistent scoring server subprocess on dedicated GPU."""
    global _SERVER_PROC, _MODEL_MTIME
    gpu_id = os.environ.get("_REAL_CUDA_VISIBLE_DEVICES", "0,1,2,3").strip().split(",")[-1].strip()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    _SERVER_PROC = subprocess.Popen(
        [sys.executable, "-m", "slime.local_rm.score_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )
    # Send config
    config = json.dumps({"base_model": base_model, "model_path": model_path})
    _SERVER_PROC.stdin.write(config + "\n")
    _SERVER_PROC.stdin.flush()

    # Wait for ready
    ready = _SERVER_PROC.stdout.readline()
    resp = json.loads(ready)
    logger.info("[custom_rm] score server started on GPU %s, device=%s", gpu_id, resp.get("device"))
    _MODEL_MTIME = os.path.getmtime(model_path)


def _ensure_server(base_model, model_path):
    """Ensure server is running and model is up to date."""
    global _SERVER_PROC, _MODEL_MTIME
    mtime = os.path.getmtime(model_path)
    if _SERVER_PROC is None or _SERVER_PROC.poll() is not None:
        _start_server(base_model, model_path)
    elif mtime > _MODEL_MTIME:
        # Reload model
        _SERVER_PROC.stdin.write(json.dumps({"cmd": "reload", "model_path": model_path}) + "\n")
        _SERVER_PROC.stdin.flush()
        resp = json.loads(_SERVER_PROC.stdout.readline())
        logger.info("[custom_rm] model reloaded: %s", resp)
        _MODEL_MTIME = mtime


def _score_batch_via_server(tokens_list):
    """Send batch to server, get rewards back."""
    request = json.dumps({"tokens": tokens_list})
    _SERVER_PROC.stdin.write(request + "\n")
    _SERVER_PROC.stdin.flush()
    response = _SERVER_PROC.stdout.readline()
    return json.loads(response)


def _worker_loop(base_model, model_path):
    """Collect samples, score via persistent GPU server."""
    _ensure_server(base_model, model_path)

    while True:
        try:
            first = _QUEUE.get(timeout=120)
        except queue.Empty:
            continue

        # Collect batch (100ms window)
        batch = [first]
        deadline = time.monotonic() + 0.1
        while len(batch) < 512:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if len(batch) >= 64:
                    break
                deadline = time.monotonic() + 0.05
            try:
                batch.append(_QUEUE.get(timeout=max(0.001, remaining)))
            except queue.Empty:
                break

        tokens_list = [req.tokens for req in batch]
        t0 = time.time()
        try:
            _ensure_server(base_model, model_path)
            rewards = _score_batch_via_server(tokens_list)
            elapsed = time.time() - t0
            if isinstance(rewards, dict) and "error" in rewards:
                logger.error("[custom_rm] server error: %s", rewards["error"])
                rewards = [0.0] * len(batch)
            else:
                logger.info("[custom_rm] batch %d in %.2fs (%.1fms/sample)",
                           len(batch), elapsed, elapsed * 1000 / max(len(batch), 1))
        except Exception as e:
            logger.error("[custom_rm] scoring failed: %s", e)
            rewards = [0.0] * len(batch)

        for i, req in enumerate(batch):
            req.result = rewards[i] if i < len(rewards) else 0.0
            req.done.set()


_ARGS_CACHE = [None, None]


def _apply_reward_shaping(reward, sample):
    response_len = getattr(sample, "response_length", 0)
    response_text = getattr(sample, "response", "")

    # Length penalties
    if response_len < _SHORT_RESPONSE_THRESHOLD:
        reward -= _SHORT_PENALTY
    if response_len >= _TRUNCATION_THRESHOLD:
        reward -= _TRUNCATION_PENALTY

    # Role confusion penalties
    if response_text.lstrip().startswith("Human:") or "\nHuman:" in response_text:
        reward -= _HUMAN_CONTINUATION_PENALTY
    if response_text.lstrip().startswith("Assistant:"):
        reward -= _ASSISTANT_PREFIX_PENALTY

    # Repetition penalty (4-gram)
    words = response_text.split()
    if len(words) >= 8:
        ngrams = {}
        for i in range(len(words) - 3):
            ng = tuple(words[i:i+4])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams:
            max_count = max(ngrams.values())
            if max_count > 3:
                penalty = 2.0 * (max_count - 3) / max(len(ngrams), 1)
                reward -= min(penalty, _REPETITION_PENALTY_MAX)
    return max(-5.0, min(reward, 5.0))


def _ensure_worker(args):
    global _WORKER
    reward_dir = getattr(args, "reward_model_dir", None) or "reward_model"
    model_path = os.path.join(reward_dir, "latest")
    base_model = getattr(args, "reward_model_init", None) or args.hf_checkpoint
    _ARGS_CACHE[0] = base_model
    _ARGS_CACHE[1] = model_path

    with _LOCK:
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(
                target=_worker_loop, args=(base_model, model_path),
                daemon=True, name="custom_rm_worker"
            )
            _WORKER.start()


async def custom_rm(args, samples):
    """Async custom RM with persistent GPU subprocess scoring."""
    reward_dir = getattr(args, "reward_model_dir", None) or "reward_model"
    model_path = os.path.join(reward_dir, "latest")
    if not os.path.exists(model_path):
        return [0.0] * len(samples) if isinstance(samples, list) else 0.0

    _ensure_worker(args)

    if isinstance(samples, list):
        reqs = [_Request(s.tokens) for s in samples]
        for req in reqs:
            _QUEUE.put(req)

        def _wait_all():
            for r in reqs:
                r.done.wait(timeout=120)
            return [_apply_reward_shaping(r.result, s) for r, s in zip(reqs, samples)]
        return await asyncio.to_thread(_wait_all)

    req = _Request(samples.tokens)
    _QUEUE.put(req)

    def _wait():
        req.done.wait(timeout=120)
        return _apply_reward_shaping(req.result, samples)
    return await asyncio.to_thread(_wait)
