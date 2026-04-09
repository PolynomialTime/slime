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
_LAST_SERVER_ACTIVITY = 0.0
_MAX_RESPONSE_LEN = int(os.environ.get("SLIME_CUSTOM_RM_MAX_RESPONSE_LEN", "384"))
_SHORT_RESPONSE_THRESHOLD = 50
_TRUNCATION_THRESHOLD = int(0.9 * _MAX_RESPONSE_LEN)
_SHORT_PENALTY = 1.0
# Keep length shaping symmetric so truncation does not structurally favor short hedge replies.
_TRUNCATION_PENALTY = 1.0
_HUMAN_CONTINUATION_PENALTY = 5.0
_ASSISTANT_PREFIX_PENALTY = 2.0
_REPETITION_PENALTY_MAX = 3.0


def _parse_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("[custom_rm] invalid %s=%r, using default=%s", name, value, default)
        return default


_IDLE_TIMEOUT_SEC = max(0.0, _parse_env_float("SLIME_CUSTOM_RM_IDLE_TIMEOUT_SEC", 0.0))


def _parse_visible_device_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _query_gpu_free_memory_mb() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as e:
        logger.warning("[custom_rm] failed to query GPU free memory with nvidia-smi: %s", e)
        return {}

    free_memory_by_gpu = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        gpu_id, free_mb = parts[0], parts[1]
        try:
            free_memory_by_gpu[gpu_id] = int(free_mb)
        except ValueError:
            continue
    return free_memory_by_gpu


def _select_reward_gpu(candidate_gpu_ids: list[str]) -> tuple[str | None, str]:
    if not candidate_gpu_ids:
        return None, "no candidate GPUs"

    free_memory_by_gpu = _query_gpu_free_memory_mb()
    if free_memory_by_gpu:
        ranked = []
        for gpu_id in candidate_gpu_ids:
            free_mb = free_memory_by_gpu.get(gpu_id, -1)
            try:
                numeric_gpu_id = int(gpu_id)
            except ValueError:
                numeric_gpu_id = -1
            ranked.append((free_mb, numeric_gpu_id, gpu_id))
        ranked.sort(reverse=True)
        best_free_mb, _, best_gpu_id = ranked[0]
        return best_gpu_id, f"auto-selected from {candidate_gpu_ids} by free memory, best_free_mb={best_free_mb}"

    # Fallback when nvidia-smi is unavailable: prefer the highest-index GPU rather than pinning GPU 0.
    return candidate_gpu_ids[-1], f"fallback-selected last visible GPU from {candidate_gpu_ids}"


def _resolve_reward_device() -> tuple[str, str, str]:
    requested_device = (os.environ.get("SLIME_CUSTOM_RM_DEVICE") or "").strip().lower()
    requested_visible_devices = (os.environ.get("SLIME_CUSTOM_RM_CUDA_VISIBLE_DEVICES") or "").strip()

    if requested_device in {"cpu", "none"}:
        return "", "cpu", "forced cpu"

    auto_tokens = {"", "auto", "best", "max_free"}
    if requested_visible_devices.lower() in auto_tokens:
        visible_devices = (
            os.environ.get("_REAL_CUDA_VISIBLE_DEVICES")
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            or "0,1,2,3"
        )
        candidate_gpu_ids = _parse_visible_device_list(visible_devices)
    else:
        candidate_gpu_ids = _parse_visible_device_list(requested_visible_devices)

    gpu_id, reason = _select_reward_gpu(candidate_gpu_ids)
    if gpu_id is None:
        return "", "cpu", reason
    return gpu_id, "cuda", reason


class _Request:
    __slots__ = ("tokens", "result", "done")
    def __init__(self, tokens):
        self.tokens = tokens
        self.result = 0.0
        self.done = threading.Event()


def _read_process_stderr(proc):
    if proc is None or proc.stderr is None:
        return ""
    try:
        return proc.stderr.read().strip()
    except Exception as e:
        return f"<failed to read stderr: {e}>"


def _drain_pending_requests():
    drained = 0
    while True:
        try:
            req = _QUEUE.get_nowait()
        except queue.Empty:
            return drained
        req.result = 0.0
        req.done.set()
        drained += 1


def _start_server(base_model, model_path):
    """Start persistent scoring server subprocess on dedicated GPU."""
    global _SERVER_PROC, _MODEL_MTIME, _LAST_SERVER_ACTIVITY
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    resolved_visible_devices, device, selection_reason = _resolve_reward_device()

    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        device_desc = "cpu"
    else:
        env["CUDA_VISIBLE_DEVICES"] = resolved_visible_devices
        device_desc = resolved_visible_devices

    _SERVER_PROC = subprocess.Popen(
        [sys.executable, "-m", "slime.local_rm.score_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )
    # Send config
    config = json.dumps({"base_model": base_model, "model_path": model_path, "device": device})
    _SERVER_PROC.stdin.write(config + "\n")
    _SERVER_PROC.stdin.flush()

    # Wait for ready
    proc = _SERVER_PROC
    ready = proc.stdout.readline()
    if not ready:
        returncode = proc.poll()
        stderr = _read_process_stderr(proc)
        logger.error(
            "[custom_rm] score server exited before ready on %s (returncode=%s). stderr:\n%s",
            device_desc, returncode, stderr or "<empty>",
        )
        _stop_server(reason="startup failed before ready")
        raise RuntimeError(
            f"score server exited before ready on {device_desc} (returncode={returncode}). "
            f"Likely CUDA OOM or import/init failure. stderr: {stderr or '<empty>'}"
        )
    try:
        resp = json.loads(ready)
    except json.JSONDecodeError as e:
        logger.error("[custom_rm] invalid score server ready payload on %s: %r", device_desc, ready.rstrip())
        _stop_server(reason="invalid startup response")
        raise RuntimeError(
            f"invalid score server ready payload on {device_desc}: {ready.rstrip()!r}"
        ) from e
    logger.info(
        "[custom_rm] score server started on %s, device=%s (%s)",
        device_desc,
        resp.get("device"),
        selection_reason,
    )
    _MODEL_MTIME = os.path.getmtime(model_path)
    _LAST_SERVER_ACTIVITY = time.monotonic()


def _stop_server(reason: str | None = None):
    global _SERVER_PROC, _MODEL_MTIME, _LAST_SERVER_ACTIVITY
    proc = _SERVER_PROC
    _SERVER_PROC = None
    _MODEL_MTIME = 0.0
    _LAST_SERVER_ACTIVITY = 0.0
    if proc is None:
        return

    if reason is not None:
        logger.info("[custom_rm] stopping score server: %s", reason)

    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass

    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception as e:
        logger.warning("[custom_rm] failed to stop score server cleanly: %s", e)
    finally:
        for stream_name in ("stdout", "stderr"):
            stream = getattr(proc, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass


def release_resources(args=None, reason: str | None = None):
    with _LOCK:
        _stop_server(reason=reason or "external release")


def _ensure_server(base_model, model_path):
    """Ensure server is running and model is up to date."""
    global _SERVER_PROC, _MODEL_MTIME, _LAST_SERVER_ACTIVITY
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
        _LAST_SERVER_ACTIVITY = time.monotonic()


def _score_batch_via_server(tokens_list):
    """Send batch to server, get rewards back."""
    global _LAST_SERVER_ACTIVITY
    request = json.dumps({"tokens": tokens_list})
    _SERVER_PROC.stdin.write(request + "\n")
    _SERVER_PROC.stdin.flush()
    response = _SERVER_PROC.stdout.readline()
    _LAST_SERVER_ACTIVITY = time.monotonic()
    return json.loads(response)


def _worker_loop(base_model, model_path):
    """Collect samples, score via persistent GPU server."""
    try:
        _ensure_server(base_model, model_path)
    except Exception:
        drained = _drain_pending_requests()
        logger.exception(
            "[custom_rm] failed to initialize score server; drained %d queued requests and exiting worker",
            drained,
        )
        return
    queue_timeout = 1.0 if _IDLE_TIMEOUT_SEC > 0 else 120.0

    while True:
        try:
            first = _QUEUE.get(timeout=queue_timeout)
        except queue.Empty:
            if (
                _IDLE_TIMEOUT_SEC > 0
                and _SERVER_PROC is not None
                and _LAST_SERVER_ACTIVITY > 0
                and time.monotonic() - _LAST_SERVER_ACTIVITY >= _IDLE_TIMEOUT_SEC
            ):
                _stop_server(reason=f"idle for {_IDLE_TIMEOUT_SEC:.1f}s")
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

    if isinstance(samples, list):
        reqs = [_Request(s.tokens) for s in samples]
        for req in reqs:
            _QUEUE.put(req)
        _ensure_worker(args)

        def _wait_all():
            for r in reqs:
                r.done.wait(timeout=120)
            return [_apply_reward_shaping(r.result, s) for r, s in zip(reqs, samples)]
        return await asyncio.to_thread(_wait_all)

    req = _Request(samples.tokens)
    _QUEUE.put(req)
    _ensure_worker(args)

    def _wait():
        req.done.wait(timeout=120)
        return _apply_reward_shaping(req.result, samples)
    return await asyncio.to_thread(_wait)
