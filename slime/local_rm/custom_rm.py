import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import queue

from slime.utils.text_hygiene import count_non_printing_chars, non_printing_char_ratio

logger = logging.getLogger(__name__)

_QUEUE = queue.Queue()
_WORKER = None
_LOCK = threading.Lock()
_SERVER_PROC = None
_MODEL_MTIME = 0.0
_LAST_SERVER_ACTIVITY = 0.0
_MAX_RESPONSE_LEN = int(os.environ.get("SLIME_CUSTOM_RM_MAX_RESPONSE_LEN", "384"))
_SHORT_RESPONSE_THRESHOLD = 50
_SHORT_PENALTY = 1.0
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


_TRUNCATION_THRESHOLD_FRAC = _parse_env_float("SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC", 0.8)
_TRUNCATION_THRESHOLD_FRAC = min(1.0, max(0.0, _TRUNCATION_THRESHOLD_FRAC))
_TRUNCATION_THRESHOLD = min(_MAX_RESPONSE_LEN - 1, int(_TRUNCATION_THRESHOLD_FRAC * _MAX_RESPONSE_LEN))
# Penalize responses that run into the max-length cap so PPO cannot farm reward by
# drifting toward ever-longer answers. The penalty ramps up near the limit and
# reaches full strength on true truncation.
_TRUNCATION_PENALTY = _parse_env_float("SLIME_CUSTOM_RM_TRUNCATION_PENALTY", 2.5)
# Penalize only zero-token rollouts (immediate-EOS). Legitimate short answers
# such as "Yes" / "No" / "42" are still 1+ tokens and remain unaffected.
_EMPTY_RESPONSE_PENALTY = _parse_env_float("SLIME_CUSTOM_RM_EMPTY_RESPONSE_PENALTY", 5.0)
_DISCLAIMER_PENALTY = _parse_env_float("SLIME_CUSTOM_RM_DISCLAIMER_PENALTY", 3.0)
_DISCLAIMER_PATTERNS = (
    "i'm unable",
    "i cannot",
    "i can't",
    "i'm not able",
    "i am unable",
    "as an ai",
    "as a language model",
    "i don't have personal",
    "i am not able",
    "i'm sorry, but",
    "i don't have the ability",
    "i don't have access",
    "i do not have the ability",
    "i don't have real-time",
    "i don't have realtime",
)
_NON_PRINTING_COUNT_THRESHOLD = int(os.environ.get("SLIME_NON_PRINTING_COUNT_THRESHOLD", "8"))
_NON_PRINTING_RATIO_THRESHOLD = _parse_env_float("SLIME_NON_PRINTING_RATIO_THRESHOLD", 0.02)
_NON_PRINTING_PENALTY_MAX = _parse_env_float("SLIME_NON_PRINTING_PENALTY_MAX", 5.0)
_IDLE_TIMEOUT_SEC = max(0.0, _parse_env_float("SLIME_CUSTOM_RM_IDLE_TIMEOUT_SEC", 0.0))

# ---------- Rubric (multi-source) reward shaping ----------
# When SLIME_RUBRIC_ENABLED=1, the final reward becomes:
#   final = w_irl * r_irl
#         + w_format * format_score(response)
#         + w_answer * answer_score(response, label)
#         - existing penalties (truncation/empty/disclaimer)
# Designed for math IRL: the IRL learned RM still drives reasoning quality;
# format/answer act as rule-based regularizers to prevent reward hacking
# (e.g. \boxed{x = 4/9} polluting answer extraction).
_RUBRIC_ENABLED = (os.environ.get("SLIME_RUBRIC_ENABLED", "0").strip() in {"1", "true", "True"})
_RUBRIC_W_IRL = _parse_env_float("SLIME_RUBRIC_W_IRL", 1.0)
_RUBRIC_W_FORMAT = _parse_env_float("SLIME_RUBRIC_W_FORMAT", 0.5)
_RUBRIC_W_ANSWER = _parse_env_float("SLIME_RUBRIC_W_ANSWER", 1.0)
_RUBRIC_LOG_EVERY = int(os.environ.get("SLIME_RUBRIC_LOG_EVERY", "256"))
_RUBRIC_CALL_COUNTER = 0

_BOXED_RE = None  # lazy-compiled
_MATH_GRADER = None  # lazy-loaded module (avoids ray/aiohttp via package __init__)


def _load_math_grader():
    """Lazy-load math_utils bypassing slime.rollout.rm_hub.__init__ to avoid
    pulling ray/aiohttp into the custom_rm subprocess that doesn't need them.
    """
    global _MATH_GRADER, _BOXED_RE
    if _MATH_GRADER is not None:
        return _MATH_GRADER
    import importlib.util
    import re as _re
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    module_path = os.path.join(repo_root, "slime", "rollout", "rm_hub", "math_utils.py")
    spec = importlib.util.spec_from_file_location("_math_grader_standalone", module_path)
    if spec is None or spec.loader is None:
        logger.warning("[rubric] cannot load math_utils from %s; rule rewards will return 0", module_path)
        _MATH_GRADER = False  # sentinel: load failed, do not retry
        return False
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.warning("[rubric] math_utils load failed: %s; rule rewards will return 0", e)
        _MATH_GRADER = False
        return False
    _MATH_GRADER = module
    _BOXED_RE = _re.compile(r"\\boxed\s*\{")
    return module


def _format_score(response: str) -> float:
    """Return a [0, 1] score for output format quality.

    1.0  exactly one \\boxed{...} AND content has no '=' (clean LHS)
    0.5  has \\boxed{...} but content contains '=' (LHS=RHS pollution)
         OR has multiple \\boxed{...} (forces single decisive answer)
    0.0  no \\boxed{...} at all
    """
    if not response:
        return 0.0
    grader = _load_math_grader()
    if not grader:
        return 0.0
    matches = list(_BOXED_RE.finditer(response))
    if not matches:
        return 0.0
    if len(matches) > 1:
        return 0.5
    # Use math_utils.last_boxed_only_string + remove_boxed for brace-aware extraction
    try:
        boxed_str = grader.last_boxed_only_string(response)
        content = grader.remove_boxed(boxed_str) if boxed_str else None
    except Exception:
        content = None
    if content is None:
        return 0.5  # found token but failed brace-balanced extraction
    return 0.5 if "=" in content else 1.0


def _answer_score(response: str, label) -> float:
    """Return 1.0 if grade_answer_verl judges the response correct, else 0.0.

    Returns 0.0 when label is None/empty (e.g. UF mode), no boxed extractable,
    or any grader exception (so a broken sample never crashes PPO).
    """
    if response is None or label is None:
        return 0.0
    label_str = str(label).strip()
    if not label_str:
        return 0.0
    grader = _load_math_grader()
    if not grader:
        return 0.0
    try:
        return 1.0 if bool(grader.grade_answer_verl(response, label_str)) else 0.0
    except Exception:
        return 0.0
# ---------------------------------------------------------


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


def _sample_status_value(sample):
    status = getattr(sample, "status", None)
    return getattr(status, "value", status)


def _empty_response_penalty(sample):
    response = getattr(sample, "response", None) or ""
    if not response.strip():
        return _EMPTY_RESPONSE_PENALTY
    return 0.0


def _disclaimer_penalty(sample):
    response = getattr(sample, "response", None) or ""
    head = response[:300].lower()
    if any(pattern in head for pattern in _DISCLAIMER_PATTERNS):
        return _DISCLAIMER_PENALTY
    return 0.0


def _truncation_penalty(sample):
    response_length = int(getattr(sample, "response_length", 0) or 0)
    if response_length <= 0:
        return 0.0

    status_value = _sample_status_value(sample)
    if status_value == "truncated" or response_length >= _MAX_RESPONSE_LEN:
        return _TRUNCATION_PENALTY

    if response_length <= _TRUNCATION_THRESHOLD:
        return 0.0

    ramp_width = max(1, _MAX_RESPONSE_LEN - _TRUNCATION_THRESHOLD)
    near_limit_ratio = (response_length - _TRUNCATION_THRESHOLD) / ramp_width
    near_limit_ratio = min(1.0, max(0.0, near_limit_ratio))
    return _TRUNCATION_PENALTY * (near_limit_ratio ** 2)


def _apply_reward_shaping(reward, sample):
    global _RUBRIC_CALL_COUNTER
    if _RUBRIC_ENABLED:
        irl_component = _RUBRIC_W_IRL * reward
        response = getattr(sample, "response", None) or ""
        label = getattr(sample, "label", None)
        fmt = _format_score(response)
        ans = _answer_score(response, label)
        format_component = _RUBRIC_W_FORMAT * fmt
        answer_component = _RUBRIC_W_ANSWER * ans
        shaped_reward = irl_component + format_component + answer_component
        _RUBRIC_CALL_COUNTER += 1
        if _RUBRIC_LOG_EVERY > 0 and (_RUBRIC_CALL_COUNTER % _RUBRIC_LOG_EVERY) == 1:
            logger.info(
                "[rubric] sample#%d r_irl=%.4f w_irl*r=%.4f fmt=%.2f w_fmt*fmt=%.4f "
                "ans=%.2f w_ans*ans=%.4f shaped=%.4f label_present=%s",
                _RUBRIC_CALL_COUNTER, float(reward), irl_component,
                fmt, format_component, ans, answer_component, shaped_reward,
                label is not None,
            )
    else:
        shaped_reward = reward
    shaped_reward -= _empty_response_penalty(sample)
    shaped_reward -= _disclaimer_penalty(sample)
    shaped_reward -= _truncation_penalty(sample)
    return shaped_reward


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
