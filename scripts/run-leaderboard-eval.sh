#!/usr/bin/env bash
# Open LLM Leaderboard v1 runner for lm-evaluation-harness + vLLM on Qwen3-8B variants.
# Tasks & shot counts follow the original OLLB-v1 convention:
#   - hellaswag      10-shot, report acc_norm
#   - winogrande      5-shot, report acc
#   - gsm8k           5-shot, report exact_match (strict-match)
#   - mmlu            5-shot, aggregate acc across 57 subjects
#   - truthfulqa_mc2  0-shot (hard-coded in task yaml), report acc
#
# Subcommands:
#   prefetch              Run ONCE on a network-capable node (e.g. login n1ko).
#                         Downloads all 5 datasets into the shared HF cache on gpfs.
#   probe  [MODEL_NAME]   Run on a GPU node. 200-sample sanity run in BOTH base and
#                         chat prompting modes so the caller can decide which one to
#                         report. Default model: policy_r4_hf.
#   eval   [MODEL_NAME …] Run on a GPU node. Full OLLB-v1 split over one or more
#                         models (default: all 7). Pick prompt mode via EVAL_MODE.
#   summary               Print a TSV of per-run metrics scraped from results_*.json.
#
# Usage examples:
#   bash scripts/run-leaderboard-eval.sh prefetch
#   bash scripts/run-leaderboard-eval.sh probe policy_r4_hf
#   EVAL_MODE=base bash scripts/run-leaderboard-eval.sh eval
#   EVAL_MODE=base bash scripts/run-leaderboard-eval.sh eval policy_r1_hf policy_r4_hf
#   bash scripts/run-leaderboard-eval.sh summary
#
# Design notes:
#   - Login node and GPU node share the same script; environment is prepared per-stage.
#     Prefetch goes online (HF_ENDPOINT=https://hf-mirror.com); probe/eval run strictly
#     offline (HF_*_OFFLINE=1) so a surprise network hop fails loud instead of hanging.
#   - GPU node never reads /home/wangqianyi's conda env via docker; run this on a host
#     where /home is visible OR clone the env to /mnt/ first. See README for rjob notes.
#   - A failed model during `eval` does not abort the batch; failures are collected and
#     replayed via the retry hint printed at the end.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (all overridable via env vars)
# -----------------------------------------------------------------------------

SLIME_ROOT="${SLIME_ROOT:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}"
HF_HOME_DIR="${HF_HOME_DIR:-/mnt/shared-storage-gpfs2/wangqianyi2/hf_cache}"
MODELS_DIR="${MODELS_DIR:-${SLIME_ROOT}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLIME_ROOT}/eval}"

CONDA_BASE="${CONDA_BASE:-/home/wangqianyi/anaconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/leaderboard}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm-eval}"

# vLLM / generation knobs. Defaults tuned for Qwen3-8B on 8xH100/A100, bf16.
EVAL_MODE="${EVAL_MODE:-base}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-256}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
VLLM_SEED="${VLLM_SEED:-1234}"
LM_EVAL_SEED="${LM_EVAL_SEED:-0,1234,1234,1234}"

# Probe stage (subsample prompt-format comparison).
PROBE_LIMIT="${PROBE_LIMIT:-200}"
PROBE_DEFAULT_MODEL="${PROBE_DEFAULT_MODEL:-policy_r4_hf}"

# Summary stage output shape. SUMMARY_WITH_META=1 prefixes each row with "<model>/<mode>".
SUMMARY_WITH_META="${SUMMARY_WITH_META:-1}"
SUMMARY_HEADER="${SUMMARY_HEADER:-1}"

DEFAULT_MODELS=(
  qwen3-1.7B-base
  qwen3-8B-base
  sft_checkpoint_8b_hf
  policy_r1_hf
  policy_r2_hf
  policy_r3_hf
  policy_r4_hf
)

# Populated lazily by prepare_gpu_stage / resolve_lm_eval_cmd.
LM_EVAL_CMD=()
GPU_COUNT=""
DP_SIZE=""

SCRIPT_PATH="${BASH_SOURCE[0]}"

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

ts()      { date '+%Y-%m-%d %H:%M:%S'; }
ts_file() { date '+%Y%m%d_%H%M%S'; }
log()     { printf '[%s] %s\n' "$(ts)" "$*" >&2; }
die()     { log "ERROR: $*"; exit 1; }

usage() {
  cat >&2 <<USAGE
Usage:
  bash ${SCRIPT_PATH} prefetch
  bash ${SCRIPT_PATH} probe [MODEL_NAME=${PROBE_DEFAULT_MODEL}]
  EVAL_MODE=base|chat bash ${SCRIPT_PATH} eval [MODEL_NAME ...]
  bash ${SCRIPT_PATH} summary

Stages:
  prefetch  Online (login node). Populates \${HF_HOME_DIR}. Uses ${HF_MIRROR_ENDPOINT}.
  probe     Offline (GPU node). Base vs chat prompt comparison at --limit ${PROBE_LIMIT}.
  eval      Offline (GPU node). Full OLLB-v1 split over all or selected models.
  summary   Aggregates results_*.json below \${OUTPUT_DIR} into a TSV on stdout.
USAGE
}

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

# Activate the pinned conda env. Safe to call multiple times.
activate_conda() {
  if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX}" == "${CONDA_ENV_PATH}" ]]; then
    return 0
  fi

  local conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ -f "${conda_sh}" ]]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    die "conda not found; set CONDA_BASE or activate ${CONDA_ENV_PATH} first"
  fi

  conda activate "${CONDA_ENV_PATH}"
}

# Pin HF cache locations without toggling offline/online flags.
configure_hf_cache_env() {
  export HF_HOME="${HF_HOME_DIR}"
  export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
  export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
}

# Online login-node mode: HF traffic goes through the mirror.
setup_prefetch_env() {
  configure_hf_cache_env
  mkdir -p "${HF_HOME_DIR}/datasets" "${HF_HOME_DIR}/hub"
  export HF_ENDPOINT="${HF_ENDPOINT:-${HF_MIRROR_ENDPOINT}}"
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
}

# Offline GPU-node mode: any hub access fails immediately.
setup_offline_env() {
  configure_hf_cache_env
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  unset HF_ENDPOINT
}

url_reachable() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSIL --connect-timeout 5 --max-time 15 "${url}" >/dev/null 2>&1
  else
    python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10).close()' "${url}" >/dev/null 2>&1
  fi
}

check_prefetch_network() {
  if url_reachable "${HF_ENDPOINT}"; then
    log "network reachable: ${HF_ENDPOINT}"
    return 0
  fi
  if url_reachable "https://huggingface.co"; then
    export HF_ENDPOINT="https://huggingface.co"
    log "mirror unavailable; falling back to ${HF_ENDPOINT}"
    return 0
  fi
  die "neither ${HF_ENDPOINT} nor https://huggingface.co is reachable"
}

# -----------------------------------------------------------------------------
# GPU-stage readiness
# -----------------------------------------------------------------------------

resolve_lm_eval_cmd() {
  read -r -a LM_EVAL_CMD <<<"${LM_EVAL_BIN}"
  if [[ "${#LM_EVAL_CMD[@]}" -gt 0 ]] && command -v "${LM_EVAL_CMD[0]}" >/dev/null 2>&1; then
    return 0
  fi
  if python -c 'import lm_eval' >/dev/null 2>&1; then
    LM_EVAL_CMD=(python -m lm_eval)
    return 0
  fi
  die "lm-eval not found in ${CONDA_ENV_PATH}"
}

detect_gpu_count() {
  local n=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]' || true)"
  fi
  if ! [[ "${n}" =~ ^[1-9][0-9]*$ ]]; then
    n="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)"
    n="${n//[[:space:]]/}"
  fi
  [[ "${n}" =~ ^[1-9][0-9]*$ ]] || n="8"
  printf '%s\n' "${n}"
}

check_cuda_available() {
  python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || die "torch.cuda.is_available() is False; eval/probe must run on a GPU node"
}

check_datasets_cache() {
  local first=""
  if [[ -d "${HF_HOME_DIR}/datasets" ]]; then
    first="$(find "${HF_HOME_DIR}/datasets" -mindepth 1 -print -quit 2>/dev/null || true)"
  fi
  [[ -n "${first}" ]] \
    || die "dataset cache empty: ${HF_HOME_DIR}/datasets; run 'bash ${SCRIPT_PATH} prefetch' on n1ko first"
}

prepare_gpu_stage() {
  activate_conda
  setup_offline_env
  resolve_lm_eval_cmd
  check_datasets_cache
  check_cuda_available

  GPU_COUNT="$(detect_gpu_count)"
  DP_SIZE="${DATA_PARALLEL_SIZE:-${GPU_COUNT}}"

  [[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "TP_SIZE must be a positive integer"
  [[ "${DP_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "DATA_PARALLEL_SIZE must be a positive integer"

  if (( DP_SIZE > 1 )); then
    python -c 'import ray' >/dev/null 2>&1 \
      || die "data_parallel_size=${DP_SIZE} requires ray; pip install ray"
  fi

  mkdir -p "${OUTPUT_DIR}"
  log "offline HF_HOME=${HF_HOME_DIR}"
  log "vLLM plan: tp=${TP_SIZE} dp=${DP_SIZE} dtype=${DTYPE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
}

# -----------------------------------------------------------------------------
# lm-eval invocation
# -----------------------------------------------------------------------------

resolve_model_path() {
  local arg="$1"
  if [[ -d "${MODELS_DIR}/${arg}" ]]; then
    printf '%s\n' "${MODELS_DIR}/${arg}"
  elif [[ -d "${arg}" ]]; then
    printf '%s\n' "${arg}"
  else
    return 1
  fi
}

# Build the vLLM model_args string. DP>1 pins the ray executor explicitly so the
# stack trace on a misconfigured ray cluster is immediate.
build_model_args() {
  local model_path="$1"
  local mode="$2"
  local args="pretrained=${model_path}"
  args+=",tensor_parallel_size=${TP_SIZE}"
  args+=",data_parallel_size=${DP_SIZE}"
  args+=",dtype=${DTYPE}"
  args+=",max_model_len=${MAX_MODEL_LEN}"
  args+=",gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  args+=",max_num_seqs=${MAX_NUM_SEQS}"
  args+=",trust_remote_code=${TRUST_REMOTE_CODE}"
  args+=",seed=${VLLM_SEED}"

  if (( DP_SIZE > 1 )); then
    args+=",distributed_executor_backend=ray"
  fi
  if [[ "${mode}" == "chat" ]]; then
    args+=",enable_thinking=False"
  fi
  if [[ -n "${EXTRA_MODEL_ARGS:-}" ]]; then
    args+=",${EXTRA_MODEL_ARGS}"
  fi
  printf '%s\n' "${args}"
}

# Run a single lm-eval invocation, tee-ing to a per-task log file. Return its rc.
run_lm_eval() {
  local model_name="$1" model_path="$2" mode="$3"
  local stage="$4" task_label="$5" tasks="$6" shots="$7"
  local limit="${8:-}"

  local model_args out_dir logs_dir log_file now
  model_args="$(build_model_args "${model_path}" "${mode}")"
  out_dir="${OUTPUT_DIR}/${model_name}/${stage}/${task_label}"
  logs_dir="${OUTPUT_DIR}/${model_name}/logs"
  now="$(ts_file)"
  log_file="${logs_dir}/${stage}_${task_label}_${now}.log"

  mkdir -p "${out_dir}" "${logs_dir}"

  local -a cmd=(
    "${LM_EVAL_CMD[@]}" run
    --model vllm
    --model_args "${model_args}"
    --tasks "${tasks}"
    --num_fewshot "${shots}"
    --batch_size auto
    --gen_kwargs "temperature=0.0,do_sample=False,max_gen_toks=${MAX_GEN_TOKS}"
    --output_path "${out_dir}"
    --log_samples
    --seed "${LM_EVAL_SEED}"
  )
  if [[ "${mode}" == "chat" ]]; then
    cmd+=(--apply_chat_template --fewshot_as_multiturn)
  fi
  if [[ -n "${limit}" ]]; then
    cmd+=(--limit "${limit}")
  fi

  log "${model_name}: stage=${stage} tasks=${tasks} shots=${shots} limit=${limit:-none}"
  log "tee log: ${log_file}"

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local -a pipe_rc=("${PIPESTATUS[@]}")
  set -e

  local rc="${pipe_rc[0]}"
  if [[ "${rc}" == "0" && "${pipe_rc[1]:-0}" != "0" ]]; then
    rc="${pipe_rc[1]}"
  fi

  if [[ "${rc}" != "0" ]]; then
    log "${model_name}: FAILED stage=${stage} tasks=${tasks} rc=${rc}"
    return "${rc}"
  fi
  log "${model_name}: done stage=${stage} tasks=${tasks}"
  return 0
}

# OLLB-v1 shot split: hellaswag 10-shot, {winogrande,gsm8k,mmlu} 5-shot bundled,
# truthfulqa_mc2 on its own (yaml pins num_fewshot=0).
run_full_model() {
  local model_name="$1" model_path="$2" mode="$3"
  local rc=0
  local stage="eval_${mode}"

  run_lm_eval "${model_name}" "${model_path}" "${mode}" "${stage}" \
      "hellaswag_10shot" "hellaswag" 10 || rc=1
  run_lm_eval "${model_name}" "${model_path}" "${mode}" "${stage}" \
      "winogrande_gsm8k_mmlu_5shot" "winogrande,gsm8k,mmlu" 5 || rc=1
  run_lm_eval "${model_name}" "${model_path}" "${mode}" "${stage}" \
      "truthfulqa_mc2_0shot" "truthfulqa_mc2" 0 || rc=1

  return "${rc}"
}

# Probe mirrors full_model but adds --limit and runs both prompt modes in sequence.
run_probe_model() {
  local model_name="$1" model_path="$2"
  local rc=0

  for mode in base chat; do
    local stage="probe_${mode}"
    run_lm_eval "${model_name}" "${model_path}" "${mode}" "${stage}" \
        "hellaswag_10shot_limit${PROBE_LIMIT}" "hellaswag" 10 "${PROBE_LIMIT}" || rc=1
    run_lm_eval "${model_name}" "${model_path}" "${mode}" "${stage}" \
        "winogrande_gsm8k_mmlu_5shot_limit${PROBE_LIMIT}" "winogrande,gsm8k,mmlu" 5 "${PROBE_LIMIT}" || rc=1
    run_lm_eval "${model_name}" "${model_path}" "${mode}" "${stage}" \
        "truthfulqa_mc2_0shot_limit${PROBE_LIMIT}" "truthfulqa_mc2" 0 "${PROBE_LIMIT}" || rc=1
  done
  return "${rc}"
}

# -----------------------------------------------------------------------------
# Subcommand: prefetch
# -----------------------------------------------------------------------------

# Must not import torch/CUDA; login node has no GPU.
cmd_prefetch() {
  activate_conda
  setup_prefetch_env
  check_prefetch_network

  log "prefetch HF datasets into ${HF_HOME_DIR}"
  python - <<'PY'
from datasets import get_dataset_config_names, load_dataset

fixed = [
    ("Rowan/hellaswag", None),
    ("allenai/winogrande", "winogrande_xl"),
    ("truthfulqa/truthful_qa", "multiple_choice"),
    ("openai/gsm8k", "main"),
]

for path, name in fixed:
    label = path if name is None else f"{path}:{name}"
    print(f"[prefetch] {label}", flush=True)
    ds = load_dataset(path) if name is None else load_dataset(path, name)
    print({k: len(v) for k, v in ds.items()}, flush=True)

# cais/mmlu: 57 subjects. Load each so every subtask yaml finds its split.
configs = [c for c in get_dataset_config_names("cais/mmlu") if c != "all"]
for cfg in configs:
    print(f"[prefetch] cais/mmlu:{cfg}", flush=True)
    ds = load_dataset("cais/mmlu", cfg)
    print({k: len(v) for k, v in ds.items()}, flush=True)
PY

  log "prefetch complete"
  du -sh "${HF_HOME_DIR}"/* 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# Subcommand: probe
# -----------------------------------------------------------------------------

cmd_probe() {
  prepare_gpu_stage

  local model_arg="${1:-${PROBE_DEFAULT_MODEL}}"
  local model_path model_name
  model_path="$(resolve_model_path "${model_arg}")" \
    || die "model not found: ${model_arg}; MODELS_DIR=${MODELS_DIR}"
  model_name="$(basename "${model_path}")"

  run_probe_model "${model_name}" "${model_path}"
}

# -----------------------------------------------------------------------------
# Subcommand: eval
# -----------------------------------------------------------------------------

cmd_eval() {
  prepare_gpu_stage

  case "${EVAL_MODE}" in
    base|chat) ;;
    *) die "EVAL_MODE must be 'base' or 'chat', got '${EVAL_MODE}'" ;;
  esac

  local -a models=("$@")
  if [[ "${#models[@]}" -eq 0 ]]; then
    models=("${DEFAULT_MODELS[@]}")
  fi

  local -a failed=()
  local arg model_path model_name

  for arg in "${models[@]}"; do
    if ! model_path="$(resolve_model_path "${arg}")"; then
      log "SKIP missing model: ${arg}"
      failed+=("${arg}:missing")
      continue
    fi
    model_name="$(basename "${model_path}")"
    log "start model=${model_name} mode=${EVAL_MODE}"

    if run_full_model "${model_name}" "${model_path}" "${EVAL_MODE}"; then
      log "model complete: ${model_name}"
    else
      failed+=("${model_name}")
      log "model failed but continuing: ${model_name}"
    fi
  done

  if [[ "${#failed[@]}" -gt 0 ]]; then
    log "failed models: ${failed[*]}"
    log "retry: EVAL_MODE=${EVAL_MODE} bash ${SCRIPT_PATH} eval ${failed[*]}"
    return 1
  fi
}

# -----------------------------------------------------------------------------
# Subcommand: summary
# -----------------------------------------------------------------------------

cmd_summary() {
  activate_conda
  python - "${OUTPUT_DIR}" "${SUMMARY_WITH_META}" "${SUMMARY_HEADER}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with_meta = sys.argv[2] == "1"
want_header = sys.argv[3] == "1"

cols = [
    "hellaswag_acc_norm",
    "winogrande_acc",
    "truthfulqa_mc2_acc",
    "gsm8k_strict",
    "gsm8k_flex",
    "mmlu_acc",
]
mode_names = {"eval_base", "eval_chat", "probe_base", "probe_chat"}


def get(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def key_for(p):
    try:
        parts = p.relative_to(root).parts
    except ValueError:
        parts = p.parts
    model = parts[0] if parts else "."
    mode = next((x for x in parts if x in mode_names), "")
    return f"{model}/{mode}" if mode else model


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.10g}"
    return str(v)


rows: dict[str, dict[str, float]] = {}
for p in sorted(root.rglob("results*.json")):
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"# skip {p}: {exc}", file=sys.stderr)
        continue

    row = rows.setdefault(key_for(p), {})
    scrape = {
        "hellaswag_acc_norm":  get(data, "results", "hellaswag", "acc_norm,none"),
        "winogrande_acc":      get(data, "results", "winogrande", "acc,none"),
        "truthfulqa_mc2_acc":  get(data, "results", "truthfulqa_mc2", "acc,none"),
        "gsm8k_strict":        get(data, "results", "gsm8k", "exact_match,strict-match"),
        "gsm8k_flex":          get(data, "results", "gsm8k", "exact_match,flexible-extract"),
        "mmlu_acc":            get(data, "groups", "mmlu", "acc,none"),
    }
    for k, v in scrape.items():
        if v is not None:
            row[k] = v

if want_header:
    if with_meta:
        print("\t".join(["run"] + cols))
    else:
        print("\t".join(cols))

for key in sorted(rows):
    vals = [fmt(rows[key].get(c)) for c in cols]
    if with_meta:
        print("\t".join([key] + vals))
    else:
        print(f"# {key}", file=sys.stderr)
        print("\t".join(vals))
PY
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------

main() {
  local subcmd="${1:-}"
  if [[ -z "${subcmd}" ]]; then
    usage
    exit 2
  fi
  shift

  case "${subcmd}" in
    prefetch) cmd_prefetch "$@" ;;
    probe)    cmd_probe    "$@" ;;
    eval)     cmd_eval     "$@" ;;
    summary)  cmd_summary  "$@" ;;
    -h|--help|help) usage ;;
    *)        usage; die "unknown subcommand: ${subcmd}" ;;
  esac
}

main "$@"
