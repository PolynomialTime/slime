#!/usr/bin/env bash
# One-shot dataset pre-fetcher for the Open LLM Leaderboard v1 task set.
# Run this ONCE on a network-capable machine (login node n1ko).
# GPU nodes are assumed offline and will later read the cache from gpfs.
#
# Datasets (~260 MB total):
#   Rowan/hellaswag
#   allenai/winogrande (winogrande_xl)
#   truthfulqa/truthful_qa (multiple_choice)
#   openai/gsm8k (main)
#   cais/mmlu (57 subjects)
#
# Usage:
#   bash scripts/prefetch-leaderboard-datasets.sh
#
# Overridable env vars:
#   HF_HOME_DIR          target cache root on gpfs (default: /mnt/.../hf_cache)
#   HF_MIRROR_ENDPOINT   mirror URL (default: https://hf-mirror.com)
#   CONDA_BASE           conda install root (default: /home/wangqianyi/anaconda3)
#   CONDA_ENV_PATH       conda env with `datasets` installed (default: .../envs/leaderboard)

set -euo pipefail

HF_HOME_DIR="${HF_HOME_DIR:-/mnt/shared-storage-gpfs2/wangqianyi2/hf_cache}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
CONDA_BASE="${CONDA_BASE:-/home/wangqianyi/anaconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/leaderboard}"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# -----------------------------------------------------------------------------
# 1. Activate the conda env that has `datasets` installed
# -----------------------------------------------------------------------------
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_PREFIX}" != "${CONDA_ENV_PATH}" ]]; then
  conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_sh}" ]]; then
    die "conda.sh not found at ${conda_sh}; set CONDA_BASE or pre-activate ${CONDA_ENV_PATH}"
  fi
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${CONDA_ENV_PATH}"
fi

# -----------------------------------------------------------------------------
# 2. Point every HF cache at gpfs; pick a reachable endpoint
# -----------------------------------------------------------------------------
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p "${HF_HOME_DIR}/datasets" "${HF_HOME_DIR}/hub"

reachable() {
  # HEAD can be rejected by some CDNs; use a tiny GET and discard the body.
  curl -fsSL --connect-timeout 5 --max-time 15 -o /dev/null "$1" 2>/dev/null
}

if reachable "${HF_MIRROR_ENDPOINT}"; then
  export HF_ENDPOINT="${HF_MIRROR_ENDPOINT}"
elif reachable "https://huggingface.co"; then
  export HF_ENDPOINT="https://huggingface.co"
else
  die "neither ${HF_MIRROR_ENDPOINT} nor https://huggingface.co is reachable"
fi
log "endpoint: ${HF_ENDPOINT}"
log "cache:    ${HF_HOME_DIR}"

# -----------------------------------------------------------------------------
# 3. Download 5 datasets via HuggingFace datasets (57 calls for MMLU subjects)
# -----------------------------------------------------------------------------
python - <<'PY'
from datasets import get_dataset_config_names, load_dataset

fixed = [
    ("Rowan/hellaswag",         None),
    ("allenai/winogrande",      "winogrande_xl"),
    ("truthfulqa/truthful_qa",  "multiple_choice"),
    ("openai/gsm8k",            "main"),
]

for path, name in fixed:
    label = path if name is None else f"{path}:{name}"
    print(f"[prefetch] {label}", flush=True)
    ds = load_dataset(path) if name is None else load_dataset(path, name)
    print({k: len(v) for k, v in ds.items()}, flush=True)

configs = [c for c in get_dataset_config_names("cais/mmlu") if c != "all"]
print(f"[prefetch] cais/mmlu -- {len(configs)} subjects", flush=True)
for cfg in configs:
    ds = load_dataset("cais/mmlu", cfg)
    print(f"  - {cfg}: {dict((k, len(v)) for k, v in ds.items())}", flush=True)
PY

# -----------------------------------------------------------------------------
# 4. Size report
# -----------------------------------------------------------------------------
log "done; disk usage:"
du -sh "${HF_HOME_DIR}"/* 2>/dev/null || true
