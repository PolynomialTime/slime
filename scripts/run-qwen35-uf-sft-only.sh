#!/bin/bash
set -euo pipefail

# Run only the Qwen3.5-9B UltraFeedback SFT phase from the slime pipeline and
# export the SFT checkpoint to HF format. No reward/PPO/GRPO phase is launched.

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"

export PATH=/usr/local/nvidia/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$SLIME:/root/Megatron-LM/${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

RUN_TAG=${RUN_TAG:-qwen35_uf_sft_slime_$(date +%Y%m%d_%H%M%S)}
MODEL_ROOT=${MODEL_ROOT:-/mnt/shared-storage-user/ma4agi-gpu/wangqianyi}
MODEL_SH=${MODEL_SH:-scripts/models/qwen3.5-9B.sh}
MODEL_NAME_FOR_EXPORT=${MODEL_NAME_FOR_EXPORT:-qwen3_5}

QWEN35_HUB_ROOT=${QWEN35_HUB_ROOT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-9B-Base}
if [ -n "${QWEN35_SNAPSHOT:-}" ]; then
  QWEN35_HF_DIR="$QWEN35_HUB_ROOT/snapshots/$QWEN35_SNAPSHOT"
elif [ -s "$QWEN35_HUB_ROOT/refs/main" ]; then
  QWEN35_HF_DIR="$QWEN35_HUB_ROOT/snapshots/$(cat "$QWEN35_HUB_ROOT/refs/main")"
else
  QWEN35_HF_DIR=$(find "$QWEN35_HUB_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')
fi
if [ -z "$QWEN35_HF_DIR" ] || [ ! -f "$QWEN35_HF_DIR/config.json" ]; then
  echo "ERROR: cannot resolve Qwen3.5 HF snapshot under $QWEN35_HUB_ROOT" >&2
  exit 1
fi

mkdir -p "$MODEL_ROOT"
ln -sfn "$QWEN35_HF_DIR" "$MODEL_ROOT/qwen3.5-9B-base"

BASE_HF_DIR=${BASE_HF_DIR:-$MODEL_ROOT/qwen3.5-9B-base}
BASE_TORCH_DIST_DIR=${BASE_TORCH_DIST_DIR:-$MODEL_ROOT/qwen3.5-9b-base_torch_dist}
SFT_HF_DIR=${SFT_HF_DIR:-$MODEL_ROOT/${RUN_TAG}_hf}
SFT_MEGATRON_DIR=${SFT_MEGATRON_DIR:-$MODEL_ROOT/${RUN_TAG}_megatron}

ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}
SFT_WARMUP_SAMPLES=${SFT_WARMUP_SAMPLES:-10000}
SFT_WARMUP_SEED=${SFT_WARMUP_SEED:-42}
SFT_SYNTH_FULL_DATA_PATH=${SFT_SYNTH_FULL_DATA_PATH:-$ULTRAFEEDBACK_DIR/uf-sft.jsonl}
SFT_WARMUP_DATA_PATH=${SFT_WARMUP_DATA_PATH:-$ULTRAFEEDBACK_DIR/uf-sft-warmup${SFT_WARMUP_SAMPLES}.jsonl}
SFT_WARMUP_REPORT_PATH=${SFT_WARMUP_REPORT_PATH:-$ULTRAFEEDBACK_DIR/uf-sft-warmup${SFT_WARMUP_SAMPLES}.report.json}
SFT_DATA_PATH=${SFT_DATA_PATH:-$SFT_WARMUP_DATA_PATH}
SFT_NUM_EPOCHS=${SFT_NUM_EPOCHS:-1}

NUM_GPUS=${NUM_GPUS:-8}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-$MODEL_ROOT/${RUN_TAG}_artifacts/tensorboard_log}
LOG_DIR=${LOG_DIR:-$MODEL_ROOT/${RUN_TAG}_artifacts/logs}
mkdir -p "$LOG_DIR" "$TENSORBOARD_ROOT"

if [ ! -f "$SFT_DATA_PATH" ]; then
  if [ "$SFT_DATA_PATH" = "$SFT_WARMUP_DATA_PATH" ] && [ -f "$SFT_SYNTH_FULL_DATA_PATH" ]; then
    python3 scripts/sample_jsonl.py \
      --input "$SFT_SYNTH_FULL_DATA_PATH" \
      --output "$SFT_WARMUP_DATA_PATH" \
      --count "$SFT_WARMUP_SAMPLES" \
      --seed "$SFT_WARMUP_SEED" \
      --report "$SFT_WARMUP_REPORT_PATH"
  else
    echo "ERROR: missing SFT data at $SFT_DATA_PATH" >&2
    exit 1
  fi
fi

mapfile -t SFT_KEYS < <(python3 - "$SFT_DATA_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "messages" in row:
            print("messages")
            print("")
            raise SystemExit(0)
        if "text" in row and "label" in row:
            print("text")
            print("label")
            raise SystemExit(0)
        if "text" in row and "chosen" in row:
            print("text")
            print("chosen")
            raise SystemExit(0)
        raise SystemExit(f"Unsupported SFT row schema: {sorted(row.keys())}")
raise SystemExit("No JSONL rows found")
PY
)
SFT_INPUT_KEY=${SFT_KEYS[0]}
SFT_LABEL_KEY=${SFT_KEYS[1]:-}

echo "Qwen3.5 slime SFT-only config:"
echo "  run_tag=$RUN_TAG"
echo "  base_hf=$BASE_HF_DIR"
echo "  base_torch_dist=$BASE_TORCH_DIST_DIR"
echo "  sft_hf=$SFT_HF_DIR"
echo "  sft_megatron=$SFT_MEGATRON_DIR"
echo "  sft_data=$SFT_DATA_PATH"
echo "  sft_keys input=$SFT_INPUT_KEY label=${SFT_LABEL_KEY:-<none>}"
echo "  num_gpus=$NUM_GPUS tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE"

if [ ! -f "$BASE_TORCH_DIST_DIR/latest_checkpointed_iteration.txt" ] || \
   { [ ! -f "$BASE_TORCH_DIST_DIR/release/common.pt" ] && \
     [ ! -f "$BASE_TORCH_DIST_DIR/release/mp_rank_00_000/model_optim_rng.pt" ]; }; then
  MODEL_SH="$MODEL_SH" \
  BASE_HF_DIR="$BASE_HF_DIR" \
  BASE_TORCH_DIST_DIR="$BASE_TORCH_DIST_DIR" \
  bash scripts/run-convert-8b-job.sh
else
  echo "Skipping base conversion: found $BASE_TORCH_DIST_DIR"
fi

if [ ! -f "$SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt" ]; then
  MODEL_SH="$MODEL_SH" \
  HF_CKPT="$BASE_HF_DIR" \
  ACTOR_CKPT="$BASE_TORCH_DIST_DIR" \
  SAVE_DIR="$SFT_MEGATRON_DIR" \
  SFT_DATA="$SFT_DATA_PATH" \
  SFT_INPUT_KEY="$SFT_INPUT_KEY" \
  SFT_LABEL_KEY="$SFT_LABEL_KEY" \
  SFT_NUM_EPOCHS="$SFT_NUM_EPOCHS" \
  NUM_GPUS="$NUM_GPUS" \
  TENSOR_MODEL_PARALLEL_SIZE="$TENSOR_MODEL_PARALLEL_SIZE" \
  TENSORBOARD_DIR="$TENSORBOARD_ROOT/slime-sft/$RUN_TAG" \
  bash scripts/run-sft-prod.sh
else
  echo "Skipping SFT train: found $SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt"
fi

if [ ! -f "$SFT_HF_DIR/config.json" ]; then
  LATEST_ITER=$(cat "$SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt")
  ITER_DIR="$SFT_MEGATRON_DIR/iter_$(printf "%07d" "$LATEST_ITER")"
  if [ -f "$ITER_DIR/common.pt" ]; then
    CKPT_DIR="$ITER_DIR"
  elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then
    CKPT_DIR="$ITER_DIR/mp_rank_00"
  else
    echo "ERROR: common.pt not found under $ITER_DIR" >&2
    exit 1
  fi
  python3 tools/convert_torch_dist_to_hf.py \
    --model-name "$MODEL_NAME_FOR_EXPORT" \
    --input-dir "$CKPT_DIR" \
    --output-dir "$SFT_HF_DIR" \
    --origin-hf-dir "$BASE_HF_DIR" \
    --force
else
  echo "Skipping HF export: found $SFT_HF_DIR/config.json"
fi

cat > "$SFT_HF_DIR/SLIME_SFT_MANIFEST.txt" <<EOF
run_tag=$RUN_TAG
created_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
hostname=$(hostname)
slime_dir=$SLIME
git_head=$(git rev-parse --short HEAD 2>/dev/null || true)
base_hf=$BASE_HF_DIR
base_hf_resolved=$(readlink -f "$BASE_HF_DIR" 2>/dev/null || echo "$BASE_HF_DIR")
base_torch_dist=$BASE_TORCH_DIST_DIR
sft_hf=$SFT_HF_DIR
sft_megatron=$SFT_MEGATRON_DIR
sft_data=$SFT_DATA_PATH
sft_data_rows=$(awk 'END {print NR}' "$SFT_DATA_PATH")
sft_num_epochs=$SFT_NUM_EPOCHS
num_gpus=$NUM_GPUS
tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE
EOF

ln -sfnT "$SFT_HF_DIR" "$MODEL_ROOT/qwen3.5-9b-uf-sft_hf_latest"
ln -sfnT "$SFT_MEGATRON_DIR" "$MODEL_ROOT/qwen3.5-9b-uf-sft_megatron_latest"

echo "SFT complete:"
echo "  hf=$SFT_HF_DIR"
echo "  megatron=$SFT_MEGATRON_DIR"
echo "  canonical_hf_latest=$MODEL_ROOT/qwen3.5-9b-uf-sft_hf_latest"
echo "  canonical_megatron_latest=$MODEL_ROOT/qwen3.5-9b-uf-sft_megatron_latest"
