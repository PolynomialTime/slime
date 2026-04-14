#!/bin/bash
set -euo pipefail

SLIME=${SLIME:-/mnt/shared-storage-gpfs2/wangqianyi2/slime}
cd "$SLIME"
export PYTHONPATH="$SLIME${PYTHONPATH:+:$PYTHONPATH}"

ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}
MODEL_SH=${MODEL_SH:-scripts/models/qwen3-8B.sh}
SFT_HF_DIR=${SFT_HF_DIR:-$SLIME/models/sft_checkpoint_8b_hf}
SFT_MEGATRON_DIR=${SFT_MEGATRON_DIR:-$SLIME/models/sft_checkpoint}
ROUND1_REWARD_DIR=${ROUND1_REWARD_DIR:-$SLIME/models/reward_model}
REWARD_MODEL_DIR=${REWARD_MODEL_DIR:-$SLIME/models/reward_model_round1_frozen}
REWARD_SNAPSHOT_DIR=${REWARD_SNAPSHOT_DIR:-$REWARD_MODEL_DIR/latest}
PROMPT_DATA=${PROMPT_DATA:-$ULTRAFEEDBACK_DIR/uf-train.jsonl}
DEMO_DATA=${DEMO_DATA:-$ULTRAFEEDBACK_DIR/uf-train.jsonl}
SAVE_DIR=${SAVE_DIR:-$SLIME/models/save_dir_single_ppo_r1_reward}
CRITIC_SAVE_DIR=${CRITIC_SAVE_DIR:-$SLIME/models/critic_single_ppo_r1_reward}
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-$SLIME/rollout/single_ppo_r1_reward}
ROLLOUT_DEBUG_PATH_TEMPLATE=${ROLLOUT_DEBUG_PATH_TEMPLATE:-}
if [ -z "${ROLLOUT_DEBUG_PATH_TEMPLATE}" ]; then
  ROLLOUT_DEBUG_PATH_TEMPLATE="${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
fi
METADATA_DIR=${METADATA_DIR:-$SLIME/logs/single_ppo}
METADATA_PATH=${METADATA_PATH:-$METADATA_DIR/single_ppo_r1_reward.metadata.json}

NUM_ROLLOUT=${NUM_ROLLOUT:-600}
PPO_SAVE_INTERVAL=${PPO_SAVE_INTERVAL:-50}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-768}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.2}
ALIGN_ROLLOUT_WITH_SFT=${ALIGN_ROLLOUT_WITH_SFT:-1}
TB_EXP_NAME=${TB_EXP_NAME:-single_ppo_r1_reward}
ALLOW_EXISTING_SAVE_DIR=${ALLOW_EXISTING_SAVE_DIR:-0}

require_file() {
  local path=$1
  if [ ! -e "$path" ]; then
    echo "ERROR: missing required path: $path" >&2
    exit 1
  fi
}

require_positive_int() {
  local value=$1
  local name=$2
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
    echo "ERROR: $name must be a positive integer, got: $value" >&2
    exit 1
  fi
}

dir_has_contents() {
  local path=$1
  [ -d "$path" ] && find "$path" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .
}

ensure_round1_reward_snapshot() {
  require_file "$ROUND1_REWARD_DIR/latest/config.json"

  if [ -f "$REWARD_SNAPSHOT_DIR/config.json" ]; then
    echo "Using existing frozen Round1 reward snapshot at $REWARD_SNAPSHOT_DIR"
    return 0
  fi

  mkdir -p "$REWARD_MODEL_DIR"

  if ! python3 - "$ROUND1_REWARD_DIR" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
later_round = False

for pattern in ("reward_eval_round_*.json", "reward_eval_external_round_*.json"):
    for path in root.glob(pattern):
        match = re.search(r"_round_(\d+)\.json$", path.name)
        if match and int(match.group(1)) > 1:
            later_round = True
            break
    if later_round:
        break

if not later_round:
    for path in root.glob("step_round*"):
        match = re.search(r"round(\d+)$", path.name)
        if match and int(match.group(1)) > 1:
            later_round = True
            break

raise SystemExit(1 if later_round else 0)
PY
  then
    echo "ERROR: $ROUND1_REWARD_DIR shows round2+ reward traces, but frozen Round1 snapshot is missing at $REWARD_SNAPSHOT_DIR" >&2
    exit 1
  fi

  local tmp_dir
  tmp_dir=$(mktemp -d "$REWARD_MODEL_DIR/.latest.tmp.XXXXXX")

  python3 - "$ROUND1_REWARD_DIR/latest" "$tmp_dir" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
shutil.copytree(src, dst, dirs_exist_ok=True)
PY

  mv "$tmp_dir" "$REWARD_SNAPSHOT_DIR"
  echo "Created frozen Round1 reward snapshot at $REWARD_SNAPSHOT_DIR"
}

write_metadata() {
  mkdir -p "$METADATA_DIR" "$SAVE_DIR" "$CRITIC_SAVE_DIR" "$ROLLOUT_DEBUG_DIR"

  python3 - \
    "$METADATA_PATH" \
    "$MODEL_SH" \
    "$SFT_HF_DIR" \
    "$SFT_MEGATRON_DIR" \
    "$PROMPT_DATA" \
    "$DEMO_DATA" \
    "$ROUND1_REWARD_DIR" \
    "$REWARD_SNAPSHOT_DIR" \
    "$SAVE_DIR" \
    "$CRITIC_SAVE_DIR" \
    "$ROLLOUT_DEBUG_PATH_TEMPLATE" \
    "$NUM_ROLLOUT" \
    "$PPO_SAVE_INTERVAL" \
    "$ROLLOUT_BATCH_SIZE" \
    "$ROLLOUT_MAX_RESPONSE_LEN" \
    "$ROLLOUT_TEMPERATURE" \
    "$ALIGN_ROLLOUT_WITH_SFT" \
    "$TB_EXP_NAME" \
    "${ACTOR_LR:-}" \
    "${CRITIC_LR:-}" \
    "${KL_LOSS_COEF:-}" <<'PY'
import json
import sys
from datetime import datetime, timezone

(
    path,
    model_sh,
    hf_ckpt,
    megatron_ckpt,
    prompt_data,
    demo_data,
    reward_source_dir,
    reward_snapshot_dir,
    save_dir,
    critic_save_dir,
    rollout_debug_path_template,
    num_rollout,
    ppo_save_interval,
    rollout_batch_size,
    rollout_max_response_len,
    rollout_temperature,
    align_rollout_with_sft,
    tb_experiment_name,
    actor_lr,
    critic_lr,
    kl_loss_coef,
) = sys.argv[1:]

payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "single_ppo_fixed_round1_reward",
    "model_sh": model_sh,
    "hf_ckpt": hf_ckpt,
    "actor_load": megatron_ckpt,
    "ref_ckpt": megatron_ckpt,
    "prompt_data": prompt_data,
    "demo_data": demo_data,
    "reward_source_dir": reward_source_dir,
    "reward_snapshot_dir": reward_snapshot_dir,
    "save_dir": save_dir,
    "critic_save_dir": critic_save_dir,
    "rollout_debug_path_template": rollout_debug_path_template,
    "num_rollout": int(num_rollout),
    "ppo_save_interval": int(ppo_save_interval),
    "rollout_batch_size": int(rollout_batch_size),
    "rollout_max_response_len": int(rollout_max_response_len),
    "rollout_temperature": float(rollout_temperature),
    "align_rollout_with_sft": int(align_rollout_with_sft),
    "tb_experiment_name": tb_experiment_name,
    "actor_lr": actor_lr or None,
    "critic_lr": critic_lr or None,
    "kl_loss_coef": kl_loss_coef or None,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
PY
}

require_positive_int "$NUM_ROLLOUT" "NUM_ROLLOUT"
require_positive_int "$PPO_SAVE_INTERVAL" "PPO_SAVE_INTERVAL"
require_positive_int "$ROLLOUT_BATCH_SIZE" "ROLLOUT_BATCH_SIZE"
require_positive_int "$ROLLOUT_MAX_RESPONSE_LEN" "ROLLOUT_MAX_RESPONSE_LEN"

require_file "$MODEL_SH"
require_file "$SFT_HF_DIR/config.json"
require_file "$SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt"
require_file "$PROMPT_DATA"
require_file "$DEMO_DATA"

if [ -e "$REWARD_SNAPSHOT_DIR" ] && [ ! -f "$REWARD_SNAPSHOT_DIR/config.json" ]; then
  echo "ERROR: found incomplete frozen reward snapshot at $REWARD_SNAPSHOT_DIR" >&2
  exit 1
fi

if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ] && [ "$ALLOW_EXISTING_SAVE_DIR" != "1" ]; then
  echo "ERROR: existing checkpoint state found at $SAVE_DIR; remove it first or set ALLOW_EXISTING_SAVE_DIR=1" >&2
  exit 1
fi

if [ "$ALLOW_EXISTING_SAVE_DIR" != "1" ]; then
  for path in "$SAVE_DIR" "$CRITIC_SAVE_DIR" "$ROLLOUT_DEBUG_DIR"; do
    if dir_has_contents "$path"; then
      echo "ERROR: found existing experiment artifacts at $path; remove them first or set ALLOW_EXISTING_SAVE_DIR=1" >&2
      exit 1
    fi
  done
fi

ensure_round1_reward_snapshot
write_metadata

echo "Single PPO experiment config:"
echo "  model_sh=$MODEL_SH"
echo "  hf_ckpt=$SFT_HF_DIR"
echo "  actor_load=$SFT_MEGATRON_DIR"
echo "  ref_ckpt=$SFT_MEGATRON_DIR"
echo "  reward_model_dir=$REWARD_MODEL_DIR"
echo "  reward_snapshot_dir=$REWARD_SNAPSHOT_DIR"
echo "  prompt_data=$PROMPT_DATA"
echo "  save_dir=$SAVE_DIR"
echo "  critic_save_dir=$CRITIC_SAVE_DIR"
echo "  rollout_debug_path=$ROLLOUT_DEBUG_PATH_TEMPLATE"
echo "  num_rollout=$NUM_ROLLOUT"
echo "  save_interval=$PPO_SAVE_INTERVAL"
echo "  rollout_batch_size=$ROLLOUT_BATCH_SIZE"
echo "  rollout_max_response_len=$ROLLOUT_MAX_RESPONSE_LEN"
echo "  rollout_temperature=$ROLLOUT_TEMPERATURE"
echo "  align_rollout_with_sft=$ALIGN_ROLLOUT_WITH_SFT"
echo "  tb_experiment=$TB_EXP_NAME"
echo "  metadata_path=$METADATA_PATH"

MODEL_SH="$MODEL_SH" \
HF_CKPT="$SFT_HF_DIR" \
REF_CKPT="$SFT_MEGATRON_DIR" \
ACTOR_LOAD="$SFT_MEGATRON_DIR" \
SAVE_DIR="$SAVE_DIR" \
PROMPT_DATA="$PROMPT_DATA" \
DEMO_DATA="$DEMO_DATA" \
NUM_ROLLOUT="$NUM_ROLLOUT" \
PPO_SAVE_INTERVAL="$PPO_SAVE_INTERVAL" \
ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
ROLLOUT_MAX_RESPONSE_LEN="$ROLLOUT_MAX_RESPONSE_LEN" \
ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE" \
ALIGN_ROLLOUT_WITH_SFT="$ALIGN_ROLLOUT_WITH_SFT" \
TB_EXP_NAME="$TB_EXP_NAME" \
REWARD_MODEL_DIR="$REWARD_MODEL_DIR" \
CRITIC_SAVE_DIR="$CRITIC_SAVE_DIR" \
ROLLOUT_DEBUG_DIR="$ROLLOUT_DEBUG_DIR" \
ROLLOUT_DEBUG_PATH_TEMPLATE="$ROLLOUT_DEBUG_PATH_TEMPLATE" \
bash scripts/run-irl-prod.sh
