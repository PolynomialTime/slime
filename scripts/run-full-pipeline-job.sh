#!/bin/bash
# Full Pipeline v6: SFT → 7 rounds of (Reward Update → PPO → Test-set export)
# Each round exports test-set outputs immediately, then prunes transient artifacts.
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME
export PYTHONPATH="$SLIME${PYTHONPATH:+:$PYTHONPATH}"
ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}
DATA_MODE=${DATA_MODE:-ultrafeedback}
MODEL_ROOT=${MODEL_ROOT:-$SLIME/models}
MODEL_SH=${MODEL_SH:-scripts/models/qwen3-8B.sh}
MODEL_NAME_FOR_EXPORT=${MODEL_NAME_FOR_EXPORT:-}
EVAL_DIR=${EVAL_DIR:-$SLIME/eval}
EVAL_WINRATE_DIR=${EVAL_WINRATE_DIR:-$SLIME/eval_winrate}
ROLLOUT_DIR=${ROLLOUT_DIR:-$SLIME/rollout}
TENSORBOARD_ROOT=${TENSORBOARD_ROOT:-$SLIME/tensorboard_log}
REWARD_DIR=${REWARD_DIR:-$MODEL_ROOT/reward_model}
REWARD_MODEL_INIT=${REWARD_MODEL_INIT:-}
BASE_HF_DIR=${BASE_HF_DIR:-$MODEL_ROOT/qwen3-8B-base}
BASE_TORCH_DIST_DIR=${BASE_TORCH_DIST_DIR:-$MODEL_ROOT/qwen3-8b-base_torch_dist}
SFT_HF_DIR=${SFT_HF_DIR:-$MODEL_ROOT/sft_checkpoint_8b_hf}
SFT_MEGATRON_DIR=${SFT_MEGATRON_DIR:-$MODEL_ROOT/sft_checkpoint}
SFT_BASELINE=${SFT_BASELINE:-$EVAL_DIR/outputs_sft_baseline.jsonl}

if [ "$DATA_MODE" = "math" ]; then
  # Math defaults applied only when user hasn't overridden them via env.
  : "${SFT_SYNTH_FULL_DATA_PATH:=$SLIME/math/train_demo.jsonl}"
  # 8B base needs a real math SFT pass (format + reasoning style) before IRL
  # can produce a useful reward signal. 10k proved too thin (R1 pass@1 22.5%,
  # AIME24 0/30); 80k subset of NuminaMath gives the base enough coverage.
  : "${SFT_WARMUP_SAMPLES:=80000}"
  # Keep warmup artifacts inside $SLIME/math/ -- default template would write
  # them under $ULTRAFEEDBACK_DIR which is semantically wrong in math mode.
  : "${SFT_WARMUP_DATA_PATH:=$SLIME/math/train_demo-warmup${SFT_WARMUP_SAMPLES}.jsonl}"
  : "${SFT_WARMUP_REPORT_PATH:=$SLIME/math/train_demo-warmup${SFT_WARMUP_SAMPLES}.report.json}"
  : "${BT_PRETRAIN_ENABLED:=0}"
  : "${WINRATE_GATE_ENABLED:=0}"
  : "${REWARD_TRAIN_DATA_PATH:=$SLIME/math/train_demo.jsonl}"
  : "${REWARD_EXTERNAL_EVAL_PATH:=}"
  # Rolling retention for reward snapshots: after round N completes, prune
  # step_round{1..N-1} and keep only step_round{N} + latest/.
  : "${KEEP_FINAL_REWARD_PER_ROUND:=0}"
  # Preserve explicit empty string in math mode so external eval short-circuits.
  : "${REWARD_EVAL_REJECTED_KEY_PASS=}"
  : "${TEST_DATA:=$SLIME/math/eval_prompts.jsonl}"
  : "${ROLLOUT_MAX_RESPONSE_LEN:=2048}"
  : "${PROMPT_DATA:=$SLIME/math/train_prompts.jsonl}"
  : "${DEMO_DATA:=$SLIME/math/train_prompts.jsonl}"
  # Rubric multi-source reward in math mode:
  # final = w_irl * r_irl + w_format * format_score + w_answer * answer_score.
  : "${SLIME_RUBRIC_ENABLED:=1}"
  : "${SLIME_RUBRIC_W_IRL:=1.0}"
  : "${SLIME_RUBRIC_W_FORMAT:=0.5}"
  : "${SLIME_RUBRIC_W_ANSWER:=1.0}"
  # PROMPT_LABEL_KEY=label so rollout samples carry ground truth for answer_score.
  : "${PROMPT_LABEL_KEY_PASS=label}"
fi
: "${REWARD_EVAL_REJECTED_KEY_PASS=rejected}"
: "${SLIME_RUBRIC_ENABLED:=0}"
: "${SLIME_RUBRIC_PLUGINS:=}"
: "${SLIME_RUBRIC_W_IRL:=1.0}"
: "${SLIME_RUBRIC_W_FORMAT:=0.5}"
: "${SLIME_RUBRIC_W_ANSWER:=1.0}"
: "${PROMPT_LABEL_KEY_PASS=}"
: "${N_SAMPLES_PER_PROMPT:=1}"
: "${ADVANTAGE_ESTIMATOR:=ppo}"
: "${NORMALIZE_ADVANTAGES:=1}"
SFT_SYNTH_FULL_DATA_PATH=${SFT_SYNTH_FULL_DATA_PATH:-${SFT_SYNTH_DATA_PATH:-$ULTRAFEEDBACK_DIR/uf-sft.jsonl}}
SFT_SYNTH_REPORT_PATH=${SFT_SYNTH_REPORT_PATH:-$ULTRAFEEDBACK_DIR/uf-sft.report.json}
SFT_WARMUP_SAMPLES=${SFT_WARMUP_SAMPLES:-10000}
SFT_WARMUP_SEED=${SFT_WARMUP_SEED:-42}
_SFT_FULL_STEM=$(basename "$SFT_SYNTH_FULL_DATA_PATH" .jsonl)
SFT_WARMUP_DATA_PATH=${SFT_WARMUP_DATA_PATH:-$ULTRAFEEDBACK_DIR/${_SFT_FULL_STEM}-warmup${SFT_WARMUP_SAMPLES}.jsonl}
SFT_WARMUP_REPORT_PATH=${SFT_WARMUP_REPORT_PATH:-$ULTRAFEEDBACK_DIR/${_SFT_FULL_STEM}-warmup${SFT_WARMUP_SAMPLES}.report.json}

KEEP_ALL_ROUND_CHECKPOINTS=${KEEP_ALL_ROUND_CHECKPOINTS:-0}
KEEP_ALL_ROUND_REWARD_SNAPSHOTS=${KEEP_ALL_ROUND_REWARD_SNAPSHOTS:-0}
KEEP_FINAL_POLICY_PER_ROUND=${KEEP_FINAL_POLICY_PER_ROUND:-1}
KEEP_FINAL_REWARD_PER_ROUND=${KEEP_FINAL_REWARD_PER_ROUND:-1}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-}
MIN_FREE_DISK_GB_PPO=${MIN_FREE_DISK_GB_PPO:-${MIN_FREE_DISK_GB:-60}}
MIN_FREE_DISK_GB_EXPORT=${MIN_FREE_DISK_GB_EXPORT:-${MIN_FREE_DISK_GB:-30}}
START_ROUND=${START_ROUND:-1}
FROM_SCRATCH=${FROM_SCRATCH:-0}
SFT_NUM_EPOCHS=${SFT_NUM_EPOCHS:-1}
NUM_ROUNDS=${NUM_ROUNDS:-7}
NUM_ROLLOUT_PER_ROUND=${NUM_ROLLOUT_PER_ROUND:-75}
BOOTSTRAP_NUM_ROLLOUT=${BOOTSTRAP_NUM_ROLLOUT:-$NUM_ROLLOUT_PER_ROUND}
BOOTSTRAP_ROLLOUT_TEMPERATURE=${BOOTSTRAP_ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-768}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.4}
ROLLOUT_HEALTH_MAX_TRUNCATED=${ROLLOUT_HEALTH_MAX_TRUNCATED:-0.2}
PPO_START_FROM_SFT=${PPO_START_FROM_SFT:-0}
# Round 1 still uses SFT as ref; later rounds default to previous round checkpoint.
PPO_REF_FIXED_TO_SFT=${PPO_REF_FIXED_TO_SFT:-0}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
ACTOR_LR=${ACTOR_LR:-3e-6}
CRITIC_LR=${CRITIC_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.05}
SLIME_CUSTOM_RM_TRUNCATION_PENALTY=${SLIME_CUSTOM_RM_TRUNCATION_PENALTY:-2.5}
SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC=${SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC:-0.8}
REWARD_EVAL_MAX_SAMPLES=${REWARD_EVAL_MAX_SAMPLES:-0}
REWARD_EVAL_SHUFFLE_SEED=${REWARD_EVAL_SHUFFLE_SEED:-42}
REWARD_UPDATE_EPOCHS=${REWARD_UPDATE_EPOCHS:-1}
REWARD_UPDATE_BATCH_SIZE=${REWARD_UPDATE_BATCH_SIZE:-4}
REWARD_ONLINE_PREF_WEIGHT=${REWARD_ONLINE_PREF_WEIGHT:-1.0}
# Online winrate judging must stay opt-in; GPU training runs on the cluster are offline.
WINRATE_GATE_ENABLED=${WINRATE_GATE_ENABLED:-0}
WINRATE_GATE_MAX_ROUND=${WINRATE_GATE_MAX_ROUND:-2}
WINRATE_GATE_MAX_SAMPLES=${WINRATE_GATE_MAX_SAMPLES:-512}
WINRATE_GATE_CONCURRENCY=${WINRATE_GATE_CONCURRENCY:-16}
WINRATE_GATE_MIN=${WINRATE_GATE_MIN:-0.48}
INTERROUND_WINRATE_GATE_MIN=${INTERROUND_WINRATE_GATE_MIN:-0.50}
WINRATE_GATE_MODEL=${WINRATE_GATE_MODEL:-gpt-4o}
WINRATE_GATE_API_KEY=${WINRATE_GATE_API_KEY:-${OPENAI_API_KEY:-${WINRATE_API_KEY:-}}}
WINRATE_GATE_BASE_URL=${WINRATE_GATE_BASE_URL:-${OPENAI_BASE_URL:-${WINRATE_BASE_URL:-}}}
WINRATE_GATE_FALLBACK_MODEL=${WINRATE_GATE_FALLBACK_MODEL:-claude-sonnet-4-6}
WINRATE_GATE_FALLBACK_API_KEY=${WINRATE_GATE_FALLBACK_API_KEY:-${WINRATE_FALLBACK_API_KEY:-}}
WINRATE_GATE_FALLBACK_BASE_URL=${WINRATE_GATE_FALLBACK_BASE_URL:-${WINRATE_FALLBACK_BASE_URL:-$WINRATE_GATE_BASE_URL}}
CODEX_RULES_PATH=${CODEX_RULES_PATH:-/mnt/shared-storage-gpfs2/wangqianyi2/slime-autoresearch-notify/CODEX_HARD_RULES.md}
BT_PRETRAIN_DIR=${BT_PRETRAIN_DIR:-$MODEL_ROOT/reward_model_bt_pretrain/latest}
BT_PRETRAIN_BASE_MODEL=${BT_PRETRAIN_BASE_MODEL:-$MODEL_ROOT/qwen3-8B-base}
BT_PRETRAIN_ENABLED=${BT_PRETRAIN_ENABLED:-1}
SFT_ENABLED=${SFT_ENABLED:-1}
REWARD_BT_FULL_DATA_PATH=${REWARD_BT_FULL_DATA_PATH:-$ULTRAFEEDBACK_DIR/uf-train-prefs.jsonl}
REWARD_BT_WARMUP_SAMPLES=${REWARD_BT_WARMUP_SAMPLES:-8000}
REWARD_BT_WARMUP_SEED=${REWARD_BT_WARMUP_SEED:-42}
_BT_FULL_STEM=$(basename "$REWARD_BT_FULL_DATA_PATH" .jsonl)
REWARD_BT_WARMUP_DATA_PATH=${REWARD_BT_WARMUP_DATA_PATH:-$ULTRAFEEDBACK_DIR/${_BT_FULL_STEM}-warmup${REWARD_BT_WARMUP_SAMPLES}.jsonl}
REWARD_BT_WARMUP_REPORT_PATH=${REWARD_BT_WARMUP_REPORT_PATH:-$ULTRAFEEDBACK_DIR/${_BT_FULL_STEM}-warmup${REWARD_BT_WARMUP_SAMPLES}.report.json}
REWARD_BT_EVAL_PATH=${REWARD_BT_EVAL_PATH:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}
REWARD_BT_EPOCHS=${REWARD_BT_EPOCHS:-1}
REWARD_BT_BATCH_SIZE=${REWARD_BT_BATCH_SIZE:-4}
REWARD_BT_EVAL_BATCH_SIZE=${REWARD_BT_EVAL_BATCH_SIZE:-8}
REWARD_BT_LR=${REWARD_BT_LR:-1e-6}
REWARD_BT_HOLDOUT_RATIO=${REWARD_BT_HOLDOUT_RATIO:-0.1}
REWARD_BT_EVAL_INTERVAL=${REWARD_BT_EVAL_INTERVAL:-100}
REWARD_BT_C_COEF_INIT=${REWARD_BT_C_COEF_INIT:-0.5}
TEST_DATA=${TEST_DATA:-$ULTRAFEEDBACK_DIR/uf-test.jsonl}
REWARD_TRAIN_DATA_PATH=${REWARD_TRAIN_DATA_PATH:-${REWARD_TRAIN_SYNTH_PATH:-$ULTRAFEEDBACK_DIR/uf-train-prefs.jsonl}}
if [ -z "${REWARD_EXTERNAL_EVAL_PATH+x}" ]; then
  REWARD_EXTERNAL_EVAL_PATH=$ULTRAFEEDBACK_DIR/uf-test.jsonl
fi
REWARD_EXTERNAL_EVAL_BATCH_SIZE=${REWARD_EXTERNAL_EVAL_BATCH_SIZE:-32}
PROMPT_DATA=${PROMPT_DATA:-$ULTRAFEEDBACK_DIR/uf-train.jsonl}
DEMO_DATA=${DEMO_DATA:-$ULTRAFEEDBACK_DIR/uf-train.jsonl}
EXPECTED_EVAL_LINES=2000
if [ -f "$TEST_DATA" ]; then
  ACTUAL_EVAL_LINES=$(awk 'END {print NR}' "$TEST_DATA")
  if [ "$ACTUAL_EVAL_LINES" -gt 0 ]; then
    EXPECTED_EVAL_LINES=$ACTUAL_EVAL_LINES
  fi
fi
EXPECTED_REWARD_EVAL_LINES=$EXPECTED_EVAL_LINES
if [ -f "$REWARD_EXTERNAL_EVAL_PATH" ]; then
  ACTUAL_REWARD_EVAL_LINES=$(awk 'END {print NR}' "$REWARD_EXTERNAL_EVAL_PATH")
  if [ "$ACTUAL_REWARD_EVAL_LINES" -gt 0 ]; then
    EXPECTED_REWARD_EVAL_LINES=$ACTUAL_REWARD_EVAL_LINES
  fi
fi

SFT_DATA_PATH=${SFT_DATA_PATH:-}
if [ -z "$SFT_DATA_PATH" ]; then
  if [ "$SFT_WARMUP_SAMPLES" -gt 0 ]; then
    SFT_DATA_PATH=$SFT_WARMUP_DATA_PATH
  elif [ -f "$SFT_SYNTH_FULL_DATA_PATH" ]; then
    SFT_DATA_PATH=$SFT_SYNTH_FULL_DATA_PATH
  else
    SFT_DATA_PATH=$ULTRAFEEDBACK_DIR/uf-sft.jsonl
  fi
fi

# ============ Global path-safety assertions (before any destructive cleanup) ============
# Block user env overrides that alias an rm target to a protected model/base path.
_BASE_PATHS_PROTECTED=(
  "$BASE_HF_DIR"
  "$BASE_TORCH_DIST_DIR"
  "$BT_PRETRAIN_BASE_MODEL"
  "$REWARD_MODEL_INIT"
  "$MODEL_ROOT/qwen3-8B-base"
  "$MODEL_ROOT/qwen3-8b-base_torch_dist"
  "$MODEL_ROOT/qwen3.5-9B-base"
  "$MODEL_ROOT/qwen3.5-9b-base_torch_dist"
  "$MODEL_ROOT/qwen3-1.7B-base"
  "$SLIME/models/qwen3-8B-base"
  "$SLIME/models/qwen3-8b-base_torch_dist"
  "$SLIME/models/qwen3-1.7B-base"
)
_assert_not_protected() {
  local var_name=$1
  local value=${!var_name}
  [ -n "$value" ] || return 0
  local rf_value
  rf_value=$(readlink -f "$value" 2>/dev/null || echo "$value")
  for protected in "${_BASE_PATHS_PROTECTED[@]}"; do
    [ -n "$protected" ] || continue
    local rf_protected
    rf_protected=$(readlink -f "$protected" 2>/dev/null || echo "$protected")
    if [ "$rf_value" = "$rf_protected" ]; then
      echo "FATAL: $var_name='$value' aliases protected path '$protected'" >&2
      echo "       Any rm -rf targeting this variable would destroy a model asset." >&2
      echo "       Refusing to continue. Fix the env override and retry." >&2
      exit 2
    fi
  done
  if [ "$rf_value" = "/" ] || [ "$rf_value" = "$SLIME" ] || [ "$rf_value" = "$SLIME/models" ] || [ "$rf_value" = "$MODEL_ROOT" ]; then
    echo "FATAL: $var_name='$value' points to a root-ish path ($rf_value); refusing." >&2
    exit 2
  fi
}
for _v in SFT_HF_DIR SFT_MEGATRON_DIR REWARD_DIR BT_PRETRAIN_DIR EVAL_DIR EVAL_WINRATE_DIR ROLLOUT_DIR TENSORBOARD_ROOT; do
  _assert_not_protected "$_v"
done
unset _v

echo "Pipeline config: DATA_MODE=$DATA_MODE MODEL_ROOT=$MODEL_ROOT MODEL_SH=$MODEL_SH MODEL_NAME_FOR_EXPORT=${MODEL_NAME_FOR_EXPORT:-<default>} START_ROUND=$START_ROUND NUM_ROUNDS=$NUM_ROUNDS NUM_ROLLOUT_PER_ROUND=$NUM_ROLLOUT_PER_ROUND FROM_SCRATCH=$FROM_SCRATCH KEEP_ALL_ROUND_CHECKPOINTS=$KEEP_ALL_ROUND_CHECKPOINTS KEEP_ALL_ROUND_REWARD_SNAPSHOTS=$KEEP_ALL_ROUND_REWARD_SNAPSHOTS KEEP_FINAL_POLICY_PER_ROUND=$KEEP_FINAL_POLICY_PER_ROUND KEEP_FINAL_REWARD_PER_ROUND=$KEEP_FINAL_REWARD_PER_ROUND REWARD_MODEL_INIT=${REWARD_MODEL_INIT:-<default>} SFT_HF_DIR=$SFT_HF_DIR SFT_MEGATRON_DIR=$SFT_MEGATRON_DIR SFT_DATA_PATH=$SFT_DATA_PATH SFT_WARMUP_SAMPLES=$SFT_WARMUP_SAMPLES SFT_NUM_EPOCHS=$SFT_NUM_EPOCHS EVAL_DIR=$EVAL_DIR ROLLOUT_DIR=$ROLLOUT_DIR TENSORBOARD_ROOT=$TENSORBOARD_ROOT EVAL_TEMPERATURE=$EVAL_TEMPERATURE BOOTSTRAP_NUM_ROLLOUT=$BOOTSTRAP_NUM_ROLLOUT BOOTSTRAP_ROLLOUT_TEMPERATURE=$BOOTSTRAP_ROLLOUT_TEMPERATURE ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE ROLLOUT_MAX_RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN ADVANTAGE_ESTIMATOR=$ADVANTAGE_ESTIMATOR N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT NORMALIZE_ADVANTAGES=$NORMALIZE_ADVANTAGES PPO_START_FROM_SFT=$PPO_START_FROM_SFT PPO_REF_FIXED_TO_SFT=$PPO_REF_FIXED_TO_SFT GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE ACTOR_LR=$ACTOR_LR CRITIC_LR=$CRITIC_LR KL_LOSS_COEF=$KL_LOSS_COEF SLIME_CUSTOM_RM_TRUNCATION_PENALTY=$SLIME_CUSTOM_RM_TRUNCATION_PENALTY SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC=$SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC REWARD_EVAL_MAX_SAMPLES=$REWARD_EVAL_MAX_SAMPLES REWARD_EVAL_SHUFFLE_SEED=$REWARD_EVAL_SHUFFLE_SEED REWARD_UPDATE_EPOCHS=$REWARD_UPDATE_EPOCHS REWARD_UPDATE_BATCH_SIZE=$REWARD_UPDATE_BATCH_SIZE REWARD_ONLINE_PREF_WEIGHT=$REWARD_ONLINE_PREF_WEIGHT MIN_FREE_DISK_GB_PPO=$MIN_FREE_DISK_GB_PPO MIN_FREE_DISK_GB_EXPORT=$MIN_FREE_DISK_GB_EXPORT REWARD_EXTERNAL_EVAL_PATH=$REWARD_EXTERNAL_EVAL_PATH"

write_codex_hard_rules() {
  mkdir -p "$(dirname "$CODEX_RULES_PATH")"
  cat > "$CODEX_RULES_PATH" <<'EOF'
# CODEX HARD RULES

## HARD RULE 1: FORBIDDEN TO MAKE TARGETED PATCHES

- Forbidden: patching around a single character, a single sample type, a single prompt pattern, or a single evaluation artifact.
- Required: every fix must be mechanism-level, distribution-wide, and generalizable.
- If a proposal mainly exists to make one observed failure disappear, reject it and redesign the fix.
EOF
}

write_codex_hard_rules

ensure_synth_sft_full_data() {
  if [ -f "$SFT_SYNTH_FULL_DATA_PATH" ]; then
    return 0
  fi
  if [ ! -f "$SFT_SYNTH_FULL_DATA_PATH" ]; then
    echo "ERROR: missing SFT data at $SFT_SYNTH_FULL_DATA_PATH" >&2
    echo "ERROR: set SFT_DATA_PATH or place uf-sft.jsonl under $ULTRAFEEDBACK_DIR before running the pipeline." >&2
    exit 1
  fi
}

ensure_sft_warmup_data() {
  if [ "$SFT_DATA_PATH" != "$SFT_WARMUP_DATA_PATH" ]; then
    return 0
  fi
  ensure_synth_sft_full_data
  if [ -f "$SFT_WARMUP_DATA_PATH" ] && [ "$(awk 'END {print NR}' "$SFT_WARMUP_DATA_PATH")" -eq "$SFT_WARMUP_SAMPLES" ]; then
    # Rebuild if full data is newer than warmup subset
    if [ "$SFT_SYNTH_FULL_DATA_PATH" -nt "$SFT_WARMUP_DATA_PATH" ]; then
      echo "WARNING: $SFT_SYNTH_FULL_DATA_PATH is newer than $SFT_WARMUP_DATA_PATH, rebuilding warmup"
    else
      return 0
    fi
  fi
  echo "===== Building SFT warmup subset ====="
  python3 scripts/sample_jsonl.py \
    --input "$SFT_SYNTH_FULL_DATA_PATH" \
    --output "$SFT_WARMUP_DATA_PATH" \
    --count "$SFT_WARMUP_SAMPLES" \
    --seed "$SFT_WARMUP_SEED" \
    --report "$SFT_WARMUP_REPORT_PATH"
}

ensure_bt_warmup_data() {
  if [ ! -f "$REWARD_BT_FULL_DATA_PATH" ]; then
    echo "ERROR: missing BT pretrain full data at $REWARD_BT_FULL_DATA_PATH" >&2
    exit 1
  fi
  if [ -f "$REWARD_BT_WARMUP_DATA_PATH" ] && [ "$(awk 'END {print NR}' "$REWARD_BT_WARMUP_DATA_PATH")" -eq "$REWARD_BT_WARMUP_SAMPLES" ]; then
    if [ "$REWARD_BT_FULL_DATA_PATH" -nt "$REWARD_BT_WARMUP_DATA_PATH" ]; then
      echo "WARNING: $REWARD_BT_FULL_DATA_PATH is newer than $REWARD_BT_WARMUP_DATA_PATH, rebuilding BT warmup"
    else
      return 0
    fi
  fi
  echo "===== Building BT warmup subset ($REWARD_BT_WARMUP_SAMPLES samples) ====="
  python3 scripts/sample_jsonl.py \
    --input "$REWARD_BT_FULL_DATA_PATH" \
    --output "$REWARD_BT_WARMUP_DATA_PATH" \
    --count "$REWARD_BT_WARMUP_SAMPLES" \
    --seed "$REWARD_BT_WARMUP_SEED" \
    --report "$REWARD_BT_WARMUP_REPORT_PATH"
}

detect_sft_data_keys() {
  python3 - "$SFT_DATA_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
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
        raise SystemExit(f"Unsupported SFT row schema in {path}: keys={sorted(row.keys())}")
raise SystemExit(f"No JSONL rows found in {path}")
PY
}

reset_pipeline_state() {
  echo "===== FROM_SCRATCH=1: removing derived pipeline artifacts ====="
  if [ "$SFT_ENABLED" -eq 1 ]; then
    rm -rf "$SFT_HF_DIR" "$SFT_MEGATRON_DIR"
  else
    echo "Preserving external SFT because SFT_ENABLED=$SFT_ENABLED: $SFT_HF_DIR / $SFT_MEGATRON_DIR"
  fi
  rm -rf "$REWARD_DIR" "$MODEL_ROOT/save_dir_bootstrap" "$MODEL_ROOT/save_dir_bootstrap_critic"
  rm -rf "$MODEL_ROOT/reward_model_bt_pretrain"
  shopt -s nullglob
  for path in \
    "$MODEL_ROOT"/save_dir_r* \
    "$MODEL_ROOT"/critic_r* \
    "$MODEL_ROOT"/policy_r*_hf \
    "$EVAL_DIR"/outputs_policy_r*.jsonl \
    "$EVAL_DIR"/winrate_r*.json \
    "$EVAL_DIR"/winrate_r*.log \
    "$EVAL_WINRATE_DIR"/winrate_r*.json \
    "$EVAL_WINRATE_DIR"/winrate_r*.log \
    "$ROLLOUT_DIR"/rollout_*.pt \
    "$TENSORBOARD_ROOT"/slime-irl \
    "$TENSORBOARD_ROOT"/slime-sft \
    "$TENSORBOARD_ROOT"/slime-reward; do
    rm -rf "$path"
  done
  shopt -u nullglob
  rm -f "$SFT_BASELINE"
  echo "Preserved: base models, base torch_dist, ultrafeedback/"
}

if [ "$SFT_DATA_PATH" = "$SFT_SYNTH_FULL_DATA_PATH" ]; then
  ensure_synth_sft_full_data
fi
ensure_sft_warmup_data
if [ "$FROM_SCRATCH" -eq 1 ]; then
  START_ROUND=1
  reset_pipeline_state
fi

# Always nuke tensorboard_log at pipeline start so each run gets a clean TB tree.
rm -rf "$TENSORBOARD_ROOT" 2>/dev/null || true

check_disk_space() {
  local stage=$1
  local path=${2:-$MODEL_ROOT}
  local min_free_gb=${3:-$MIN_FREE_DISK_GB}
  mkdir -p "$path"
  local avail_gb
  avail_gb=$(df -Pk "$path" | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if [ -z "$avail_gb" ]; then
    echo "WARNING: failed to determine free disk space before $stage (path=$path)"
    return 0
  fi
  echo "Disk check [$stage]: available=${avail_gb}GB required>=${min_free_gb}GB path=$path"
  if [ "$avail_gb" -lt "$min_free_gb" ]; then
    echo "ERROR: low disk space before $stage: available=${avail_gb}GB required>=${min_free_gb}GB path=$path"
    echo "ERROR: delete old checkpoints or keep KEEP_ALL_ROUND_CHECKPOINTS=0."
    exit 1
  fi
}

SKIP_SFT=0
if [ "$SFT_ENABLED" -ne 1 ]; then
  SKIP_SFT=1
  echo "SFT_ENABLED=0: skipping SFT training (requires existing checkpoints at $SFT_HF_DIR and $SFT_MEGATRON_DIR)"
elif [ -d "$SFT_HF_DIR" ] && [ -f "$SFT_HF_DIR/config.json" ] && [ -f "$SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt" ]; then
  SKIP_SFT=1
fi

RESUME_PARTIAL_ROUND1=${RESUME_PARTIAL_ROUND1:-0}

SKIP_SFT_BASELINE=0
if [ -f "$SFT_BASELINE" ] && [ "$(awk 'END {print NR}' "$SFT_BASELINE")" -ge "$EXPECTED_EVAL_LINES" ]; then
  # Skip only if the baseline matches the current prompt format and is newer than the current SFT HF export.
  FIRST_PROMPT=$(python3 -c "import json; print(json.loads(open('$SFT_BASELINE').readline()).get('prompt','')[:20])" 2>/dev/null || echo "")
  BASELINE_IS_FRESH=$(python3 - <<PY
from pathlib import Path
baseline = Path("$SFT_BASELINE")
sft_dir = Path("$SFT_HF_DIR")
latest_sft_mtime = max((p.stat().st_mtime for p in sft_dir.rglob('*') if p.is_file()), default=0.0)
print(1 if baseline.exists() and baseline.stat().st_mtime >= latest_sft_mtime else 0)
PY
)
  if echo "$FIRST_PROMPT" | grep -q "^Human:"; then
    echo "WARNING: SFT baseline contains old HH-RLHF format, regenerating"
    SKIP_SFT_BASELINE=0
  elif [ "$BASELINE_IS_FRESH" -ne 1 ]; then
    echo "WARNING: SFT baseline is older than current SFT HF export, regenerating"
    SKIP_SFT_BASELINE=0
  else
    SKIP_SFT_BASELINE=1
  fi
fi

# ============ Clean ALL artifacts ============
# Always clean old 1.7B artifacts to save disk space
rm -rf $MODEL_ROOT/qwen3-1.7b-base_torch_dist
rm -rf $MODEL_ROOT/sft_checkpoint_hf  # old 1.7B SFT HF

if [ "$START_ROUND" -eq 1 ]; then
  if [ "$RESUME_PARTIAL_ROUND1" -eq 1 ]; then
    echo "Resume mode: preserving partial round-1 artifacts (reward_model, rollout, tensorboard_log, eval outputs, save_dir_r*)"
    rm -rf $MODEL_ROOT/policy_r*_hf
    mkdir -p $EVAL_DIR
  else
    if [ "$SKIP_SFT" -ne 1 ]; then
      rm -rf $SFT_MEGATRON_DIR
      rm -rf $SFT_HF_DIR
    fi
    rm -rf $REWARD_DIR
    rm -rf $MODEL_ROOT/save_dir_r*
    rm -rf $MODEL_ROOT/critic_r*
    rm -rf $MODEL_ROOT/save_dir_bootstrap $MODEL_ROOT/save_dir_bootstrap_critic
    rm -rf $MODEL_ROOT/policy_r*_hf
    rm -rf $ROLLOUT_DIR
    rm -rf $TENSORBOARD_ROOT
    # Clean per-round policy outputs but preserve SFT baseline
    find $EVAL_DIR -maxdepth 1 -name "outputs_policy_r*.jsonl" -delete 2>/dev/null || true
    find $EVAL_DIR -maxdepth 1 -name "eval_math_round_*.json" -delete 2>/dev/null || true
    find $EVAL_DIR -maxdepth 1 -name "winrate_r*.json" -delete 2>/dev/null || true
    find $EVAL_DIR -maxdepth 1 -name "winrate_r*.log" -delete 2>/dev/null || true
    mkdir -p $EVAL_DIR
  fi
else
  echo "Resume mode: preserving reward_model, save_dir_r*, rollout, tensorboard_log, and eval outputs"
  rm -rf $MODEL_ROOT/policy_r*_hf
  mkdir -p $EVAL_DIR
fi

# ============ Phase -1: Convert base model HF → torch_dist ============
echo "===== Converting base HF → torch_dist ====="
if [ ! -d "$BASE_HF_DIR" ]; then
  echo "ERROR: missing base HF checkpoint at $BASE_HF_DIR" >&2
  exit 1
fi
if [ ! -f "$BASE_TORCH_DIST_DIR/release/mp_rank_00_000/model_optim_rng.pt" ] && \
   [ ! -d "$BASE_TORCH_DIST_DIR/release" ]; then
  bash scripts/run-convert-8b-job.sh
  # Kill any leftover processes from conversion before SFT
  pkill -9 python || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray || true
  sleep 3
else
  echo "Skipping conversion: $BASE_TORCH_DIST_DIR already exists"
fi

# ============ Phase 0: SFT ============
echo "===== Phase 0: SFT ====="

if [ "$SKIP_SFT" -eq 1 ]; then
  echo "Skipping SFT: found $SFT_HF_DIR/config.json and $SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt"
else
  if [ ! -f "$SFT_DATA_PATH" ]; then
    echo "ERROR: missing cleaned SFT data at $SFT_DATA_PATH"
    exit 1
  fi
  mapfile -t SFT_KEYS < <(detect_sft_data_keys)
  SFT_INPUT_KEY=${SFT_KEYS[0]}
  SFT_LABEL_KEY=${SFT_KEYS[1]}
  MODEL_SH=$MODEL_SH \
  HF_CKPT=$BASE_HF_DIR \
  ACTOR_CKPT=$BASE_TORCH_DIST_DIR \
  SAVE_DIR=$SFT_MEGATRON_DIR \
  SFT_DATA=$SFT_DATA_PATH \
  SFT_INPUT_KEY=$SFT_INPUT_KEY \
  SFT_LABEL_KEY=$SFT_LABEL_KEY \
  SFT_NUM_EPOCHS=$SFT_NUM_EPOCHS \
  TENSORBOARD_DIR=$TENSORBOARD_ROOT/slime-sft/${PIPELINE_RUN_TAG:-sft} \
  bash scripts/run-sft-prod.sh
fi

# ============ Weight conversion ============
echo "===== Weight Conversion ====="
if [ "$SKIP_SFT" -eq 1 ]; then
  echo "Skipping weight conversion: found $SFT_HF_DIR/config.json"
else
  LATEST_ITER=$(cat $SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt)
  ITER_DIR=$SFT_MEGATRON_DIR/iter_$(printf "%07d" $LATEST_ITER)
  if [ -f "$ITER_DIR/common.pt" ]; then CKPT_DIR=$ITER_DIR
  elif [ -f "$ITER_DIR/mp_rank_00/common.pt" ]; then CKPT_DIR=$ITER_DIR/mp_rank_00
  else echo "ERROR: common.pt not found"; exit 1; fi

  python3 tools/convert_torch_dist_to_hf.py \
    ${MODEL_NAME_FOR_EXPORT:+--model-name $MODEL_NAME_FOR_EXPORT} \
    --input-dir $CKPT_DIR \
    --output-dir $SFT_HF_DIR \
    --origin-hf-dir $BASE_HF_DIR \
    --force
fi

# Helper: generate outputs via SGLang (fast multi-GPU inference)
generate_with_sglang() {
  local MODEL_PATH=$1
  local OUTPUT=$2
  local PORT=${SGLANG_PORT:-30010}  # avoid conflict with PPO's SGLang port 15000
  local -a sglang_extra_args=()
  if [ -n "${SGLANG_EXTRA_ARGS:-}" ]; then
    read -r -a sglang_extra_args <<< "$SGLANG_EXTRA_ARGS"
  fi
  echo "Starting SGLang server on port $PORT for $MODEL_PATH"
  CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-0,1,2,3} python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --tp ${SGLANG_TP:-4} \
    --host 127.0.0.1 \
    --trust-remote-code \
    "${sglang_extra_args[@]}" &
  SGLANG_PID=$!
  # Wait for server ready
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
      echo "SGLang ready after ${i}s"
      break
    fi
    sleep 2
  done
  python3 scripts/eval_generate_sglang.py \
    --model-path $MODEL_PATH \
    --sglang-url http://127.0.0.1:${PORT} \
    --prompt-data $TEST_DATA \
    --output $OUTPUT \
    --temperature $EVAL_TEMPERATURE \
    --apply-chat-template \
    --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --max-new-tokens 1024 \
    --concurrency 256
  kill $SGLANG_PID 2>/dev/null || true
  wait $SGLANG_PID 2>/dev/null || true
}

rollout_health_gate() {
  local stage=$1
  local rollout_end=$2
  local window=$3
  python3 - "$stage" "$ROLLOUT_DIR" "$rollout_end" "$window" "$ROLLOUT_HEALTH_MAX_TRUNCATED" <<'PY'
from pathlib import Path
import sys

from slime.local_rm.data import summarize_rollout_samples

stage = sys.argv[1]
rollout_dir = Path(sys.argv[2])
rollout_end = int(sys.argv[3])
window = int(sys.argv[4])
max_truncated = float(sys.argv[5])
start = max(0, rollout_end - window + 1)

summary = {
    "total": 0,
    "empty": 0,
    "eos_only": 0,
    "truncated": 0,
    "response_chars": 0,
    "non_printing_chars": 0,
    "non_printing_samples": 0,
}
missing = []
for rollout_id in range(start, rollout_end + 1):
    path = rollout_dir / f"rollout_{rollout_id}.pt"
    if not path.exists():
        missing.append(str(path))
        continue
    stats = summarize_rollout_samples(str(path))
    for key in summary:
        summary[key] += stats.get(key, 0)

if missing:
    print(f"{stage} rollout health missing files: {missing[:3]}")
    raise SystemExit(1)

total = summary["total"]
if total <= 0:
    print(f"{stage} rollout health found no samples")
    raise SystemExit(1)

truncated_ratio = summary["truncated"] / total
empty_ratio = summary["empty"] / total
eos_only_ratio = summary["eos_only"] / total
non_printing_sample_ratio = summary["non_printing_samples"] / total
non_printing_char_ratio = summary["non_printing_chars"] / max(summary["response_chars"], 1)

print(
    f"{stage} rollout health: total={total} truncated_ratio={truncated_ratio:.4f} "
    f"empty_ratio={empty_ratio:.4f} eos_only_ratio={eos_only_ratio:.4f} "
    f"non_printing_sample_ratio={non_printing_sample_ratio:.4f} "
    f"non_printing_char_ratio={non_printing_char_ratio:.6f}"
)

issues = []
if truncated_ratio > max_truncated:
    issues.append(f"truncated_ratio={truncated_ratio:.4f} > {max_truncated:.4f}")
if issues:
    print(f"{stage} rollout gate failed: {'; '.join(issues)}")
    raise SystemExit(1)
PY
}

eval_output_hygiene_gate() {
  local round=$1
  local output_path=$2
  python3 - "$round" "$output_path" <<'PY'
import json
import sys

from slime.utils.text_hygiene import count_non_printing_chars

round_id = sys.argv[1]
path = sys.argv[2]
total = 0
empty = 0
non_printing_rows = 0
non_printing_chars = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("response", "")
        total += 1
        if not str(text).strip():
            empty += 1
        count = count_non_printing_chars(text)
        non_printing_chars += count
        if count > 0:
            non_printing_rows += 1

print(
    f"Round {round_id} eval hygiene: total={total} empty={empty} "
    f"non_printing_rows={non_printing_rows} non_printing_chars={non_printing_chars}"
)
if total <= 0 or non_printing_rows > 0:
    raise SystemExit(1)
PY
}

run_blind_winrate_eval() {
  local outputs_a=$1
  local outputs_b=$2
  local output_json=$3
  local label=$4
  local min_winrate=$5
  mkdir -p "$EVAL_WINRATE_DIR"
  local run_eval=1
  if [ -f "$output_json" ] && [ "$output_json" -nt "$outputs_a" ] && [ "$output_json" -nt "$outputs_b" ]; then
    run_eval=0
  fi
  if [ "$run_eval" -eq 1 ]; then
    python3 scripts/eval_winrate.py \
      --outputs-a "$outputs_a" \
      --outputs-b "$outputs_b" \
      --output "$output_json" \
      --mode blind \
      --api-key "$WINRATE_GATE_API_KEY" \
      --base-url "$WINRATE_GATE_BASE_URL" \
      --model "$WINRATE_GATE_MODEL" \
      --fallback-model "$WINRATE_GATE_FALLBACK_MODEL" \
      --fallback-api-key "$WINRATE_GATE_FALLBACK_API_KEY" \
      --fallback-base-url "$WINRATE_GATE_FALLBACK_BASE_URL" \
      --concurrency "$WINRATE_GATE_CONCURRENCY" \
      --max-samples "$WINRATE_GATE_MAX_SAMPLES"
  else
    echo "Skipping blind winrate eval for $label: found current $output_json"
  fi
  python3 - "$output_json" "$label" "$min_winrate" <<'PY'
import json
import sys

path, label, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
data = json.load(open(path, encoding="utf-8"))
winrate = float(data["winrate_a"])
print(
    f"{label}: winrate_a={winrate:.4f} a_wins={data['a_wins']} "
    f"b_wins={data['b_wins']} ties={data['ties']} total={data['total']}"
)
if winrate < threshold:
    raise SystemExit(1)
PY
}

round_smoke_gates() {
  local round=$1
  local round_output=$2
  eval_output_hygiene_gate "$round" "$round_output"

  if [ "$DATA_MODE" = "math" ]; then
    if ! python3 scripts/math_accuracy_gate.py \
      --outputs "$round_output" \
      --eval-prompts "$TEST_DATA" \
      --output "$EVAL_DIR/eval_math_round_${round}.json" \
      --round "$round" \
      --num-workers 16; then
      echo "WARNING: round $round math accuracy gate failed (non-blocking)" >&2
    fi
  fi

  if [ "$WINRATE_GATE_ENABLED" -ne 1 ] || [ "$round" -gt "$WINRATE_GATE_MAX_ROUND" ]; then
    return 0
  fi

  run_blind_winrate_eval \
    "$round_output" \
    "$SFT_BASELINE" \
    "$EVAL_WINRATE_DIR/winrate_r${round}_vs_sft.json" \
    "Round $round blind winrate vs SFT" \
    "$WINRATE_GATE_MIN"

  if [ "$round" -ge 2 ]; then
    local prev_round_output=$EVAL_DIR/outputs_policy_r$((round - 1)).jsonl
    if [ -f "$prev_round_output" ]; then
      run_blind_winrate_eval \
        "$round_output" \
        "$prev_round_output" \
        "$EVAL_WINRATE_DIR/winrate_r${round}_vs_r$((round - 1)).json" \
        "Round $round blind winrate vs Round $((round - 1))" \
        "$INTERROUND_WINRATE_GATE_MIN"
    fi
  fi
}

round_train_complete() {
  local round=$1
  local round_save_dir=$MODEL_ROOT/save_dir_r${round}
  local round_reward_eval=$REWARD_DIR/reward_eval_round_${round}.json
  local round_reward_snapshot
  round_reward_snapshot=$(round_reward_model_snapshot "$round")
  # reward_eval_round_${N}.json is the inline holdout report. Emptying
  # REWARD_EXTERNAL_EVAL_PATH only disables the separate external eval.
  [ -f "$round_save_dir/latest_checkpointed_iteration.txt" ] && \
    [ -f "$round_reward_eval" ] && \
    [ -f "$round_reward_snapshot/config.json" ] && \
    { [ -f "$round_reward_snapshot/pytorch_model.bin" ] || [ -f "$round_reward_snapshot/pytorch_model.bin.index.json" ]; }
}

round_export_complete() {
  local round=$1
  round_output_complete "$round" && round_external_eval_complete "$round"
}

round_output_complete() {
  local round=$1
  local round_output=$EVAL_DIR/outputs_policy_r${round}.jsonl
  [ -f "$round_output" ] && [ "$(awk 'END {print NR}' "$round_output")" -ge "$EXPECTED_EVAL_LINES" ]
}

reward_external_eval_data_ready() {
  [ -f "$REWARD_EXTERNAL_EVAL_PATH" ] || return 1
  python3 - "$REWARD_EXTERNAL_EVAL_PATH" "$EXPECTED_REWARD_EVAL_LINES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
count = 0
with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row.get("text", "")).strip():
            raise SystemExit(1)
        if not str(row.get("chosen", "")).strip():
            raise SystemExit(1)
        if "rejected" not in row:
            raise SystemExit(1)
        count += 1
raise SystemExit(0 if count == expected else 1)
PY
}

round_external_eval_path() {
  local round=$1
  echo "$REWARD_DIR/reward_eval_external_round_${round}.json"
}

round_reward_model_snapshot() {
  local round=$1
  echo "$REWARD_DIR/step_round${round}"
}

round_external_eval_complete() {
  local round=$1
  [ -n "$REWARD_EXTERNAL_EVAL_PATH" ] || return 0
  local eval_json
  eval_json=$(round_external_eval_path "$round")
  [ -f "$eval_json" ] || return 1
  reward_external_eval_data_ready || return 1
  python3 - "$eval_json" "$REWARD_EXTERNAL_EVAL_PATH" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
positive_path = Path(sys.argv[2]).resolve()

with report_path.open(encoding="utf-8") as f:
    data = json.load(f)


def norm(value: str | None) -> str:
    return str(Path(value).resolve()) if value else ""


ok = (
    data.get("eval_source") == "external"
    and isinstance(data.get("matched_acc"), (int, float))
    and data.get("total", 0) > 0
    and norm(data.get("positive_path")) == str(positive_path)
    and norm(data.get("target_path")) == str(positive_path)
)
raise SystemExit(0 if ok else 1)
PY
}

run_external_reward_eval() {
  local round=$1
  if [ -z "$REWARD_EXTERNAL_EVAL_PATH" ]; then
    echo "Skipping external reward eval for round $round: REWARD_EXTERNAL_EVAL_PATH is empty"
    return 0
  fi
  local eval_json
  eval_json=$(round_external_eval_path "$round")
  local args_json=$REWARD_DIR/reward_eval_external_round_${round}.args.json
  local model_path
  model_path=$(round_reward_model_snapshot "$round")

  if ! reward_external_eval_data_ready; then
    echo "ERROR: missing external reward eval positives at $REWARD_EXTERNAL_EVAL_PATH" >&2
    echo "ERROR: expected a chosen/rejected UltraFeedback eval file at the configured path." >&2
    exit 1
  fi
  if [ ! -d "$model_path" ]; then
    echo "ERROR: missing reward model snapshot for round $round at $model_path" >&2
    exit 1
  fi

  mkdir -p "$REWARD_DIR"
  rm -f "$eval_json" "$args_json"
  python3 - "$args_json" "$SFT_HF_DIR" "$REWARD_DIR" "$model_path" "$REWARD_EXTERNAL_EVAL_PATH" "$eval_json" "$REWARD_EXTERNAL_EVAL_BATCH_SIZE" "$REWARD_MODEL_INIT" <<'PY'
import json
import sys
from pathlib import Path

args_path = Path(sys.argv[1])
reward_model_init = sys.argv[8] or None
cfg = {
    "hf_checkpoint": sys.argv[2],
    "reward_model_dir": sys.argv[3],
    "reward_model_path": sys.argv[4],
    "reward_model_init": reward_model_init,
    "apply_chat_template": True,
    "apply_chat_template_kwargs": {"enable_thinking": False},
    "reward_eval_path": sys.argv[5],
    "reward_eval_prompt_key": "text",
    "reward_eval_chosen_key": "chosen",
    "reward_eval_rejected_key": "rejected",
    "reward_eval_target_path": "",
    "reward_eval_batch_size": int(sys.argv[7]),
    "reward_eval_output_path": sys.argv[6],
    "reward_eval_source": "external",
}
args_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

  echo "===== Round $round: External reward eval on $REWARD_EXTERNAL_EVAL_PATH ====="
  python3 -m slime.local_rm.reward_eval_cli --args-json "$args_json" --rollout-id "$round"
  rm -f "$args_json"

  if ! round_external_eval_complete "$round"; then
    echo "ERROR: round $round external reward eval failed or is incomplete: $eval_json" >&2
    exit 1
  fi
}

prune_stale_checkpoints() {
  local current_round=$1
  if ! round_export_complete "$current_round"; then
    echo "WARNING: current round $current_round export or external reward eval is incomplete; keeping existing checkpoints"
    return 0
  fi

  if [ "$KEEP_ALL_ROUND_CHECKPOINTS" -ne 1 ] && [ "$KEEP_FINAL_POLICY_PER_ROUND" -ne 1 ]; then
    local keep_dir=$MODEL_ROOT/save_dir_r${current_round}
    for save_dir in "$MODEL_ROOT"/save_dir_r*; do
      if [ ! -d "$save_dir" ]; then
        continue
      fi
      if [ "$save_dir" = "$keep_dir" ]; then
        continue
      fi
      echo "Pruning obsolete checkpoint $save_dir after successful export for round $current_round"
      rm -rf "$save_dir"
    done
  fi

  if [ "$KEEP_ALL_ROUND_CHECKPOINTS" -ne 1 ]; then
    local keep_critic_dir=$MODEL_ROOT/critic_r${current_round}
    for critic_dir in "$MODEL_ROOT"/critic_r*; do
      if [ ! -d "$critic_dir" ]; then
        continue
      fi
      if [ "$critic_dir" = "$keep_critic_dir" ]; then
        continue
      fi
      echo "Pruning obsolete critic checkpoint $critic_dir after successful export for round $current_round"
      rm -rf "$critic_dir"
    done
  fi

  if [ "$KEEP_ALL_ROUND_REWARD_SNAPSHOTS" -ne 1 ] && [ "$KEEP_FINAL_REWARD_PER_ROUND" -ne 1 ]; then
    local keep_reward_dir
    keep_reward_dir=$(round_reward_model_snapshot "$current_round")
    for reward_step_dir in "$REWARD_DIR"/step_round*; do
      if [ ! -d "$reward_step_dir" ]; then
        continue
      fi
      if [ "$reward_step_dir" = "$keep_reward_dir" ]; then
        continue
      fi
      echo "Pruning reward snapshot $reward_step_dir after successful external eval for round $current_round"
      rm -rf "$reward_step_dir"
    done
  fi

  # SFT Megatron checkpoint is a long-lived asset needed for Round 1 ref/resume.
  # Do not prune it here; only delete at the very end of the pipeline if desired.
}

startup_prune_legacy_reward_snapshots() {
  if [ "$KEEP_ALL_ROUND_REWARD_SNAPSHOTS" -eq 1 ] || [ "$KEEP_FINAL_REWARD_PER_ROUND" -eq 1 ]; then
    return 0
  fi

  local keep_reward_dir=""
  local keep_round=-1
  local reward_step_dir
  local reward_step_name
  local round_suffix
  local round_num
  local snapshot_count=0

  for reward_step_dir in "$REWARD_DIR"/step_round*; do
    if [ ! -d "$reward_step_dir" ]; then
      continue
    fi
    reward_step_name=$(basename "$reward_step_dir")
    round_suffix=${reward_step_name#step_round}
    if ! [[ "$round_suffix" =~ ^[0-9]+$ ]]; then
      echo "startup-prune: ignoring non-round reward snapshot directory $reward_step_dir"
      continue
    fi
    round_num=$((10#$round_suffix))
    snapshot_count=$((snapshot_count + 1))
    if [ "$round_num" -gt "$keep_round" ]; then
      keep_round=$round_num
      keep_reward_dir=$reward_step_dir
    fi
  done

  if [ "$snapshot_count" -le 1 ]; then
    return 0
  fi

  for reward_step_dir in "$REWARD_DIR"/step_round*; do
    if [ ! -d "$reward_step_dir" ]; then
      continue
    fi
    reward_step_name=$(basename "$reward_step_dir")
    round_suffix=${reward_step_name#step_round}
    if ! [[ "$round_suffix" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if [ "$reward_step_dir" = "$keep_reward_dir" ]; then
      continue
    fi
    echo "startup-prune: removing legacy reward snapshot $reward_step_dir (keeping $keep_reward_dir)"
    rm -rf "$reward_step_dir"
  done
}

cleanup_transient_training_artifacts() {
  local bootstrap_dir=$MODEL_ROOT/save_dir_bootstrap

  if [ -d "$bootstrap_dir" ]; then
    echo "Removing transient bootstrap checkpoint $bootstrap_dir"
    rm -rf "$bootstrap_dir"
  fi

  for critic_dir in "$MODEL_ROOT"/critic_r*; do
    if [ ! -d "$critic_dir" ]; then
      continue
    fi
    echo "Removing transient critic checkpoint $critic_dir"
    rm -rf "$critic_dir"
  done

  for policy_hf_dir in "$MODEL_ROOT"/policy_r*_hf; do
    if [ ! -d "$policy_hf_dir" ]; then
      continue
    fi
    echo "Removing transient HF export $policy_hf_dir"
    rm -rf "$policy_hf_dir"
  done
}

# ============ Generate SFT baseline outputs (for offline winrate comparison) ============
echo "===== Generating SFT baseline outputs ====="
mkdir -p $EVAL_DIR
if [ "$SKIP_SFT_BASELINE" -eq 1 ]; then
  echo "Skipping SFT baseline generation: found $SFT_BASELINE"
else
  generate_with_sglang $SFT_HF_DIR $SFT_BASELINE
fi

# ============ Phase 0.5: BT reward pretrain (auto-run if checkpoint missing) ============
SKIP_BT_PRETRAIN=0
if [ "$BT_PRETRAIN_ENABLED" -ne 1 ]; then
  SKIP_BT_PRETRAIN=1
  echo "BT_PRETRAIN_ENABLED=0: skipping BT pretrain and BT seed (reward model will start from scratch)"
else
  rm -rf "$BT_PRETRAIN_DIR"
  echo "BT_PRETRAIN_ENABLED=1: forcing fresh BT pretrain (cleared $BT_PRETRAIN_DIR)"
fi

if [ "$SKIP_BT_PRETRAIN" -eq 1 ]; then
  if [ "$BT_PRETRAIN_ENABLED" -eq 1 ]; then
    echo "Skipping BT reward pretrain: found $BT_PRETRAIN_DIR"
  fi
else
  echo "===== Phase 0.5: BT reward pretrain ($REWARD_BT_WARMUP_SAMPLES samples) ====="
  ensure_bt_warmup_data

  pkill -9 sglang || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3

  REWARD_BT_BASE_MODEL=$BT_PRETRAIN_BASE_MODEL \
  REWARD_BT_ROOT=$MODEL_ROOT/reward_model_bt_pretrain \
  REWARD_BT_OUTPUT_DIR=$BT_PRETRAIN_DIR \
  REWARD_BT_PREF_PATH=$REWARD_BT_WARMUP_DATA_PATH \
  REWARD_BT_EVAL_PATH=$REWARD_BT_EVAL_PATH \
  REWARD_BT_EPOCHS=$REWARD_BT_EPOCHS \
  REWARD_BT_BATCH_SIZE=$REWARD_BT_BATCH_SIZE \
  REWARD_BT_EVAL_BATCH_SIZE=$REWARD_BT_EVAL_BATCH_SIZE \
  REWARD_BT_LR=$REWARD_BT_LR \
  REWARD_BT_HOLDOUT_RATIO=$REWARD_BT_HOLDOUT_RATIO \
  REWARD_BT_EVAL_INTERVAL=$REWARD_BT_EVAL_INTERVAL \
  REWARD_BT_C_COEF_INIT=$REWARD_BT_C_COEF_INIT \
  bash scripts/pretrain-reward-bt.sh

  echo "===== Killing BT pretrain processes ====="
  pkill -9 python || true
  sleep 3

  if [ ! -f "$BT_PRETRAIN_DIR/config.json" ]; then
    echo "ERROR: BT pretrain finished but checkpoint missing at $BT_PRETRAIN_DIR" >&2
    exit 1
  fi
fi

# ============ Seed reward_model/latest from BT pretrain ============
NEED_BT_SEED=0
if [ "$BT_PRETRAIN_ENABLED" -eq 1 ] && [ -f "$BT_PRETRAIN_DIR/config.json" ]; then
  if [ "$START_ROUND" -le 1 ] && [ ! -d "$REWARD_DIR/step_round1" ]; then
    NEED_BT_SEED=1
  fi
fi

if [ "$NEED_BT_SEED" -eq 1 ]; then
  echo "===== Seeding reward_model/latest from BT pretrain: $BT_PRETRAIN_DIR ====="
  mkdir -p "$REWARD_DIR"
  rm -rf "$REWARD_DIR/latest"
  cp -r "$BT_PRETRAIN_DIR" "$REWARD_DIR/latest"
elif [ "$START_ROUND" -gt 1 ]; then
  PREV_ROUND_REWARD=$REWARD_DIR/step_round$((START_ROUND - 1))
  if [ ! -d "$REWARD_DIR/latest" ] && [ -d "$PREV_ROUND_REWARD" ]; then
    echo "===== Resume: restoring reward_model/latest from $PREV_ROUND_REWARD ====="
    cp -r "$PREV_ROUND_REWARD" "$REWARD_DIR/latest"
  elif [ -d "$REWARD_DIR/latest" ]; then
    echo "Resume: preserving existing reward_model/latest for round $START_ROUND"
  else
    echo "WARNING: resume from round $START_ROUND but no reward_model/latest and no $PREV_ROUND_REWARD; reward will cold-start"
  fi
else
  echo "No BT seed needed (BT disabled or round-1 reward already trained)"
fi

if [ "$BT_PRETRAIN_ENABLED" -eq 1 ] && [ -f "$BT_PRETRAIN_DIR/config.json" ] && [ -z "${REWARD_MODEL_INIT:-}" ]; then
  REWARD_MODEL_INIT=$BT_PRETRAIN_BASE_MODEL
  echo "Auto-set REWARD_MODEL_INIT=$REWARD_MODEL_INIT (BT backbone persists across resume)"
fi

startup_prune_legacy_reward_snapshots

# Round 0: Bootstrap rollout — collect SFT-aligned rollouts so R1 can train RM
echo "===== Round 0: Bootstrap rollout (SFT-aligned rollouts for R1 reward update) ====="
BOOTSTRAP_SAVE_DIR=$MODEL_ROOT/save_dir_bootstrap
if [ ! -d "$ROLLOUT_DIR" ] || [ "$(ls $ROLLOUT_DIR/rollout_*.pt 2>/dev/null | wc -l)" -lt "$BOOTSTRAP_NUM_ROLLOUT" ]; then
  rm -rf $BOOTSTRAP_SAVE_DIR 2>/dev/null || true

	  MODEL_SH=$MODEL_SH \
	  HF_CKPT=$SFT_HF_DIR \
	  REF_CKPT=$SFT_MEGATRON_DIR \
	  SAVE_DIR=$BOOTSTRAP_SAVE_DIR \
	  PROMPT_DATA=$PROMPT_DATA \
	  DEMO_DATA=$DEMO_DATA \
	  PROMPT_INPUT_KEY=text \
	  PROMPT_LABEL_KEY=$PROMPT_LABEL_KEY_PASS \
	  NUM_ROLLOUT=$BOOTSTRAP_NUM_ROLLOUT \
	  N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT \
	  ADVANTAGE_ESTIMATOR=$ADVANTAGE_ESTIMATOR \
	  NORMALIZE_ADVANTAGES=$NORMALIZE_ADVANTAGES \
	  ALIGN_ROLLOUT_WITH_SFT=1 \
	  DEBUG_ROLLOUT_ONLY=1 \
	  REWARD_MODEL_DIR=$REWARD_DIR \
	  REWARD_MODEL_INIT=$REWARD_MODEL_INIT \
	  ROLLOUT_DEBUG_DIR=$ROLLOUT_DIR \
	  ROLLOUT_TEMPERATURE=$BOOTSTRAP_ROLLOUT_TEMPERATURE \
	  ROLLOUT_MAX_RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN \
	  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE \
	  ACTOR_LR=$ACTOR_LR \
	  CRITIC_LR=$CRITIC_LR \
	  KL_LOSS_COEF=$KL_LOSS_COEF \
	  SLIME_CUSTOM_RM_TRUNCATION_PENALTY=$SLIME_CUSTOM_RM_TRUNCATION_PENALTY \
	  SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC=$SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC \
	  SLIME_RUBRIC_ENABLED=$SLIME_RUBRIC_ENABLED \
	  SLIME_RUBRIC_PLUGINS=$SLIME_RUBRIC_PLUGINS \
	  SLIME_RUBRIC_W_IRL=$SLIME_RUBRIC_W_IRL \
	  SLIME_RUBRIC_W_FORMAT=$SLIME_RUBRIC_W_FORMAT \
	  SLIME_RUBRIC_W_ANSWER=$SLIME_RUBRIC_W_ANSWER \
	  TENSORBOARD_DIR=$TENSORBOARD_ROOT/slime-irl/bootstrap \
	  TB_EXP_NAME=bootstrap \
	  bash scripts/run-irl-prod.sh

  echo "===== Killing bootstrap processes ====="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  rm -rf $BOOTSTRAP_SAVE_DIR
else
  echo "Rollout data already exists, skipping bootstrap"
fi

rollout_health_gate "bootstrap" $(( BOOTSTRAP_NUM_ROLLOUT - 1 )) "$BOOTSTRAP_NUM_ROLLOUT"

# ============ Rounds 1-7 ============
PREV_SAVE_DIR=""
if [ "$START_ROUND" -gt 1 ]; then
  PREV_SAVE_DIR=$MODEL_ROOT/save_dir_r$((START_ROUND - 1))
  if [ ! -d "$PREV_SAVE_DIR" ]; then
    echo "ERROR: START_ROUND=$START_ROUND requires previous checkpoint at $PREV_SAVE_DIR"
    exit 1
  fi
  echo "Resuming from round $START_ROUND with PREV_SAVE_DIR=$PREV_SAVE_DIR"
fi
for ROUND in $(seq "$START_ROUND" $NUM_ROUNDS); do
  ROUND_POLICY_HF=$MODEL_ROOT/policy_r${ROUND}_hf
  ROUND_OUTPUT=$EVAL_DIR/outputs_policy_r${ROUND}.jsonl
  ROUND_SAVE_DIR=$MODEL_ROOT/save_dir_r${ROUND}
  ROUND_REWARD_EVAL=$REWARD_DIR/reward_eval_round_${ROUND}.json
  ROUND_TRAIN_READY=0
  ROUND_EXPORT_READY=0
  if round_train_complete "$ROUND"; then
    ROUND_TRAIN_READY=1
  fi
  if round_export_complete "$ROUND"; then
    ROUND_EXPORT_READY=1
  fi
  if [ "$ROUND_TRAIN_READY" -eq 1 ]; then
    echo "Skipping Round $ROUND training: found checkpoint + reward eval artifacts"
    if ! round_output_complete "$ROUND"; then
      echo "Round $ROUND test-set output missing; exporting now"
      check_disk_space "round${ROUND}-pre-export" "$MODEL_ROOT" "$MIN_FREE_DISK_GB_EXPORT"
      EVAL_TEMPERATURE=$EVAL_TEMPERATURE KEEP_POLICY_HF=0 bash scripts/export-policy-round.sh "$ROUND"
    fi
    if ! round_output_complete "$ROUND"; then
      echo "ERROR: round $ROUND output generation failed or is incomplete: $ROUND_OUTPUT"
      exit 1
    fi
    round_smoke_gates "$ROUND" "$ROUND_OUTPUT"
    if ! round_external_eval_complete "$ROUND"; then
      echo "Round $ROUND external reward eval missing; running now"
      run_external_reward_eval "$ROUND"
    fi
    rm -rf "$ROUND_POLICY_HF"
    if [ -d "$ROUND_SAVE_DIR" ]; then
      PREV_SAVE_DIR=$ROUND_SAVE_DIR
    fi
    prune_stale_checkpoints "$ROUND"
    continue
  fi

  # Reward update FIRST (every round including R1)
  if [ -f "$ROUND_REWARD_EVAL" ]; then
    echo "Skipping Round $ROUND reward update: found $ROUND_REWARD_EVAL"
  else
    if [ "$ROUND" -eq 1 ]; then
      RU_ROLLOUT_END=$(( BOOTSTRAP_NUM_ROLLOUT - 1 ))
      RU_NUM_ROLLOUT=$BOOTSTRAP_NUM_ROLLOUT
    else
      RU_ROLLOUT_END=$(( NUM_ROLLOUT_PER_ROUND - 1 ))
      RU_NUM_ROLLOUT=$NUM_ROLLOUT_PER_ROUND
    fi
    echo "===== Round $ROUND/$NUM_ROUNDS: Reward Update ====="

    SLIME=$SLIME \
    HF_CKPT=$SFT_HF_DIR \
    ROUND_ID=$ROUND \
    ROLLOUT_END=$RU_ROLLOUT_END \
    NUM_ROLLOUT_PER_ROUND=$RU_NUM_ROLLOUT \
	    REWARD_DIR=$REWARD_DIR \
	    ROLLOUT_DIR=$ROLLOUT_DIR \
	    REWARD_TRAIN_DATA_PATH=$REWARD_TRAIN_DATA_PATH \
	    REWARD_EVAL_PATH=$REWARD_EXTERNAL_EVAL_PATH \
    REWARD_EVAL_REJECTED_KEY=$REWARD_EVAL_REJECTED_KEY_PASS \
    REWARD_EVAL_TARGET_PATH= \
    REWARD_EVAL_MAX_SAMPLES=$REWARD_EVAL_MAX_SAMPLES \
    REWARD_EVAL_SHUFFLE_SEED=$REWARD_EVAL_SHUFFLE_SEED \
    REWARD_EVAL_BATCH_SIZE=$REWARD_EXTERNAL_EVAL_BATCH_SIZE \
    REWARD_UPDATE_EPOCHS=$REWARD_UPDATE_EPOCHS \
    REWARD_UPDATE_BATCH_SIZE=$REWARD_UPDATE_BATCH_SIZE \
    REWARD_ONLINE_PREF_WEIGHT=$REWARD_ONLINE_PREF_WEIGHT \
    REWARD_MODEL_INIT=$REWARD_MODEL_INIT \
    bash scripts/run-reward-update.sh
  fi

  echo "===== Killing processes before PPO ====="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  echo "===== Round $ROUND/$NUM_ROUNDS: PPO ====="
  check_disk_space "round${ROUND}-pre-ppo" "$MODEL_ROOT" "$MIN_FREE_DISK_GB_PPO"

  if [ "$PPO_REF_FIXED_TO_SFT" -eq 1 ]; then
    ROUND_REF_CKPT=$SFT_MEGATRON_DIR
  elif [ "$ROUND" -eq 1 ]; then
    ROUND_REF_CKPT=$SFT_MEGATRON_DIR
  else
    if [ -z "$PREV_SAVE_DIR" ] || [ ! -d "$PREV_SAVE_DIR" ]; then
      echo "ERROR: missing previous-round checkpoint directory for round $ROUND ref model: $PREV_SAVE_DIR"
      exit 1
    fi
    ROUND_REF_CKPT=$PREV_SAVE_DIR
  fi

  ROUND_ACTOR_LOAD=""
  if [ "$PPO_START_FROM_SFT" -eq 1 ]; then
    ROUND_ACTOR_LOAD=$SFT_MEGATRON_DIR
  else
    ROUND_ACTOR_LOAD=$PREV_SAVE_DIR
  fi

  ROUND_CRITIC_SAVE=$MODEL_ROOT/critic_r${ROUND}
  ROUND_CRITIC_LOAD=""
  rm -rf "$ROUND_SAVE_DIR" "$ROUND_CRITIC_SAVE" 2>/dev/null || true
  # Critic cold start every round: prevent value function from tracking reward
  # too quickly and collapsing advantages to zero (which kills PPO gradient signal).
  # if [ "$ROUND" -gt 1 ]; then
  #   PREV_CRITIC=$MODEL_ROOT/critic_r$((ROUND - 1))
  #   if [ -d "$PREV_CRITIC" ] && [ -f "$PREV_CRITIC/latest_checkpointed_iteration.txt" ]; then
  #     ROUND_CRITIC_LOAD=$PREV_CRITIC
  #   fi
  # fi

  MODEL_SH=$MODEL_SH \
  HF_CKPT=$SFT_HF_DIR \
  REF_CKPT=$ROUND_REF_CKPT \
  SAVE_DIR=$ROUND_SAVE_DIR \
  ACTOR_LOAD=$ROUND_ACTOR_LOAD \
	  CRITIC_SAVE_DIR=$ROUND_CRITIC_SAVE \
	  CRITIC_LOAD_DIR=$ROUND_CRITIC_LOAD \
	  PROMPT_DATA=$PROMPT_DATA \
	  DEMO_DATA=$DEMO_DATA \
	  PROMPT_INPUT_KEY=text \
	  PROMPT_LABEL_KEY=$PROMPT_LABEL_KEY_PASS \
	  NUM_ROLLOUT=$NUM_ROLLOUT_PER_ROUND \
	  N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT \
	  ADVANTAGE_ESTIMATOR=$ADVANTAGE_ESTIMATOR \
	  NORMALIZE_ADVANTAGES=$NORMALIZE_ADVANTAGES \
	  ALIGN_ROLLOUT_WITH_SFT=1 \
	  REWARD_MODEL_DIR=$REWARD_DIR \
	  REWARD_MODEL_INIT=$REWARD_MODEL_INIT \
	  ROLLOUT_DEBUG_DIR=$ROLLOUT_DIR \
	  ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE \
	  ROLLOUT_MAX_RESPONSE_LEN=$ROLLOUT_MAX_RESPONSE_LEN \
	  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE \
  ACTOR_LR=$ACTOR_LR \
  CRITIC_LR=$CRITIC_LR \
	  KL_LOSS_COEF=$KL_LOSS_COEF \
	  SLIME_CUSTOM_RM_TRUNCATION_PENALTY=$SLIME_CUSTOM_RM_TRUNCATION_PENALTY \
	  SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC=$SLIME_CUSTOM_RM_TRUNCATION_THRESHOLD_FRAC \
	  SLIME_RUBRIC_ENABLED=$SLIME_RUBRIC_ENABLED \
	  SLIME_RUBRIC_PLUGINS=$SLIME_RUBRIC_PLUGINS \
	  SLIME_RUBRIC_W_IRL=$SLIME_RUBRIC_W_IRL \
	  SLIME_RUBRIC_W_FORMAT=$SLIME_RUBRIC_W_FORMAT \
	  SLIME_RUBRIC_W_ANSWER=$SLIME_RUBRIC_W_ANSWER \
	  TENSORBOARD_DIR=$TENSORBOARD_ROOT/slime-irl/round${ROUND} \
	  TB_EXP_NAME=round${ROUND} \
	  bash scripts/run-irl-prod.sh

  rollout_health_gate "round${ROUND}" $(( NUM_ROLLOUT_PER_ROUND - 1 )) "$NUM_ROLLOUT_PER_ROUND"

  echo "=== Killing PPO processes before test-set export ==="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  check_disk_space "round${ROUND}-pre-export" "$MODEL_ROOT" "$MIN_FREE_DISK_GB_EXPORT"
  echo "===== Round $ROUND: Exporting test-set outputs ====="
  EVAL_TEMPERATURE=$EVAL_TEMPERATURE KEEP_POLICY_HF=0 bash scripts/export-policy-round.sh "$ROUND"
  if ! round_output_complete "$ROUND"; then
    echo "ERROR: round $ROUND output generation failed or is incomplete: $ROUND_OUTPUT"
    exit 1
  fi
  round_smoke_gates "$ROUND" "$ROUND_OUTPUT"
  rm -rf "$ROUND_POLICY_HF"
  run_external_reward_eval "$ROUND"

  if [ -d "$ROUND_SAVE_DIR" ]; then
    PREV_SAVE_DIR=$ROUND_SAVE_DIR
  fi

  prune_stale_checkpoints "$ROUND"

  # ===== Analyze tensorboard metrics =====
  echo "===== Round $ROUND: Analyzing tensorboard ====="
  python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

base_dir = Path('${TENSORBOARD_ROOT}/slime-irl/round${ROUND}')
run_dirs = [base_dir / 'actor', base_dir / 'critic', base_dir / 'rollout', base_dir / 'reward', base_dir]

def load_series(tag):
    best = None
    best_step = -1
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get('scalars', [])
        if tag not in tags:
            continue
        vals = ea.Scalars(tag)
        if vals and vals[-1].step >= best_step:
            best = vals
            best_step = vals[-1].step
    return best

try:
    issues = []
    metrics = {}
    for tag in [
        'train/loss',
        'train/pg_clipfrac',
        'train/ppo_kl',
        'train/grad_norm',
        'train/critic-grad_norm',
        'train/critic-value_clipfrac',
        'rollout/truncated_ratio',
        'rollout/empty_rate',
        'rollout/eos_only_rate',
        'rollout/user_prefix_rate',
        'rollout/rewards',
        'rollout/raw_reward',
    ]:
        series = load_series(tag)
        if series:
            vals = [e.value for e in series]
            metrics[tag] = {'first': vals[0], 'last': vals[-1], 'max': max(vals)}
    truncated = metrics.get('rollout/truncated_ratio', {}).get('last', 0)
    empty_rate = metrics.get('rollout/empty_rate', {}).get('last', 0)
    eos_only_rate = metrics.get('rollout/eos_only_rate', {}).get('last', 0)
    user_prefix_rate = metrics.get('rollout/user_prefix_rate', {}).get('last', 0)
    kl = metrics.get('train/ppo_kl', {}).get('last', 0)
    grad = metrics.get('train/grad_norm', {}).get('max', 0)
    clipfrac = metrics.get('train/pg_clipfrac', {}).get('last', 0)
    critic_grad = metrics.get('train/critic-grad_norm', {}).get('max', 0)
    critic_clipfrac = metrics.get('train/critic-value_clipfrac', {}).get('last', 0)
    if truncated > ${ROLLOUT_HEALTH_MAX_TRUNCATED}: issues.append(f'HIGH TRUNCATION {truncated:.2f} (increase max_response_len)')
    if empty_rate > 0.1: issues.append(f'HIGH EMPTY RATE {empty_rate:.2f}')
    if eos_only_rate > 0.1: issues.append(f'HIGH EOS-ONLY RATE {eos_only_rate:.2f}')
    if user_prefix_rate > 0.05: issues.append(f'HIGH USER PREFIX RATE {user_prefix_rate:.2f}')
    if abs(kl) < 0.001: issues.append(f'KL~0 ({kl:.4f}) policy barely changed (reduce kl_coef?)')
    if kl > 0.1: issues.append(f'HIGH KL {kl:.4f} (policy drifting)')
    if grad > 10: issues.append(f'GRAD EXPLODING {grad:.2f}')
    if critic_grad > 1000: issues.append(f'CRITIC GRAD VERY HIGH {critic_grad:.2f}')
    if clipfrac > 0.3: issues.append(f'HIGH CLIP {clipfrac:.2f} (lr too high?)')
    if critic_clipfrac > 0.9: issues.append(f'CRITIC VALUE CLIP SATURATED {critic_clipfrac:.2f}')
    print(f'Round ${ROUND} TB: truncated={truncated:.2f} empty={empty_rate:.2f} eos_only={eos_only_rate:.2f} user_prefix={user_prefix_rate:.2f} kl={kl:.4f} grad_max={grad:.2f} critic_grad_max={critic_grad:.2f} clipfrac={clipfrac:.3f} critic_value_clipfrac={critic_clipfrac:.3f}')
    if issues: print('ISSUES: ' + '; '.join(issues))
    else: print('Metrics OK')
except Exception as e:
    print(f'TB analysis failed: {e}')
" 2>&1

  echo "===== Round $ROUND/$NUM_ROUNDS completed ====="
done

cleanup_transient_training_artifacts

echo "===== All $NUM_ROUNDS rounds completed ====="
echo "Eval outputs:"
ls -la $EVAL_DIR/outputs_*.jsonl
echo "External reward eval outputs:"
ls -la $REWARD_DIR/reward_eval_external_round_*.json
echo "Per-round test-set outputs are generated automatically under eval/."
echo "Per-round final policy checkpoints are retained under $MODEL_ROOT/save_dir_r*."
echo "Per-round final reward snapshots are retained under $REWARD_DIR/step_round*."
echo "Transient critic checkpoints and temporary HF exports are removed automatically."
echo "Run scripts/run-winrate-offline.sh on a networked machine to compute winrate."
