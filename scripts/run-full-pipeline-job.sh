#!/bin/bash
# Full Pipeline v6: SFT → 7 rounds of (Reward Update → PPO → Test-set export)
# Each round exports test-set outputs immediately, then prunes transient artifacts.
set -ex

SLIME=/mnt/shared-storage-gpfs2/wangqianyi2/slime
cd $SLIME
ULTRAFEEDBACK_DIR=${ULTRAFEEDBACK_DIR:-$SLIME/ultrafeedback}
REWARD_DIR=${REWARD_DIR:-$SLIME/models/reward_model}
SFT_SYNTH_FULL_DATA_PATH=${SFT_SYNTH_FULL_DATA_PATH:-${SFT_SYNTH_DATA_PATH:-$ULTRAFEEDBACK_DIR/uf-sft-clean-synth.jsonl}}
SFT_SYNTH_REPORT_PATH=${SFT_SYNTH_REPORT_PATH:-$ULTRAFEEDBACK_DIR/uf-sft-clean-synth.report.json}
SFT_WARMUP_SAMPLES=${SFT_WARMUP_SAMPLES:-10000}
SFT_WARMUP_SEED=${SFT_WARMUP_SEED:-42}
SFT_WARMUP_DATA_PATH=${SFT_WARMUP_DATA_PATH:-$ULTRAFEEDBACK_DIR/uf-sft-clean-synth-warmup${SFT_WARMUP_SAMPLES}.jsonl}
SFT_WARMUP_REPORT_PATH=${SFT_WARMUP_REPORT_PATH:-$ULTRAFEEDBACK_DIR/uf-sft-clean-synth-warmup${SFT_WARMUP_SAMPLES}.report.json}

KEEP_ALL_ROUND_CHECKPOINTS=${KEEP_ALL_ROUND_CHECKPOINTS:-0}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.2}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-}
MIN_FREE_DISK_GB_PPO=${MIN_FREE_DISK_GB_PPO:-${MIN_FREE_DISK_GB:-60}}
MIN_FREE_DISK_GB_EXPORT=${MIN_FREE_DISK_GB_EXPORT:-${MIN_FREE_DISK_GB:-30}}
START_ROUND=${START_ROUND:-1}
FROM_SCRATCH=${FROM_SCRATCH:-0}
SFT_NUM_EPOCHS=${SFT_NUM_EPOCHS:-1}
NUM_ROUNDS=7
NUM_ROLLOUT_PER_ROUND=150
BOOTSTRAP_NUM_ROLLOUT=${BOOTSTRAP_NUM_ROLLOUT:-$NUM_ROLLOUT_PER_ROUND}
BOOTSTRAP_ROLLOUT_TEMPERATURE=${BOOTSTRAP_ROLLOUT_TEMPERATURE:-0.7}
TEST_DATA=$ULTRAFEEDBACK_DIR/uf-test.jsonl
REWARD_EXTERNAL_EVAL_PATH=${REWARD_EXTERNAL_EVAL_PATH:-$ULTRAFEEDBACK_DIR/uf-test-synth-prefs.jsonl}
REWARD_EXTERNAL_EVAL_BATCH_SIZE=${REWARD_EXTERNAL_EVAL_BATCH_SIZE:-32}
EXPECTED_EVAL_LINES=2000
if [ -f "$TEST_DATA" ]; then
  ACTUAL_EVAL_LINES=$(awk 'END {print NR}' "$TEST_DATA")
  if [ "$ACTUAL_EVAL_LINES" -gt 0 ]; then
    EXPECTED_EVAL_LINES=$ACTUAL_EVAL_LINES
  fi
fi

SFT_HF_DIR=$SLIME/models/sft_checkpoint_8b_hf
SFT_MEGATRON_DIR=$SLIME/models/sft_checkpoint
SFT_BASELINE=$SLIME/eval/outputs_sft_baseline.jsonl
SFT_DATA_PATH=${SFT_DATA_PATH:-}
if [ -z "$SFT_DATA_PATH" ]; then
  if [ "$SFT_WARMUP_SAMPLES" -gt 0 ]; then
    SFT_DATA_PATH=$SFT_WARMUP_DATA_PATH
  elif [ -f "$SFT_SYNTH_FULL_DATA_PATH" ]; then
    SFT_DATA_PATH=$SFT_SYNTH_FULL_DATA_PATH
  else
    SFT_DATA_PATH=$ULTRAFEEDBACK_DIR/uf-sft-clean.jsonl
  fi
fi

echo "Pipeline config: START_ROUND=$START_ROUND NUM_ROUNDS=$NUM_ROUNDS FROM_SCRATCH=$FROM_SCRATCH KEEP_ALL_ROUND_CHECKPOINTS=$KEEP_ALL_ROUND_CHECKPOINTS SFT_DATA_PATH=$SFT_DATA_PATH SFT_WARMUP_SAMPLES=$SFT_WARMUP_SAMPLES SFT_NUM_EPOCHS=$SFT_NUM_EPOCHS EVAL_TEMPERATURE=$EVAL_TEMPERATURE BOOTSTRAP_NUM_ROLLOUT=$BOOTSTRAP_NUM_ROLLOUT BOOTSTRAP_ROLLOUT_TEMPERATURE=$BOOTSTRAP_ROLLOUT_TEMPERATURE MIN_FREE_DISK_GB_PPO=$MIN_FREE_DISK_GB_PPO MIN_FREE_DISK_GB_EXPORT=$MIN_FREE_DISK_GB_EXPORT REWARD_EXTERNAL_EVAL_PATH=$REWARD_EXTERNAL_EVAL_PATH"

ensure_synth_sft_full_data() {
  if [ -f "$SFT_SYNTH_FULL_DATA_PATH" ]; then
    return 0
  fi
  local synth_input=$ULTRAFEEDBACK_DIR/uf-train-synth-chosen.jsonl
  if [ ! -f "$synth_input" ]; then
    echo "ERROR: missing synthetic train chosen data at $synth_input" >&2
    echo "ERROR: generate uf-train-synth-chosen.jsonl before running from synthetic SFT data." >&2
    exit 1
  fi
  echo "===== Building synthetic SFT clean dataset ====="
  python3 scripts/build_sft_clean_from_synth.py \
    --input "$synth_input" \
    --output "$SFT_SYNTH_FULL_DATA_PATH" \
    --report "$SFT_SYNTH_REPORT_PATH"
}

ensure_sft_warmup_data() {
  if [ "$SFT_DATA_PATH" != "$SFT_WARMUP_DATA_PATH" ]; then
    return 0
  fi
  ensure_synth_sft_full_data
  if [ -f "$SFT_WARMUP_DATA_PATH" ] && [ "$(awk 'END {print NR}' "$SFT_WARMUP_DATA_PATH")" -eq "$SFT_WARMUP_SAMPLES" ]; then
    return 0
  fi
  echo "===== Building synthetic SFT warmup subset ====="
  python3 scripts/sample_jsonl.py \
    --input "$SFT_SYNTH_FULL_DATA_PATH" \
    --output "$SFT_WARMUP_DATA_PATH" \
    --count "$SFT_WARMUP_SAMPLES" \
    --seed "$SFT_WARMUP_SEED" \
    --report "$SFT_WARMUP_REPORT_PATH"
}

reset_pipeline_state() {
  echo "===== FROM_SCRATCH=1: removing derived pipeline artifacts ====="
  rm -rf "$SFT_HF_DIR" "$SFT_MEGATRON_DIR" "$REWARD_DIR" "$SLIME/models/save_dir_bootstrap"
  shopt -s nullglob
  for path in \
    "$SLIME"/models/save_dir_r* \
    "$SLIME"/eval/outputs_policy_r*.jsonl \
    "$SLIME"/rollout/rollout_*.pt \
    "$SLIME"/tensorboard_log/slime-irl \
    "$SLIME"/tensorboard_log/slime-sft; do
    rm -rf "$path"
  done
  shopt -u nullglob
  rm -f "$SFT_BASELINE"
}

if [ "$SFT_DATA_PATH" = "$SFT_SYNTH_FULL_DATA_PATH" ]; then
  ensure_synth_sft_full_data
fi
ensure_sft_warmup_data
if [ "$FROM_SCRATCH" -eq 1 ]; then
  START_ROUND=1
  reset_pipeline_state
fi

check_disk_space() {
  local stage=$1
  local path=${2:-$SLIME/models}
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
if [ -d "$SFT_HF_DIR" ] && [ -f "$SFT_HF_DIR/config.json" ] && [ -f "$SFT_MEGATRON_DIR/latest_checkpointed_iteration.txt" ]; then
  SKIP_SFT=1
fi

RESUME_PARTIAL_ROUND1=0
if [ "$FROM_SCRATCH" -ne 1 ] && [ "$START_ROUND" -eq 1 ] && [ "$SKIP_SFT" -eq 1 ]; then
  if [ -d "$SLIME/rollout" ] || [ -d "$REWARD_DIR" ] || [ -d "$SLIME/models/save_dir_r1" ]; then
    RESUME_PARTIAL_ROUND1=1
  fi
fi

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
rm -rf $SLIME/models/qwen3-1.7b-base_torch_dist
rm -rf $SLIME/models/sft_checkpoint_hf  # old 1.7B SFT HF

if [ "$START_ROUND" -eq 1 ]; then
  if [ "$RESUME_PARTIAL_ROUND1" -eq 1 ]; then
    echo "Resume mode: preserving partial round-1 artifacts (reward_model, rollout, tensorboard_log, eval outputs, save_dir_r*)"
    rm -rf $SLIME/models/policy_r*_hf
    mkdir -p $SLIME/eval
  else
    if [ "$SKIP_SFT" -ne 1 ]; then
      rm -rf $SFT_MEGATRON_DIR
      rm -rf $SFT_HF_DIR
    fi
    rm -rf $SLIME/models/reward_model
    rm -rf $SLIME/models/save_dir_r*
    rm -rf $SLIME/models/policy_r*_hf
    rm -rf $SLIME/rollout
    rm -rf $SLIME/tensorboard_log
    # Clean per-round policy outputs but preserve SFT baseline
    find $SLIME/eval -maxdepth 1 -name "outputs_policy_r*.jsonl" -delete 2>/dev/null || true
    find $SLIME/eval -maxdepth 1 -name "winrate_r*.json" -delete 2>/dev/null || true
    find $SLIME/eval -maxdepth 1 -name "winrate_r*.log" -delete 2>/dev/null || true
    mkdir -p $SLIME/eval
  fi
else
  echo "Resume mode: preserving reward_model, save_dir_r*, rollout, tensorboard_log, and eval outputs"
  rm -rf $SLIME/models/policy_r*_hf
  mkdir -p $SLIME/eval
fi

# ============ Phase -1: Convert base model HF → torch_dist ============
echo "===== Converting Qwen3-8B HF → torch_dist ====="
if [ ! -f "$SLIME/models/qwen3-8b-base_torch_dist/release/mp_rank_00_000/model_optim_rng.pt" ] && \
   [ ! -d "$SLIME/models/qwen3-8b-base_torch_dist/release" ]; then
  bash scripts/run-convert-8b-job.sh
  # Kill any leftover processes from conversion before SFT
  pkill -9 python || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray || true
  sleep 3
else
  echo "Skipping conversion: qwen3-8b-base_torch_dist already exists"
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
  MODEL_SH=scripts/models/qwen3-8B.sh \
  HF_CKPT=$SLIME/models/qwen3-8b-base \
  ACTOR_CKPT=$SLIME/models/qwen3-8b-base_torch_dist \
  SAVE_DIR=$SFT_MEGATRON_DIR \
  SFT_DATA=$SFT_DATA_PATH \
  SFT_NUM_EPOCHS=$SFT_NUM_EPOCHS \
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
    --input-dir $CKPT_DIR \
    --output-dir $SFT_HF_DIR \
    --origin-hf-dir $SLIME/models/qwen3-8b-base \
    --force
fi

# Helper: generate outputs via SGLang (fast multi-GPU inference)
generate_with_sglang() {
  local MODEL_PATH=$1
  local OUTPUT=$2
  local PORT=30010  # avoid conflict with PPO's SGLang port 15000
  echo "Starting SGLang server on port $PORT for $MODEL_PATH"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --tp 4 \
    --host 127.0.0.1 \
    --trust-remote-code &
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
    --max-new-tokens 512 \
    --concurrency 256
  kill $SGLANG_PID 2>/dev/null || true
  wait $SGLANG_PID 2>/dev/null || true
}

round_train_complete() {
  local round=$1
  local round_save_dir=$SLIME/models/save_dir_r${round}
  local round_reward_eval=$SLIME/models/reward_model/reward_eval_round_${round}.json
  [ -f "$round_save_dir/latest_checkpointed_iteration.txt" ] && [ -f "$round_reward_eval" ]
}

round_export_complete() {
  local round=$1
  round_output_complete "$round" && round_external_eval_complete "$round"
}

round_output_complete() {
  local round=$1
  local round_output=$SLIME/eval/outputs_policy_r${round}.jsonl
  [ -f "$round_output" ] && [ "$(awk 'END {print NR}' "$round_output")" -ge "$EXPECTED_EVAL_LINES" ]
}

reward_external_eval_data_ready() {
  [ -f "$REWARD_EXTERNAL_EVAL_PATH" ] || return 1
  python3 - "$REWARD_EXTERNAL_EVAL_PATH" "$EXPECTED_EVAL_LINES" <<'PY'
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
  local eval_json
  eval_json=$(round_external_eval_path "$round")
  local args_json=$REWARD_DIR/reward_eval_external_round_${round}.args.json
  local model_path
  model_path=$(round_reward_model_snapshot "$round")

  if ! reward_external_eval_data_ready; then
    echo "ERROR: missing external reward eval positives at $REWARD_EXTERNAL_EVAL_PATH" >&2
    echo "ERROR: generate uf-test synthetic prefs data before running the pipeline." >&2
    exit 1
  fi
  if [ ! -d "$model_path" ]; then
    echo "ERROR: missing reward model snapshot for round $round at $model_path" >&2
    exit 1
  fi

  mkdir -p "$REWARD_DIR"
  rm -f "$eval_json" "$args_json"
  python3 - "$args_json" "$SFT_HF_DIR" "$REWARD_DIR" "$model_path" "$REWARD_EXTERNAL_EVAL_PATH" "$eval_json" "$REWARD_EXTERNAL_EVAL_BATCH_SIZE" <<'PY'
import json
import sys
from pathlib import Path

args_path = Path(sys.argv[1])
cfg = {
    "hf_checkpoint": sys.argv[2],
    "reward_model_dir": sys.argv[3],
    "reward_model_path": sys.argv[4],
    "reward_model_init": None,
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

  echo "===== Round $round: External reward eval on uf-test ====="
  python3 -m slime.local_rm.reward_eval_cli --args-json "$args_json" --rollout-id "$round"
  rm -f "$args_json"

  if ! round_external_eval_complete "$round"; then
    echo "ERROR: round $round external reward eval failed or is incomplete: $eval_json" >&2
    exit 1
  fi
}

prune_stale_checkpoints() {
  local current_round=$1
  if [ "$KEEP_ALL_ROUND_CHECKPOINTS" -eq 1 ]; then
    return 0
  fi

  if ! round_export_complete "$current_round"; then
    echo "WARNING: current round $current_round export or external reward eval is incomplete; keeping existing checkpoints"
    return 0
  fi

  local keep_dir=$SLIME/models/save_dir_r${current_round}
  for save_dir in "$SLIME"/models/save_dir_r*; do
    if [ ! -d "$save_dir" ]; then
      continue
    fi
    if [ "$save_dir" = "$keep_dir" ]; then
      continue
    fi
    echo "Pruning obsolete checkpoint $save_dir after successful export for round $current_round"
    rm -rf "$save_dir"
  done

  for reward_step_dir in "$REWARD_DIR"/step_round*; do
    if [ ! -d "$reward_step_dir" ]; then
      continue
    fi
    echo "Pruning reward snapshot $reward_step_dir after successful external eval for round $current_round"
    rm -rf "$reward_step_dir"
  done

  if [ "$current_round" -ge 1 ] && [ -d "$SFT_MEGATRON_DIR" ]; then
    echo "Pruning SFT Megatron checkpoint $SFT_MEGATRON_DIR after round $current_round export completed"
    rm -rf "$SFT_MEGATRON_DIR"
  fi
}

# ============ Generate SFT baseline outputs (for offline winrate comparison) ============
echo "===== Generating SFT baseline outputs ====="
mkdir -p $SLIME/eval
if [ "$SKIP_SFT_BASELINE" -eq 1 ]; then
  echo "Skipping SFT baseline generation: found $SFT_BASELINE"
else
  generate_with_sglang $SFT_HF_DIR $SFT_BASELINE
fi

# Round 0: Bootstrap rollout — collect SFT-aligned rollouts so R1 can train RM
echo "===== Round 0: Bootstrap rollout (SFT-aligned rollouts for R1 reward update) ====="
BOOTSTRAP_SAVE_DIR=$SLIME/models/save_dir_bootstrap
if [ ! -d "$SLIME/rollout" ] || [ "$(ls $SLIME/rollout/rollout_*.pt 2>/dev/null | wc -l)" -lt "$BOOTSTRAP_NUM_ROLLOUT" ]; then
  rm -rf $BOOTSTRAP_SAVE_DIR /tmp/critic_ckpt 2>/dev/null || true

  MODEL_SH=scripts/models/qwen3-8B.sh \
  HF_CKPT=$SFT_HF_DIR \
  REF_CKPT=$SFT_MEGATRON_DIR \
  SAVE_DIR=$BOOTSTRAP_SAVE_DIR \
  PROMPT_DATA=$ULTRAFEEDBACK_DIR/uf-train.jsonl \
  DEMO_DATA=$ULTRAFEEDBACK_DIR/uf-train.jsonl \
  NUM_ROLLOUT=$BOOTSTRAP_NUM_ROLLOUT \
  ALIGN_ROLLOUT_WITH_SFT=1 \
  DEBUG_ROLLOUT_ONLY=1 \
  ROLLOUT_TEMPERATURE=$BOOTSTRAP_ROLLOUT_TEMPERATURE \
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

# ============ Rounds 1-7 ============
PREV_SAVE_DIR=""
if [ "$START_ROUND" -gt 1 ]; then
  PREV_SAVE_DIR=$SLIME/models/save_dir_r$((START_ROUND - 1))
  if [ ! -d "$PREV_SAVE_DIR" ]; then
    echo "ERROR: START_ROUND=$START_ROUND requires previous checkpoint at $PREV_SAVE_DIR"
    exit 1
  fi
  echo "Resuming from round $START_ROUND with PREV_SAVE_DIR=$PREV_SAVE_DIR"
fi
for ROUND in $(seq "$START_ROUND" $NUM_ROUNDS); do
  ROUND_POLICY_HF=$SLIME/models/policy_r${ROUND}_hf
  ROUND_OUTPUT=$SLIME/eval/outputs_policy_r${ROUND}.jsonl
  ROUND_SAVE_DIR=$SLIME/models/save_dir_r${ROUND}
  ROUND_REWARD_EVAL=$SLIME/models/reward_model/reward_eval_round_${ROUND}.json
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
      check_disk_space "round${ROUND}-pre-export" "$SLIME/models" "$MIN_FREE_DISK_GB_EXPORT"
      KEEP_POLICY_HF=0 bash scripts/export-policy-round.sh "$ROUND"
    fi
    if ! round_output_complete "$ROUND"; then
      echo "ERROR: round $ROUND output generation failed or is incomplete: $ROUND_OUTPUT"
      exit 1
    fi
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
    REWARD_EVAL_PATH=$REWARD_EXTERNAL_EVAL_PATH \
    REWARD_EVAL_REJECTED_KEY=rejected \
    REWARD_EVAL_TARGET_PATH= \
    REWARD_EVAL_BATCH_SIZE=$REWARD_EXTERNAL_EVAL_BATCH_SIZE \
    bash scripts/run-reward-update.sh
  fi

  echo "===== Killing processes before PPO ====="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  echo "===== Round $ROUND/$NUM_ROUNDS: PPO ====="
  check_disk_space "round${ROUND}-pre-ppo" "$SLIME/models" "$MIN_FREE_DISK_GB_PPO"

  rm -rf $ROUND_SAVE_DIR /tmp/critic_ckpt 2>/dev/null || true

  if [ "$ROUND" -eq 1 ]; then
    ROUND_REF_CKPT=$SFT_MEGATRON_DIR
  else
    if [ -z "$PREV_SAVE_DIR" ] || [ ! -d "$PREV_SAVE_DIR" ]; then
      echo "ERROR: missing previous-round checkpoint directory for round $ROUND ref model: $PREV_SAVE_DIR"
      exit 1
    fi
    ROUND_REF_CKPT=$PREV_SAVE_DIR
  fi

  MODEL_SH=scripts/models/qwen3-8B.sh \
  HF_CKPT=$SFT_HF_DIR \
  REF_CKPT=$ROUND_REF_CKPT \
  SAVE_DIR=$ROUND_SAVE_DIR \
  ACTOR_LOAD=$PREV_SAVE_DIR \
  PROMPT_DATA=$ULTRAFEEDBACK_DIR/uf-train.jsonl \
  DEMO_DATA=$ULTRAFEEDBACK_DIR/uf-train.jsonl \
  NUM_ROLLOUT=$NUM_ROLLOUT_PER_ROUND \
  ALIGN_ROLLOUT_WITH_SFT=1 \
  ROLLOUT_TEMPERATURE=0.2 \
  TB_EXP_NAME=round${ROUND} \
  bash scripts/run-irl-prod.sh

  echo "=== Killing PPO processes before test-set export ==="
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 5

  check_disk_space "round${ROUND}-pre-export" "$SLIME/models" "$MIN_FREE_DISK_GB_EXPORT"
  echo "===== Round $ROUND: Exporting test-set outputs ====="
  KEEP_POLICY_HF=0 bash scripts/export-policy-round.sh "$ROUND"
  if ! round_output_complete "$ROUND"; then
    echo "ERROR: round $ROUND output generation failed or is incomplete: $ROUND_OUTPUT"
    exit 1
  fi
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

base_dir = Path('/mnt/shared-storage-gpfs2/wangqianyi2/slime/tensorboard_log/slime-irl/round${ROUND}')
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
    if truncated > 0.5: issues.append(f'HIGH TRUNCATION {truncated:.2f} (increase max_response_len)')
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

echo "===== All $NUM_ROUNDS rounds completed ====="
echo "Eval outputs:"
ls -la $SLIME/eval/outputs_*.jsonl
echo "External reward eval outputs:"
ls -la $REWARD_DIR/reward_eval_external_round_*.json
echo "Per-round test-set outputs are generated automatically under eval/."
echo "Only the newest training checkpoint is retained by default."
echo "Keep all round checkpoints only when needed: KEEP_ALL_ROUND_CHECKPOINTS=1"
echo "Run scripts/run-winrate-offline.sh on a networked machine to compute winrate."
