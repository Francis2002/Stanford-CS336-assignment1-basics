#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Everything runs from the project root, so that "python -m" puts the repo on sys.path
# and the package imports resolve without the project having to be installed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x ./.venv_cluster/bin/python ]]; then
  DEFAULT_PYTHON=./.venv_cluster/bin/python
elif [[ -x ./.venv/bin/python ]]; then
  DEFAULT_PYTHON=./.venv/bin/python
else
  DEFAULT_PYTHON=python3
fi

PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
TRAIN_MODULE="${TRAIN_MODULE:-cs336_basics.training_together}"

TS_TRAIN_TEXT="${TS_TRAIN_TEXT:-./data/TinyStoriesV2-GPT4-train.txt}"
TS_VAL_TEXT="${TS_VAL_TEXT:-./data/TinyStoriesV2-GPT4-valid.txt}"

OWT_TRAIN_TEXT="${OWT_TRAIN_TEXT:-./data/owt_train.txt}"
OWT_VAL_TEXT="${OWT_VAL_TEXT:-./data/owt_valid.txt}"

# Absolute, because the training script resolves relative output dirs against cs336_basics/
# and the sweep tooling below has to read the same directories back
SAVE_DIR="${SAVE_DIR:-$SCRIPT_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

# torch.compile artefacts live here so that cold-cache and warm-cache compile cost
# can be measured instead of leaking into whichever run happens to go second
INDUCTOR_CACHE_ROOT="${INDUCTOR_CACHE_ROOT:-$SCRIPT_DIR/.inductor_cache}"

# -----------------------------------------------------------------------------
# Stage selection
# -----------------------------------------------------------------------------
ALL_STAGES=(
  feasibility
  systems
  ts_lr
  ts_batch
  ts_ctx
  ts_final
  owt_lr
  owt_final
  owt_lr_wide
  owt_recommended
  ts_extended
  owt_extended
)

STAGES="${STAGES:-all}"

if [[ "$STAGES" == "list" ]]; then
  printf '%s\n' "${ALL_STAGES[@]}"
  exit 0
fi

stage_enabled () {
  [[ "$STAGES" == "all" ]] && return 0
  local wanted
  for wanted in $STAGES; do
    [[ "$wanted" == "$1" ]] && return 0
  done
  return 1
}

for wanted in $STAGES; do
  [[ "$wanted" == "all" ]] && continue
  if ! printf '%s\n' "${ALL_STAGES[@]}" | grep -qx "$wanted"; then
    echo "Unknown stage: $wanted" >&2
    echo "Available: ${ALL_STAGES[*]}" >&2
    exit 1
  fi
done

echo "[stages] requested: $STAGES"

# -----------------------------------------------------------------------------
# Shared experiment settings
# -----------------------------------------------------------------------------
SEED="${SEED:-42}"

D_MODEL=512
D_FF=1344
NUM_HEADS=16
NUM_LAYERS=4

SHORT_TOKENS=40960000
FULL_TOKENS=327680000
EXTENDED_TOKENS=655360000

BASE_CONTEXT=256
BASE_BATCH=32

EVAL_EVERY=250
EVAL_BATCHES=20

# Every experiment after the systems section runs on the optimized stack, because
# that stack is what makes the recommended model sizes fit and finish in reasonable time
OPTIMIZED=(--mixed_precision --compile --async_io)

mkdir -p "$SAVE_DIR" "$LOG_DIR" "$INDUCTOR_CACHE_ROOT"

steps_for_tokens () {
  local tokens="$1"
  local batch="$2"
  local context="$3"
  echo $(( tokens / batch / context ))
}

warmup_for_steps () {
  local steps="$1"
  local warmup=$(( steps / 20 ))
  if (( warmup < 1 )); then warmup=1; fi
  echo "$warmup"
}

float_mul () {
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
print(float(sys.argv[1]) * float(sys.argv[2]))
PY
}

run_model () {
  local run_name="$1"; shift

  "$PYTHON" -m "$TRAIN_MODULE" \
    --run_name "$run_name" \
    --seed "$SEED" \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --max_grad_norm 1.0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --epsilon 1e-8 \
    --weight_decay 0.05 \
    --rope_theta 10000 \
    --eval_every "$EVAL_EVERY" \
    --eval_batches "$EVAL_BATCHES" \
    --print_every "$EVAL_EVERY" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --keep_last 2 \
    "$@"
}

run_with_cache () {
  local cache_name="$1"
  local run_name="$2"
  shift 2
  (
    export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_ROOT/$cache_name"
    export TRITON_CACHE_DIR="$INDUCTOR_CACHE_ROOT/$cache_name/triton"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    run_model "$run_name" "$@"
  )
}

run_with_cold_cache () {
  local cache_name="$1"
  rm -rf "$INDUCTOR_CACHE_ROOT/$cache_name"
  run_with_cache "$@"
}

peak_vram () {
  "$PYTHON" - "$SAVE_DIR" "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / sys.argv[2] / "summary.json"
if not path.exists():
    print("peak_vram=unknown")
else:
    s = json.loads(path.read_text())
    vram = s.get("peak_vram_mb")
    tps = s.get("steady_state_tokens_per_sec")
    print(f"peak_vram={vram:.0f}MB tokens_per_sec={tps:,.0f}" if vram else "peak_vram=unknown")
PY
}

# A probe is allowed to fail: an out-of-memory error is the measurement, not an error
probe () {
  local run_name="$1"; shift
  local status=0
  set +e
  run_model "$run_name" "$@" >"$LOG_DIR/${run_name}.log" 2>&1
  status=$?
  set -e

  if (( status == 0 )); then
    echo "[probe] FITS $run_name $(peak_vram "$run_name")"
  elif grep -qiE "out of memory|CUDA out of memory" "$LOG_DIR/${run_name}.log"; then
    echo "[probe] OOM  $run_name (log: $LOG_DIR/${run_name}.log)"
  else
    echo "[probe] FAIL $run_name exit=$status (log: $LOG_DIR/${run_name}.log)"
  fi
}

best_lr_from_prefix () {
  local prefix="$1"
  "$PYTHON" - "$SAVE_DIR" "$prefix" <<'PY'
import json, pathlib, sys, math

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
candidates = []

for run_dir in sorted(root.glob(prefix + "*")):
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        continue

    vals = []
    with metrics_path.open() as f:
        for line in f:
            row = json.loads(line)
            v = row.get("val_loss")
            if v is not None and math.isfinite(v):
                vals.append(v)

    if not vals:
        continue

    cfg = json.loads(config_path.read_text())
    candidates.append((vals[-1], cfg["learning_rate_base"], run_dir.name))

if not candidates:
    raise SystemExit(f"No completed runs found for prefix: {prefix}")

candidates.sort()
loss, lr, name = candidates[0]
print(lr)
PY
}

resolve_ts_lr () {
  if [[ -z "${TS_BEST_LR:-}" ]]; then
    TS_BEST_LR="$(best_lr_from_prefix "ts_lr_")"
    echo "[lr] TinyStories short-run best LR: $TS_BEST_LR"
  fi
}

# Prefer the wide bracket at the batch we actually train at; fall back to the
# transfer test from TinyStories if the bracket has not been run yet
resolve_owt_lr () {
  if [[ -z "${OWT_BEST_LR:-}" ]]; then
    if compgen -G "$SAVE_DIR/owt_wide_*" >/dev/null; then
      OWT_BEST_LR="$(best_lr_from_prefix "owt_wide_")"
      echo "[lr] OpenWebText best LR (wide bracket at batch $BASE_BATCH): $OWT_BEST_LR"
    else
      OWT_BEST_LR="$(best_lr_from_prefix "owt_lr_")"
      echo "[lr] OpenWebText best LR (transfer test): $OWT_BEST_LR"
    fi
  fi
}

# =============================================================================
# STAGE feasibility
# Does the recommended configuration actually fit in FP32 on this GPU?
# 200 steps per probe, FP32 against the optimized stack, at the sizes we want to train.
# An OOM here is a result, not a failure: it is the constraint the next stage works around.
# =============================================================================

if stage_enabled feasibility; then
  echo "=== stage: feasibility ==="

  PROBE_STEPS=200
  PROBE_COMMON=(
    --num_steps "$PROBE_STEPS"
    --learning_rate_base 1e-3
    --learning_rate_min 1e-5
    --steps_for_warmup 20
    --steps_for_cosine "$PROBE_STEPS"
    --eval_every "$PROBE_STEPS"
    --print_every 100
    --save_every 1000000
  )

  TS_PROBE=(
    --train_path_text "$TS_TRAIN_TEXT"
    --val_path_text "$TS_VAL_TEXT"
    --vocab_size 10000
    --context_length "$BASE_CONTEXT"
  )

  OWT_PROBE=(
    --train_path_text "$OWT_TRAIN_TEXT"
    --val_path_text "$OWT_VAL_TEXT"
    --vocab_size 32000
    --context_length "$BASE_CONTEXT"
  )

  # Batches are swept past the point of failure on purpose, to locate the memory
  # wall for each precision rather than only confirming the sizes we train at
  for BATCH in ${TS_PROBE_BATCHES:-32 64 96 128}; do
    probe "probe_ts_bs${BATCH}_fp32" "${TS_PROBE[@]}" "${PROBE_COMMON[@]}" --batch_size "$BATCH"
    probe "probe_ts_bs${BATCH}_opt" "${TS_PROBE[@]}" "${PROBE_COMMON[@]}" --batch_size "$BATCH" "${OPTIMIZED[@]}"
  done

  for BATCH in ${OWT_PROBE_BATCHES:-16 32 48 64}; do
    probe "probe_owt_bs${BATCH}_fp32" "${OWT_PROBE[@]}" "${PROBE_COMMON[@]}" --batch_size "$BATCH"
    probe "probe_owt_bs${BATCH}_opt" "${OWT_PROBE[@]}" "${PROBE_COMMON[@]}" --batch_size "$BATCH" "${OPTIMIZED[@]}"
  done
fi

# =============================================================================
# STAGE systems
# Staged stack: eager FP32 -> AMP -> AMP+compile (cold cache) -> AMP+compile (warm
# cache) -> AMP+compile+async I/O -> async I/O with infrequent checkpoints.
#
# The cold/warm pair separates one-time compilation cost from steady-state gain.
# The async run reuses the warm cache so that its comparison against sys_03 is a
# clean A/B on I/O alone. The final run reduces checkpoint frequency to test
# whether the post-checkpoint step-time spikes come from the writer thread.
# =============================================================================

if stage_enabled systems; then
  echo "=== stage: systems ==="

  SYS_STEPS=1000
  SYS_WARMUP=$(warmup_for_steps "$SYS_STEPS")
  SYS_LR="${SYS_LR:-1e-3}"
  SYS_CACHE=sys_compile

  SYSTEM_COMMON=(
    --train_path_text "$TS_TRAIN_TEXT"
    --val_path_text "$TS_VAL_TEXT"
    --vocab_size 10000
    --context_length 256
    --batch_size 32
    --num_steps "$SYS_STEPS"
    --learning_rate_base "$SYS_LR"
    --learning_rate_min 1e-5
    --steps_for_warmup "$SYS_WARMUP"
    --steps_for_cosine "$SYS_STEPS"
    --save_every 100
  )

  run_model "sys_00_eager_fp32" "${SYSTEM_COMMON[@]}"
  run_model "sys_01_amp" "${SYSTEM_COMMON[@]}" --mixed_precision

  run_with_cold_cache "$SYS_CACHE" "sys_02_amp_compile_cold" \
    "${SYSTEM_COMMON[@]}" --mixed_precision --compile

  run_with_cache "$SYS_CACHE" "sys_03_amp_compile_warm" \
    "${SYSTEM_COMMON[@]}" --mixed_precision --compile

  run_with_cache "$SYS_CACHE" "sys_04_amp_compile_async" \
    "${SYSTEM_COMMON[@]}" --mixed_precision --compile --async_io

  run_with_cache "$SYS_CACHE" "sys_05_async_rare_ckpt" \
    "${SYSTEM_COMMON[@]}" --mixed_precision --compile --async_io --save_every 1000
fi

# =============================================================================
# STAGE ts_lr
# Fixed: 40.96M tokens, batch=32, context=256
# =============================================================================

if stage_enabled ts_lr; then
  echo "=== stage: ts_lr ==="

  TS_LR_STEPS=$(steps_for_tokens "$SHORT_TOKENS" "$BASE_BATCH" "$BASE_CONTEXT")
  TS_LR_WARMUP=$(warmup_for_steps "$TS_LR_STEPS")

  for LR in 3e-4 7e-4 1.5e-3 3e-3 1e-2; do
    SAFE_LR=$(echo "$LR" | tr '.+-' 'p__')
    run_model "ts_lr_${SAFE_LR}" \
      --train_path_text "$TS_TRAIN_TEXT" \
      --val_path_text "$TS_VAL_TEXT" \
      --vocab_size 10000 \
      --context_length "$BASE_CONTEXT" \
      --batch_size "$BASE_BATCH" \
      --num_steps "$TS_LR_STEPS" \
      --learning_rate_base "$LR" \
      --learning_rate_min 1e-5 \
      --steps_for_warmup "$TS_LR_WARMUP" \
      --steps_for_cosine "$TS_LR_STEPS" \
      --save_every "$TS_LR_STEPS" \
      "${OPTIMIZED[@]}"
  done
fi

# =============================================================================
# STAGE ts_batch
# Fixed: 40.96M tokens, context=256
# =============================================================================

if stage_enabled ts_batch; then
  echo "=== stage: ts_batch ==="
  resolve_ts_lr

  for BATCH in 16 32 64; do
    STEPS=$(steps_for_tokens "$SHORT_TOKENS" "$BATCH" "$BASE_CONTEXT")
    WARMUP=$(warmup_for_steps "$STEPS")

    run_model "ts_batch_${BATCH}" \
      --train_path_text "$TS_TRAIN_TEXT" \
      --val_path_text "$TS_VAL_TEXT" \
      --vocab_size 10000 \
      --context_length "$BASE_CONTEXT" \
      --batch_size "$BATCH" \
      --num_steps "$STEPS" \
      --learning_rate_base "$TS_BEST_LR" \
      --learning_rate_min 1e-5 \
      --steps_for_warmup "$WARMUP" \
      --steps_for_cosine "$STEPS" \
      --save_every "$STEPS" \
      "${OPTIMIZED[@]}"
  done
fi

# =============================================================================
# STAGE ts_ctx
# Fixed: 40.96M tokens AND 8192 tokens/update
# 128x64 = 256x32 = 512x16 = 8192
# =============================================================================

if stage_enabled ts_ctx; then
  echo "=== stage: ts_ctx ==="
  resolve_ts_lr

  CTX_STEPS=$(steps_for_tokens "$SHORT_TOKENS" 32 256)
  CTX_WARMUP=$(warmup_for_steps "$CTX_STEPS")

  for PAIR in "128 64" "256 32" "512 16"; do
    read -r CTX BATCH <<<"$PAIR"

    run_model "ts_ctx_${CTX}" \
      --train_path_text "$TS_TRAIN_TEXT" \
      --val_path_text "$TS_VAL_TEXT" \
      --vocab_size 10000 \
      --context_length "$CTX" \
      --batch_size "$BATCH" \
      --num_steps "$CTX_STEPS" \
      --learning_rate_base "$TS_BEST_LR" \
      --learning_rate_min 1e-5 \
      --steps_for_warmup "$CTX_WARMUP" \
      --steps_for_cosine "$CTX_STEPS" \
      --save_every "$CTX_STEPS" \
      "${OPTIMIZED[@]}"
  done
fi

# =============================================================================
# STAGE ts_final
# Recommended configuration at the assignment budget: 327.68M tokens
# =============================================================================

if stage_enabled ts_final; then
  echo "=== stage: ts_final ==="
  resolve_ts_lr

  TS_FINAL_STEPS=$(steps_for_tokens "$FULL_TOKENS" "$BASE_BATCH" "$BASE_CONTEXT")
  TS_FINAL_WARMUP=$(warmup_for_steps "$TS_FINAL_STEPS")

  run_model "ts_final" \
    --train_path_text "$TS_TRAIN_TEXT" \
    --val_path_text "$TS_VAL_TEXT" \
    --vocab_size 10000 \
    --context_length "$BASE_CONTEXT" \
    --batch_size "$BASE_BATCH" \
    --num_steps "$TS_FINAL_STEPS" \
    --learning_rate_base "$TS_BEST_LR" \
    --learning_rate_min 1e-5 \
    --steps_for_warmup "$TS_FINAL_WARMUP" \
    --steps_for_cosine "$TS_FINAL_STEPS" \
    --save_every 5000 \
    "${OPTIMIZED[@]}"
fi

# =============================================================================
# STAGE owt_lr
# Transfer test: 0.5x / 1x / 2x the TinyStories LR, fixed 40.96M tokens, batch 16
# =============================================================================

if stage_enabled owt_lr; then
  echo "=== stage: owt_lr ==="
  resolve_ts_lr

  OWT_BATCH="${OWT_BATCH:-16}"
  OWT_CONTEXT="${OWT_CONTEXT:-256}"
  OWT_SHORT_STEPS=$(steps_for_tokens "$SHORT_TOKENS" "$OWT_BATCH" "$OWT_CONTEXT")
  OWT_SHORT_WARMUP=$(warmup_for_steps "$OWT_SHORT_STEPS")

  for LABEL in half same double; do
    case "$LABEL" in
      half) LR=$(float_mul "$TS_BEST_LR" 0.5) ;;
      same) LR="$TS_BEST_LR" ;;
      double) LR=$(float_mul "$TS_BEST_LR" 2.0) ;;
    esac

    run_model "owt_lr_${LABEL}" \
      --train_path_text "$OWT_TRAIN_TEXT" \
      --val_path_text "$OWT_VAL_TEXT" \
      --vocab_size 32000 \
      --context_length "$OWT_CONTEXT" \
      --batch_size "$OWT_BATCH" \
      --num_steps "$OWT_SHORT_STEPS" \
      --learning_rate_base "$LR" \
      --learning_rate_min 1e-5 \
      --steps_for_warmup "$OWT_SHORT_WARMUP" \
      --steps_for_cosine "$OWT_SHORT_STEPS" \
      --save_every "$OWT_SHORT_STEPS" \
      "${OPTIMIZED[@]}"
  done
fi

# =============================================================================
# STAGE owt_final
# First OpenWebText run at the assignment budget, batch 16 (token-matched to
# ts_final but NOT iteration-matched: 80000 steps against 40000)
# =============================================================================

if stage_enabled owt_final; then
  echo "=== stage: owt_final ==="
  resolve_owt_lr

  OWT_BATCH="${OWT_BATCH:-16}"
  OWT_CONTEXT="${OWT_CONTEXT:-256}"
  OWT_FINAL_STEPS=$(steps_for_tokens "$FULL_TOKENS" "$OWT_BATCH" "$OWT_CONTEXT")
  OWT_FINAL_WARMUP=$(warmup_for_steps "$OWT_FINAL_STEPS")

  run_model "owt_final" \
    --train_path_text "$OWT_TRAIN_TEXT" \
    --val_path_text "$OWT_VAL_TEXT" \
    --vocab_size 32000 \
    --context_length "$OWT_CONTEXT" \
    --batch_size "$OWT_BATCH" \
    --num_steps "$OWT_FINAL_STEPS" \
    --learning_rate_base "$OWT_BEST_LR" \
    --learning_rate_min 1e-5 \
    --steps_for_warmup "$OWT_FINAL_WARMUP" \
    --steps_for_cosine "$OWT_FINAL_STEPS" \
    --save_every 5000 \
    "${OPTIMIZED[@]}"
fi

# =============================================================================
# STAGE owt_lr_wide
# The transfer test only bracketed the optimum from above, and short budgets
# favour large learning rates. This brackets from below at the batch size the
# final runs actually use. Fixed: 40.96M tokens, batch 32.
# =============================================================================

if stage_enabled owt_lr_wide; then
  echo "=== stage: owt_lr_wide ==="

  WIDE_STEPS=$(steps_for_tokens "$SHORT_TOKENS" "$BASE_BATCH" "$BASE_CONTEXT")
  WIDE_WARMUP=$(warmup_for_steps "$WIDE_STEPS")

  for LR in 2.5e-4 5e-4 1e-3 2e-3; do
    SAFE_LR=$(echo "$LR" | tr '.+-' 'p__')
    run_model "owt_wide_${SAFE_LR}" \
      --train_path_text "$OWT_TRAIN_TEXT" \
      --val_path_text "$OWT_VAL_TEXT" \
      --vocab_size 32000 \
      --context_length "$BASE_CONTEXT" \
      --batch_size "$BASE_BATCH" \
      --num_steps "$WIDE_STEPS" \
      --learning_rate_base "$LR" \
      --learning_rate_min 1e-5 \
      --steps_for_warmup "$WIDE_WARMUP" \
      --steps_for_cosine "$WIDE_STEPS" \
      --save_every "$WIDE_STEPS" \
      "${OPTIMIZED[@]}"
  done
fi

# =============================================================================
# STAGE owt_recommended
# OpenWebText at the same architecture, batch, and iteration count as ts_final,
# which is what the assignment asks for: 40000 steps x 32 x 256 = 327.68M tokens
# =============================================================================

if stage_enabled owt_recommended; then
  echo "=== stage: owt_recommended ==="
  resolve_owt_lr

  OWT_REC_STEPS=$(steps_for_tokens "$FULL_TOKENS" "$BASE_BATCH" "$BASE_CONTEXT")
  OWT_REC_WARMUP=$(warmup_for_steps "$OWT_REC_STEPS")

  run_model "owt_recommended" \
    --train_path_text "$OWT_TRAIN_TEXT" \
    --val_path_text "$OWT_VAL_TEXT" \
    --vocab_size 32000 \
    --context_length "$BASE_CONTEXT" \
    --batch_size "$BASE_BATCH" \
    --num_steps "$OWT_REC_STEPS" \
    --learning_rate_base "$OWT_BEST_LR" \
    --learning_rate_min 1e-5 \
    --steps_for_warmup "$OWT_REC_WARMUP" \
    --steps_for_cosine "$OWT_REC_STEPS" \
    --save_every 5000 \
    "${OPTIMIZED[@]}"
fi

# =============================================================================
# STAGE ts_extended
# Doubled budget: 655.36M tokens. Tests whether the fixed-budget curve was still
# descending and how much loss a second budget doubling buys.
# =============================================================================

if stage_enabled ts_extended; then
  echo "=== stage: ts_extended ==="
  resolve_ts_lr

  TS_EXT_STEPS=$(steps_for_tokens "$EXTENDED_TOKENS" "$BASE_BATCH" "$BASE_CONTEXT")
  TS_EXT_WARMUP=$(warmup_for_steps "$TS_EXT_STEPS")

  run_model "ts_extended" \
    --train_path_text "$TS_TRAIN_TEXT" \
    --val_path_text "$TS_VAL_TEXT" \
    --vocab_size 10000 \
    --context_length "$BASE_CONTEXT" \
    --batch_size "$BASE_BATCH" \
    --num_steps "$TS_EXT_STEPS" \
    --learning_rate_base "$TS_BEST_LR" \
    --learning_rate_min 1e-5 \
    --steps_for_warmup "$TS_EXT_WARMUP" \
    --steps_for_cosine "$TS_EXT_STEPS" \
    --save_every 5000 \
    "${OPTIMIZED[@]}"
fi

# =============================================================================
# STAGE owt_extended
# Doubled budget on OpenWebText: 655.36M tokens. This is the run most likely to
# produce the lowest OpenWebText loss, since the fixed-budget curve never flattened.
# =============================================================================

if stage_enabled owt_extended; then
  echo "=== stage: owt_extended ==="
  resolve_owt_lr

  OWT_EXT_STEPS=$(steps_for_tokens "$EXTENDED_TOKENS" "$BASE_BATCH" "$BASE_CONTEXT")
  OWT_EXT_WARMUP=$(warmup_for_steps "$OWT_EXT_STEPS")

  run_model "owt_extended" \
    --train_path_text "$OWT_TRAIN_TEXT" \
    --val_path_text "$OWT_VAL_TEXT" \
    --vocab_size 32000 \
    --context_length "$BASE_CONTEXT" \
    --batch_size "$BASE_BATCH" \
    --num_steps "$OWT_EXT_STEPS" \
    --learning_rate_base "$OWT_BEST_LR" \
    --learning_rate_min 1e-5 \
    --steps_for_warmup "$OWT_EXT_WARMUP" \
    --steps_for_cosine "$OWT_EXT_STEPS" \
    --save_every 5000 \
    "${OPTIMIZED[@]}"
fi

echo "All requested stages complete."
