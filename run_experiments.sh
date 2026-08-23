#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PYTHON="${PYTHON:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-cs336_basics/training_together.py}"

TS_TRAIN_TEXT="${TS_TRAIN_TEXT:-./data/TinyStoriesV2-GPT4-train.txt}"
TS_VAL_TEXT="${TS_VAL_TEXT:-./data/TinyStoriesV2-GPT4-valid.txt}"

OWT_TRAIN_TEXT="${OWT_TRAIN_TEXT:-./data/owt_train.txt}"
OWT_VAL_TEXT="${OWT_VAL_TEXT:-./data/owt_valid.txt}"

SAVE_DIR="${SAVE_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"

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

BASE_CONTEXT=256
BASE_BATCH=32

EVAL_EVERY=250
EVAL_BATCHES=20

mkdir -p "$SAVE_DIR" "$LOG_DIR"

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

  "$PYTHON" "$TRAIN_SCRIPT" \
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

# =============================================================================
# 1. SYSTEMS BENCHMARK
# Staged stack: eager FP32 -> AMP -> AMP+compile -> AMP+compile+async I/O
# =============================================================================

SYS_STEPS=1000
SYS_WARMUP=$(warmup_for_steps "$SYS_STEPS")
SYS_LR="${SYS_LR:-1e-3}"

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
run_model "sys_02_amp_compile" "${SYSTEM_COMMON[@]}" --mixed_precision --compile
run_model "sys_03_amp_compile_async" "${SYSTEM_COMMON[@]}" --mixed_precision --compile --async_io

# =============================================================================
# 2. TINYSTORIES LEARNING-RATE SWEEP
# Fixed: 40.96M tokens, batch=32, context=256
# =============================================================================

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
    --mixed_precision --compile --async_io
done

TS_BEST_LR="${TS_BEST_LR:-$(best_lr_from_prefix "ts_lr_")}"
echo "TinyStories short-run best LR: $TS_BEST_LR"

# =============================================================================
# 3. TINYSTORIES BATCH-SIZE SWEEP
# Fixed: 40.96M tokens, context=256
# =============================================================================

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
    --mixed_precision --compile --async_io
done

# =============================================================================
# 4. TINYSTORIES CONTEXT-LENGTH SWEEP
# Fixed: 40.96M tokens AND 8192 tokens/update
# 128x64 = 256x32 = 512x16 = 8192
# =============================================================================

CTX_STEPS=$(steps_for_tokens "$SHORT_TOKENS" 32 256)
CTX_WARMUP=$(warmup_for_steps "$CTX_STEPS")

run_model "ts_ctx_128" \
  --train_path_text "$TS_TRAIN_TEXT" \
  --val_path_text "$TS_VAL_TEXT" \
  --vocab_size 10000 \
  --context_length 128 \
  --batch_size 64 \
  --num_steps "$CTX_STEPS" \
  --learning_rate_base "$TS_BEST_LR" \
  --learning_rate_min 1e-5 \
  --steps_for_warmup "$CTX_WARMUP" \
  --steps_for_cosine "$CTX_STEPS" \
  --save_every "$CTX_STEPS" \
  --mixed_precision --compile --async_io

run_model "ts_ctx_256" \
  --train_path_text "$TS_TRAIN_TEXT" \
  --val_path_text "$TS_VAL_TEXT" \
  --vocab_size 10000 \
  --context_length 256 \
  --batch_size 32 \
  --num_steps "$CTX_STEPS" \
  --learning_rate_base "$TS_BEST_LR" \
  --learning_rate_min 1e-5 \
  --steps_for_warmup "$CTX_WARMUP" \
  --steps_for_cosine "$CTX_STEPS" \
  --save_every "$CTX_STEPS" \
  --mixed_precision --compile --async_io

run_model "ts_ctx_512" \
  --train_path_text "$TS_TRAIN_TEXT" \
  --val_path_text "$TS_VAL_TEXT" \
  --vocab_size 10000 \
  --context_length 512 \
  --batch_size 16 \
  --num_steps "$CTX_STEPS" \
  --learning_rate_base "$TS_BEST_LR" \
  --learning_rate_min 1e-5 \
  --steps_for_warmup "$CTX_WARMUP" \
  --steps_for_cosine "$CTX_STEPS" \
  --save_every "$CTX_STEPS" \
  --mixed_precision --compile --async_io

# =============================================================================
# 5. FINAL TINYSTORIES RUN
# Fixed: 327.68M tokens, baseline batch/context, best short-run LR
# =============================================================================

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
  --mixed_precision --compile --async_io

# =============================================================================
# 6. OPENWEBTEXT LR TRANSFER
# 0.5x / 1x / 2x TinyStories best LR, fixed 40.96M tokens
# =============================================================================

OWT_BATCH="${OWT_BATCH:-16}"
OWT_CONTEXT="${OWT_CONTEXT:-256}"
OWT_SHORT_STEPS=$(steps_for_tokens "$SHORT_TOKENS" "$OWT_BATCH" "$OWT_CONTEXT")
OWT_SHORT_WARMUP=$(warmup_for_steps "$OWT_SHORT_STEPS")

OWT_LR_HALF=$(float_mul "$TS_BEST_LR" 0.5)
OWT_LR_SAME="$TS_BEST_LR"
OWT_LR_DOUBLE=$(float_mul "$TS_BEST_LR" 2.0)

for LABEL in half same double; do
  case "$LABEL" in
    half) LR="$OWT_LR_HALF" ;;
    same) LR="$OWT_LR_SAME" ;;
    double) LR="$OWT_LR_DOUBLE" ;;
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
    --mixed_precision --compile --async_io
done

OWT_BEST_LR="${OWT_BEST_LR:-$(best_lr_from_prefix "owt_lr_")}"
echo "OpenWebText short-run best LR: $OWT_BEST_LR"

# =============================================================================
# 7. FINAL OPENWEBTEXT RUN
# Fixed: 327.68M tokens
# =============================================================================

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
  --mixed_precision --compile --async_io

echo "All runs complete."
