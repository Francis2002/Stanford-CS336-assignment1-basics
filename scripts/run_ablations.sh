#!/bin/bash
# =============================================================================
# CS336 Assignment 1 - Ablation Study Runner
# Run in a screen session on the remote GPU machine:
#   screen -S ablations
#   bash scripts/run_ablations.sh
# =============================================================================

PYTHON="./.venv_cluster/bin/python"
SCRIPT="cs336_basics/training_together.py"

# Common settings for TinyStories ablations
COMMON_ARGS="--train_path_text ./data/TinyStoriesV2-GPT4-train.txt \
  --val_path_text ./data/TinyStoriesV2-GPT4-valid.txt \
  --vocab_size 10000 \
  --context_length 256 \
  --num_steps 10000 \
  --print_every 500 \
  --save_every 2500 \
  --keep_last 2 \
  --mixed_precision \
  --async_io"

echo "============================================="
echo "Starting Ablation Suite: $(date)"
echo "============================================="

# --- BASELINE ---
echo "[1/12] Running BASELINE (d=512, lr=1e-3, bs=32, 4 layers)"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32

# --- LEARNING RATE ABLATION ---
echo "[2/12] Running LR=5e-4"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 5e-4 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32

echo "[3/12] Running LR=2e-3"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 2e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32

echo "[4/12] Running LR=5e-3"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 5e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32

# --- BATCH SIZE ABLATION ---
echo "[5/12] Running BS=16"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 16

echo "[6/12] Running BS=64"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 64

echo "[7/12] Running BS=128"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 128

# --- DEPTH/WIDTH ABLATION (same param budget ~25M) ---
echo "[8/12] Running DEEP (d=384, 8 layers)"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 384 --d_ff 1024 --num_heads 12 --num_layers 8 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32

echo "[9/12] Running WIDE (d=768, 2 layers)"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 768 --d_ff 2048 --num_heads 12 --num_layers 2 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32

# --- WARMUP ABLATION ---
echo "[10/12] Running NO WARMUP"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 0 --steps_for_cosine 10000 \
  --batch_size 32

echo "[11/12] Running LONG WARMUP (2000 steps)"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 2000 --steps_for_cosine 10000 \
  --batch_size 32

# --- CONTEXT LENGTH ABLATION ---
echo "[12/12] Running CONTEXT=512"
$PYTHON $SCRIPT $COMMON_ARGS \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --learning_rate_base 1e-3 --learning_rate_min 1e-5 \
  --steps_for_warmup 500 --steps_for_cosine 10000 \
  --batch_size 32 --context_length 512

echo "============================================="
echo "Ablation Suite Complete: $(date)"
echo "============================================="
echo "Results are in cs336_basics/checkpoints/*/metrics.jsonl"
echo "Run: python scripts/plot_results.py to visualize"
