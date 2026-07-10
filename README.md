# Language Modeling from Scratch

End-to-end implementation of a GPT-style language model training stack, built from scratch in PyTorch. Every component — BPE tokenizer, multi-head attention with RoPE, SwiGLU feed-forward, RMSNorm, AdamW optimizer, training loop, and autoregressive generation — is implemented without high-level abstractions.

Extended with mixed-precision training, asynchronous data loading, and systematic hyperparameter ablations. Trained on TinyStories and OpenWebText.

**Technical Report:** [docs/assignment1-report.md](docs/assignment1-report.md)

### What's Implemented

- Byte-pair encoding tokenizer with streaming encode for large corpora
- Decoder-only Transformer: RoPE, SwiGLU, RMSNorm, causal masking
- AdamW with decoupled weight decay and bias correction
- Cosine learning-rate schedule with linear warmup
- Mixed-precision training (FP16 + GradScaler)
- Async data loading (DataLoader + memmap) and async checkpointing
- Autoregressive generation with temperature and nucleus sampling
- Ablation suite across learning rate, batch size, depth/width, warmup, context length

### Quick Start

```bash
# Install dependencies
uv sync

# Train (TinyStories, best config)
uv run python cs336_basics/training_together.py \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --batch_size 64 --num_steps 20000 \
  --learning_rate_base 3e-3 --learning_rate_min 3e-5 \
  --steps_for_warmup 1000 --steps_for_cosine 20000 \
  --mixed_precision --async_io \
  --train_path_text ./data/TinyStoriesV2-GPT4-train.txt \
  --val_path_text ./data/TinyStoriesV2-GPT4-valid.txt

# Generate text
uv run python cs336_basics/decoding.py \
  --load_path cs336_basics/checkpoints/<run_slug>/checkpoint_020000.pt \
  --prompt "Once upon a time" --temp 0.8 --top_p 0.9

# Run ablation suite
bash scripts/run_ablations.sh
```

### Results (TinyStories)

Best validation loss: **1.19** (lr=3e-3, bs=64, 20k steps, ~42 min on RTX GPU)

See the [full report](docs/assignment1-report.md) for ablation analysis and generation samples.

---

*Based on Stanford CS336: Language Modeling from Scratch (Spring 2025). For the original assignment description, see [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf).*

---

## Setup

### Environment

Dependencies managed with [uv](https://github.com/astral-sh/uv) for reproducibility.

```sh
uv sync
uv run python <script.py>
```

### Tests

```sh
uv run pytest
```

### Data

```sh
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
cd ..
```

