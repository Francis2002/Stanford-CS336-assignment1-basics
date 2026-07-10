# Language Modeling from Scratch: Building and Profiling a Minimal GPT Training Stack

CS336 Assignment 1 implementation with mixed-precision training, asynchronous data loading, hyperparameter ablations, and text generation. All components (tokenizer, transformer, optimizer, training loop) built from scratch in PyTorch.

## 1. Summary

This project implements the complete language-model training stack end-to-end: a byte-pair encoding tokenizer, a decoder-only Transformer with ROPE and SwiGLU activation, an AdamW optimizer, a training loop with cosine learning-rate scheduling, checkpointing, and text generation. Due to memory and throughput bottlenecks on a consumer RTX GPU, the system was then extended with mixed-precision training (FP16 via GradScaler) and asynchronous data loading. Hyperparameter ablations on learning rate, batch size, model depth/width, warmup duration, and context length were run on TinyStories datasets.

The best configuration achieved a validation loss of 1.19 on TinyStories after 20k steps (~42 minutes on an RTX GPU with mixed precision), and the trained model generates coherent short stories.


## 2. Motivation

<!-- 
Write 3-4 paragraphs explaining:
- Why build from scratch instead of using HuggingFace/nanoGPT
- What you learn by implementing each piece yourself (tokenizer bugs are silent, GPU memory constraints force real systems thinking, training speed depends on data movement not just model code)
- Your personal goal: understand the full stack deeply enough to replicate the core ideas behind modern LLMs
- Brief mention of the CS336 course context
-->

One can learn about Transformers and look at diagrams of the GPT architecture all they want, but then, inside the little boxes and arrows, with terms like '


## 3. System Overview

<!-- 
Insert a clean architecture diagram here (draw.io, excalidraw, or mermaid rendered to PNG).
The diagram should show the full pipeline:

Raw Text → BPE Tokenizer → Token IDs (.bin) → DataLoader (async workers) → Transformer → Cross-Entropy Loss → AdamW → Checkpoint / Eval / Generation

Label each component with the file that implements it.
-->

[PLACEHOLDER: architecture diagram — pipeline from raw text to generation]


## 4. Tokenizer

The tokenizer implements byte-pair encoding from scratch, starting from raw UTF-8 bytes and iteratively merging the most frequent adjacent pair until reaching the target vocabulary size.

<!-- 
Write 3-4 paragraphs covering:
- The BPE algorithm: start from byte-level, count pair frequencies, merge greedily
- Pre-tokenization with regex splitting (GPT-2 pattern) to avoid merges across word boundaries
- Special token handling (<|endoftext|>)
- Streaming encode_iterable for large files without loading everything into memory
- One concrete example: show a sentence and its token IDs, then decode back
- What was tricky: performance (the naive approach is O(n²) per merge), handling edge cases with unicode
- How you verified correctness: comparison against tiktoken on the same input
-->


## 5. Transformer Architecture

The model follows the modern decoder-only Transformer design: pre-norm residual blocks with RMSNorm, rotary positional embeddings, SwiGLU feed-forward networks, and causal masking.

### Attention

<!-- 
Write 2-3 paragraphs:
- Scaled dot-product attention with causal mask
- Multi-head self-attention: project Q/K/V, split into heads, apply RoPE to Q and K, attend, concatenate, project out
- Why RoPE over learned positional embeddings (relative position encoding, extrapolation)
-->

### Feed-Forward Network

<!-- 
Write 1-2 paragraphs:
- SwiGLU: gate mechanism with SiLU activation
- Why SwiGLU over standard ReLU MLP (better loss at same parameter count, per Shazeer 2020)
- The d_ff choice: 8/3 * d_model rounded to nearest multiple (1344 for d_model=512)
-->

### Normalization and Residuals

<!-- 
Write 1-2 paragraphs:
- RMSNorm over LayerNorm (simpler, no mean subtraction, empirically equivalent)
- Pre-norm architecture (norm before attention/FFN, not after)
- Float32 upcast inside RMSNorm for numerical stability under mixed precision
-->

### Parameter Count

<!-- 
State the parameter count for your default model (d_model=512, 4 layers, vocab=10000).
Briefly explain where the parameters live: embeddings, attention projections, FFN, output projection.
-->


## 6. Optimizer and Training Loop

### AdamW

<!-- 
Write 2 paragraphs:
- Decoupled weight decay (applied directly to weights, not through gradient)
- Bias correction for first and second moments
- Why AdamW over Adam: weight decay regularization is cleaner when decoupled from the adaptive learning rate
-->

### Learning Rate Schedule

<!-- 
Write 1-2 paragraphs:
- Linear warmup → cosine annealing → constant minimum
- Why warmup matters (stabilizes early training when gradients are noisy and large)
- The ablation confirmed this: no warmup led to worse final loss
-->

### Gradient Clipping

<!-- 
Write 1 paragraph:
- Global norm clipping at max_norm=1.0
- Prevents gradient explosions during early training or on unusual batches
-->

### Checkpointing

<!-- 
Write 1-2 paragraphs:
- Atomic writes (write to .tmp, then os.replace) to prevent corruption on crash
- Checkpoint rotation: keep only last N checkpoints to save disk
- Config saved alongside checkpoint for full reproducibility
- torch.compile compatibility: strip _orig_mod. prefix on load
-->


## 7. Systems Bottlenecks and Solutions

This section documents the practical engineering challenges encountered when training on a consumer GPU and the solutions implemented to overcome them.

### The Problem: OOM and Slow Training

<!-- 
Write 2-3 paragraphs:
- Initial naive implementation: synchronous data loading, FP32 everything, no memory management
- OOM on RTX GPU when trying batch_size=64 with context_length=256 at FP32
- Training was slow: GPU was idle waiting for data, checkpointing blocked the training loop
- Quantify: what was the throughput before optimizations vs after
-->

### Mixed Precision Training

<!-- 
Write 2-3 paragraphs:
- torch.amp.autocast with FP16 for forward pass (matmuls run in half precision)
- GradScaler to prevent gradient underflow in FP16
- Key detail: RMSNorm upcasts to FP32 internally for numerical stability
- Memory reduction: roughly 2x for activations, allowing larger batch sizes
- Speed improvement: tensor cores on RTX GPUs accelerate FP16 matmuls
- What to watch out for: loss scaling, overflow detection, unscaling before gradient clipping
-->

### Asynchronous Data Loading

<!-- 
Write 2-3 paragraphs:
- Problem: CPU-to-GPU data transfer was blocking the training loop
- Solution: PyTorch DataLoader with num_workers=4 and pin_memory=True
- TokenDataset wraps numpy memmap for zero-copy reads from disk
- Random sampling (ignore DataLoader index, sample uniformly) for LLM pre-training
- Result: GPU utilization went from ~60% to ~95%
-->

### Asynchronous Checkpointing and Logging

<!-- 
Write 1-2 paragraphs:
- ThreadPoolExecutor with 1 worker for non-blocking saves
- Deep copy model state to CPU before handing to background thread (avoids race condition with next forward pass)
- Metrics logging also async to avoid filesystem blocking
-->

### Memory Management

<!-- 
Write 1-2 paragraphs:
- Explicit del of large tensors after each step to prevent peak memory overlap
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for better fragmentation handling
- torch.cuda.empty_cache() before checkpointing
-->

### torch.compile

<!-- 
Write 1 paragraph:
- Fused kernels via torch.compile reduce kernel launch overhead
- First step is slow (compilation), subsequent steps are faster
- Requires handling the _orig_mod. prefix in state dicts
-->


## 8. Ablations

All ablations were run on TinyStories (vocab_size=10000) on an RTX GPU with mixed precision enabled. The baseline configuration is: d_model=512, d_ff=1344, num_heads=16, num_layers=4, context_length=256, batch_size=32, lr=1e-3, warmup=500 steps, cosine decay over 10k steps.

The two best runs used 20k steps with batch_size=64 and warmup=1000, trained over ~42 minutes each.

### Learning Rate

<!-- 
Write 2-3 paragraphs interpreting the results:
- lr=3e-3 was the best (1.19 val loss at 20k steps), followed by lr=1e-3 (1.24)
- lr=5e-3 diverged badly (2.60 val loss) — too aggressive
- lr=5e-4 was too conservative (1.39) — underfitting at this step budget
- Conclusion: for this model size and dataset, the optimal lr is around 2-3e-3. Higher learning rates need more warmup to stabilize.
-->

[PLACEHOLDER: learning rate comparison plot — val loss vs steps for lr ∈ {5e-4, 1e-3, 2e-3, 3e-3, 5e-3}]

### Batch Size

<!-- 
Write 2-3 paragraphs:
- bs=64 with 20k steps was best (processes 2x tokens total vs bs=32 at 10k steps)
- At fixed step count (10k), bs=64 (1.32) slightly beat bs=32 (1.34) — more tokens per step helps
- bs=16 (1.39) was worse — too noisy gradients
- bs=128 run likely OOM'd or didn't complete (not in results) — confirm and note
- Key insight: larger batch size is better IF you can afford the memory and have enough steps
-->

[PLACEHOLDER: batch size comparison plot — val loss vs steps for bs ∈ {16, 32, 64}]

### Model Architecture: Depth vs Width

<!-- 
Write 2-3 paragraphs:
- Deep (d=384, 8 layers): 1.32 val loss — competitive despite smaller hidden dim
- Baseline (d=512, 4 layers): 1.34
- Wide (d=768, 2 layers): 1.39 — worst of the three
- Conclusion: depth matters more than width at this scale. More layers = more representational capacity for sequential reasoning. This aligns with the literature (deeper models generalize better per parameter).
-->

[PLACEHOLDER: depth vs width comparison plot]

### Warmup Duration

<!-- 
Write 1-2 paragraphs:
- No warmup (1.35) vs 500 steps (1.34) vs 1000 steps (1.24 at 20k) vs 2000 steps (1.29)
- Warmup helps, but too much warmup wastes steps on suboptimal learning rates
- Sweet spot around 5-10% of total training steps
-->

### Context Length

<!-- 
Write 1-2 paragraphs:
- context_length=512 (1.24) vs 256 (1.34) at same step count
- Longer context sees more tokens per step and captures longer-range dependencies
- Tradeoff: 2x memory cost per sample, so effective batch size is halved at same memory budget
-->

### Summary of Findings

<!-- 
Write 1 paragraph synthesizing the key takeaways:
- Best config: lr=3e-3, bs=64, warmup=1000, 20k steps, context=256 → 1.19 val loss
- Most impactful factors in order: learning rate > training duration > batch size > depth > context length > warmup
- These findings transfer to OWT with minor adjustments (OWT is harder, expect higher loss)
-->

[PLACEHOLDER: summary table of all runs sorted by val loss]


## 9. Text Generation

The trained model generates coherent short stories in the style of TinyStories. Generation uses autoregressive sampling with temperature scaling and nucleus (top-p) filtering.

<!-- 
Write 1-2 paragraphs about the decoding implementation:
- Temperature controls randomness (lower = more deterministic)
- Top-p (nucleus sampling) truncates the tail of the distribution
- Generation stops at <|endoftext|> or max_length
-->

### Sample Outputs

**Prompt:** "There was a boy named Joao"
**Parameters:** temperature=0.8, top_p=0.9

> There was a boy named Joao. She was very sad because she didn't have any friends. One day, Jooo saw a big tree with a hole in it. She walked closer and looked inside. To her surprise, she found a little puppy! The puppy was very happy and wagged its tail. Joo wanted to play with the puppy, so she gave it a big kiss. The puppy licked her face and they played together all day. They had so much fun! From that day on, Joo and the puppy were best friends.

<!-- 
Add 2-3 more generation samples with different prompts.
Note the model's strengths (coherent narrative, proper story structure) and weaknesses (gender confusion from "Joao" being unusual in training data, name corruption across tokens).
-->


## 10. Lessons Learned

<!-- 
Write 5-7 bullet points of honest engineering lessons. Examples:
- Memory management in PyTorch is not automatic — you must think about tensor lifetimes
- Mixed precision is not free: you need to handle loss scaling, upcast norms, and watch for NaN gradients
- The biggest speedup came from not blocking the GPU (async data loading), not from faster math
- Hyperparameter sensitivity is real: lr=5e-3 vs lr=3e-3 is the difference between divergence and best result
- torch.compile is powerful but adds complexity to checkpointing (prefix stripping)
- Atomic writes for checkpoints are essential — a crash during save corrupts the file without them
- Random sampling (ignoring DataLoader indices) is the right approach for LLM pre-training but breaks standard PyTorch assumptions about epochs
-->


## 11. Reproducibility

**Hardware:** NVIDIA RTX GPU (consumer), CUDA 12.4
**Software:** Python 3.12, PyTorch 2.6.0+cu124, uv for dependency management
**Dataset:** TinyStories V2 (GPT-4 generated), vocab_size=10000

### Best Run Configuration

```
d_model=512, d_ff=1344, num_heads=16, num_layers=4
context_length=256, batch_size=64, num_steps=20000
lr=3e-3, lr_min=3e-5, warmup=1000, cosine_decay=20000
weight_decay=0.05, grad_clip=1.0
mixed_precision=True, async_io=True
```

### Commands

```bash
# Train best model
.venv_cluster/bin/python cs336_basics/training_together.py \
  --d_model 512 --d_ff 1344 --num_heads 16 --num_layers 4 \
  --context_length 256 --batch_size 64 --num_steps 20000 \
  --learning_rate_base 3e-3 --learning_rate_min 3e-5 \
  --steps_for_warmup 1000 --steps_for_cosine 20000 \
  --mixed_precision --async_io

# Generate text
.venv_cluster/bin/python cs336_basics/decoding.py \
  --load_path cs336_basics/checkpoints/<run_slug>/checkpoint_020000.pt \
  --prompt "Once upon a time" --temp 0.8 --top_p 0.9

# Run ablation suite
bash scripts/run_ablations.sh
```


## 12. Limitations

<!-- 
Write 3-4 bullet points:
- Single GPU only — no distributed training (that's Assignment 2)
- No Flash Attention — attention is O(n²) in memory, limiting context length
- Small model (~25M params) and small dataset — results don't directly transfer to larger scales
- BPE tokenizer is slow (Python-level iteration) — production tokenizers use Rust/C++
- No dropout in current implementation (assignment spec uses it but ablations didn't vary it)
-->


## 13. What's Next

<!-- 
Write 2-3 sentences:
- Assignment 2: Flash Attention in Triton + Distributed Data Parallel for multi-GPU training
- After systems optimizations: attempt the OWT leaderboard with a larger model under time constraints
- Potential extension: Mixture of Experts to explore sparse architectures at this scale
-->
