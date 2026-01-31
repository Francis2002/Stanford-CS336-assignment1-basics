import torch
import argparse
import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.data_loading import load_data
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.adamw import AdamW
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.rope import RotaryPositionalEmbedding

import pickle
from pathlib import Path
import json
import logging
import uuid
from datetime import datetime
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_bpe_trainer(text_path, vocab_size, special_characters, use_profiler=True):
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    full_text_path = PROJECT_ROOT / text_path
    vocab, merges = train_bpe(full_text_path, vocab_size, special_characters, use_profiler=use_profiler)

    longest_token = max(vocab.values(), key=len)
    print(f"Longest token (bytes): {longest_token}")
    print(f"Length in bytes: {len(longest_token)}")
    print(f"Longest token as string: {longest_token.decode('utf-8', errors='replace')}")

    # Create vocab and merges paths
    vocab_path = text_path.replace(".txt", f"_vocab_{vocab_size}.json")
    merges_path = text_path.replace(".txt", f"_merges_{vocab_size}.json")
    full_vocab_path = PROJECT_ROOT / vocab_path
    full_merges_path = PROJECT_ROOT / merges_path

    # Save as JSON
    json_vocab = {str(k): list(v) for k, v in vocab.items()}
    with open(full_vocab_path, 'w') as f:
        json.dump(json_vocab, f, indent=2)
    
    json_merges = [[list(m[0]), list(m[1])] for m in merges]
    with open(full_merges_path, 'w') as f:
        json.dump(json_merges, f, indent=2)

    return vocab_path, merges_path

def tokenize_data(text_path, merges_path, vocab_path, special_characters):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    full_vocab_path = PROJECT_ROOT / vocab_path
    full_merges_path = PROJECT_ROOT / merges_path
    my_tokenizer = Tokenizer.from_files(full_vocab_path, full_merges_path, special_characters)
    
    with open(PROJECT_ROOT / text_path, 'r') as f:
        text = f.read()

    token_ids = my_tokenizer.encode(text, logging=True)

    # Save as raw binary .bin
    path_tokens = PROJECT_ROOT / text_path.replace(".txt", f"_{len(my_tokenizer.vocab)}.bin")
    np.array(token_ids, dtype=np.uint16).tofile(path_tokens)

    return path_tokens

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    # LR Hyperparameters
    parser.add_argument("--learning_rate_base", type=float, default=1e-3)
    parser.add_argument("--learning_rate_min", type=float, default=1e-5)
    parser.add_argument("--steps_for_warmup", type=int, default=0)
    parser.add_argument("--steps_for_cosine", type=int, default=5000)
    # Optimizer hyperparameters
    parser.add_argument("--batch_size", type=int, default=32) # Aim for batch_size * num_steps * context_length = 327_680_000 total tokens processed on GPU or 40_000_000 on CPU
    parser.add_argument("--num_steps", type=int, default=5000) # iteration starts at 1 for convenience
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    # BPE arguments
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--special_characters", type=list, default=['<|endoftext|>'])
    parser.add_argument("--use_profiler", type=bool, default=True)
    # Rope
    parser.add_argument("--rope_theta", type=int, default=10000)
    # Training paths
    parser.add_argument("--train_path_tokens", type=str, default="./data/tinystories_token_ids_10000.bin")
    parser.add_argument("--val_path_tokens", type=str, default="./data/tinystories_token_ids_10000.bin")
    parser.add_argument("--train_path_text", type=str, default="./data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--val_path_text", type=str, default="./data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_path", type=str, default="./data/tinystories_vocab_10000.json")
    parser.add_argument("--merges_path", type=str, default="./data/tinystories_merges_10000.json")
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--log_dir", type=str, default="./logs")
    # Checkpoint paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=250)
    # Device
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    # Tokenize data if needed
    if args.train_path_tokens is None:
        print("No token path provided. Tokenizing the training data...")
        if args.vocab_path is None or args.merges_path is None:
            print("No vocab path or merges path provided. We will train BPE from scratch.")
            vocab_path, merges_path = run_bpe_trainer(args.train_path_text, args.vocab_size, args.special_characters, use_profiler)
            args.vocab_path = vocab_path # update these so that for val we already have it
            args.merges_path = merges_path
        args.train_path_tokens = tokenize_data(args.train_path_text, args.merges_path, args.vocab_path, args.special_characters)
    
    if args.val_path_tokens is None:
        print("No token path provided. Tokenizing the validation data...")
        args.val_path_tokens = tokenize_data(args.val_path_text, args.merges_path, args.vocab_path, args.special_characters)

    # Load data efficiently with np.memmap
    train_data = np.memmap(args.train_path_tokens, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path_tokens, dtype=np.uint16, mode="r")

    # Load model
    model = TransformerLM(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, device=device)
    model.to(device)

    # Load optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate_base, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=args.weight_decay)

    # Load checkpoint
    if args.load_path is not None:
        iteration = load_checkpoint(args.load_path, model, optimizer)
    else:
        iteration = 1

    # Experiment naming with slugs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = f"{timestamp}_{uuid.uuid4().hex[:4]}"
    
    # Define directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    LOG_DIR = os.path.join(BASE_DIR, args.log_dir)
    SAVE_DIR = os.path.join(BASE_DIR, args.save_dir)
    RUN_DIR = os.path.join(SAVE_DIR, run_slug)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    
    # Keep a central index of all runs
    summary_index_path = os.path.join(LOG_DIR, "runs_summary.jsonl")
    run_entry = {
        "run_slug": run_slug,
        "timestamp": timestamp,
        "config": vars(args),
        "run_dir": RUN_DIR
    }
    with open(summary_index_path, "a") as f:
        f.write(json.dumps(run_entry) + "\n")
    
    # Metrics logging path
    metrics_path = os.path.join(RUN_DIR, "metrics.jsonl")
    
    logger.info(f"Starting run {run_slug}. Logging to {RUN_DIR}")

    # Initialize Rope
    rope = RotaryPositionalEmbedding(args.rope_theta, args.d_model // args.num_heads, args.context_length, device=device)

    start_time = time.time()
    
    # Training Loop
    for step in range(iteration, args.num_steps + 1):
        
        # Training
        optimizer.zero_grad()
        x, y = load_data(train_data, args.batch_size, args.context_length, args.device)
        y_pred = model(x, rope=rope)
        loss = cross_entropy(y_pred, y)
        loss.backward()

        if args.max_grad_norm is not None:
            gradient_clipping(model.parameters(), args.max_grad_norm)

        lr = learning_rate_schedule(step, args.learning_rate_base, args.learning_rate_min, args.steps_for_warmup, args.steps_for_cosine)
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        # Validation
        with torch.no_grad():
            x_val, y_val = load_data(val_data, args.batch_size, args.context_length, args.device)
            y_pred_val = model(x_val, rope=rope)
            val_loss = cross_entropy(y_pred_val, y_val)

        # Log metrics every step (streaming to JSONL)
        metrics = {
            "step": step,
            "wall_time": time.time() - start_time,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "lr": lr,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if step % args.print_every == 0:
            logger.info(f"Step {step}/{args.num_steps} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if step % args.save_every == 0:
            checkpoint_path = os.path.join(RUN_DIR, f"checkpoint_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path, config=vars(args))
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training complete.")
        
                        