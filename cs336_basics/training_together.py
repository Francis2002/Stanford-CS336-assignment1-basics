import torch
import argparse
import numpy as np
import os

# Set CUDA allocator configuration to handle fragmentation more gracefully.
# 'expandable_segments:True' allows the allocator to use more flexible segment management.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.data_loading import load_data, TokenDataset
from torch.utils.data import DataLoader
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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_bpe_trainer(text_path, vocab_size, special_characters, use_profiler=True):
    
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

    full_vocab_path = PROJECT_ROOT / vocab_path
    full_merges_path = PROJECT_ROOT / merges_path
    my_tokenizer = Tokenizer.from_files(full_vocab_path, full_merges_path, special_characters)
    
    file_size = os.path.getsize(PROJECT_ROOT / text_path)

    def progress_wrapper(iterable, pbar):
        for item in iterable:
            yield item
            pbar.update(len(item.encode('utf-8')))

    with open(PROJECT_ROOT / text_path, 'r') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Tokenizing {os.path.basename(text_path)}") as pbar:
            token_ids = list(my_tokenizer.encode_iterable(progress_wrapper(f, pbar)))

    # Save as raw binary .bin
    path_tokens = PROJECT_ROOT / text_path.replace(".txt", f"_token_ids_{len(my_tokenizer.vocab)}.bin")
    np.array(token_ids, dtype=np.uint16).tofile(path_tokens)

    return str(path_tokens)

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
    parser.add_argument("--steps_for_cosine", type=int, default=40000)
    # Optimizer hyperparameters
    parser.add_argument("--batch_size", type=int, default=32) # Assignment: batch_size * num_steps * context_length = 327_680_000 total tokens processed on GPU or 40_000_000 on CPU
    parser.add_argument("--num_steps", type=int, default=40000) # iteration starts at 1 for convenience
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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
    parser.add_argument("--train_path_tokens", type=str, default=None)
    parser.add_argument("--val_path_tokens", type=str, default=None)
    parser.add_argument("--train_path_text", type=str, default="./data/owt_train.txt")
    parser.add_argument("--val_path_text", type=str, default="./data/owt_valid.txt")
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--merges_path", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--log_dir", type=str, default="./logs")
    # Checkpoint paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--keep_last", type=int, default=3, help="Number of checkpoints to retain. -1 to keep all.")
    # Extensions
    parser.add_argument("--mixed_precision", action="store_true", help="Enable FP16/BF16 training")
    parser.add_argument("--async_io", action="store_true", help="Enable asynchronous logging and checkpointing")
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
        print("No token path provided. Checking if token path with same name as text path exists...")
        print(f"Checking for token path: {PROJECT_ROOT / args.train_path_text.replace(".txt", f"_token_ids_{args.vocab_size}.bin")}")

        possible_train_token_path = PROJECT_ROOT / args.train_path_text.replace(".txt", f"_token_ids_{args.vocab_size}.bin")
        if possible_train_token_path.exists():
            args.train_path_tokens = str(possible_train_token_path)
            print(f"Found token path: {args.train_path_tokens}")
        else:
            print("No token path provided. Tokenizing the training data...")
        
            if args.vocab_path is None or args.merges_path is None:
                print("No vocab path or merges path provided. Checking if vocab and merges paths with same name as text path exist...")
                print(f"Checking for merges path: {PROJECT_ROOT / args.train_path_text.replace(".txt", f"_merges_{args.vocab_size}.json")}")
                print(f"Checking for vocab path: {PROJECT_ROOT / args.train_path_text.replace(".txt", f"_vocab_{args.vocab_size}.json")}")

                possible_train_merges_path = PROJECT_ROOT / args.train_path_text.replace(".txt", f"_merges_{args.vocab_size}.json")
                possible_train_vocab_path = PROJECT_ROOT / args.train_path_text.replace(".txt", f"_vocab_{args.vocab_size}.json")

                if possible_train_merges_path.exists() and possible_train_vocab_path.exists():
                    args.merges_path = str(possible_train_merges_path)
                    args.vocab_path = str(possible_train_vocab_path)
                    print(f"Found merges path: {args.merges_path}")
                    print(f"Found vocab path: {args.vocab_path}")
                else:
                    print("No vocab path or merges path provided. We will train BPE from scratch.")
                    vocab_path, merges_path = run_bpe_trainer(args.train_path_text, args.vocab_size, args.special_characters, args.use_profiler)
                    args.vocab_path = vocab_path # update these so that for val we already have it
                    args.merges_path = merges_path
            args.train_path_tokens = tokenize_data(args.train_path_text, args.merges_path, args.vocab_path, args.special_characters)
    
    if args.val_path_tokens is None:
        print("No token path provided. Checking if token path with same name as text path exists...")
        print(f"Checking for token path: {PROJECT_ROOT / args.val_path_text.replace(".txt", f"_token_ids_{args.vocab_size}.bin")}")

        possible_val_token_path = PROJECT_ROOT / args.val_path_text.replace(".txt", f"_token_ids_{args.vocab_size}.bin")
        if possible_val_token_path.exists():
            args.val_path_tokens = str(possible_val_token_path)
            print(f"Found token path: {args.val_path_tokens}")
        else:
            print("No token path provided. Tokenizing the validation data...")
            args.val_path_tokens = tokenize_data(args.val_path_text, args.merges_path, args.vocab_path, args.special_characters)

    # Load data efficiently with np.memmap
    train_data = np.memmap(args.train_path_tokens, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path_tokens, dtype=np.uint16, mode="r")

    # Load model
    model = TransformerLM(args.vocab_size, args.context_length, args.num_layers, args.d_model, args.num_heads, args.d_ff, device=device)
    model.to(device)

    # Torch Compile (Fused Kernels), kinda like jit
    if device.type == "cuda":
        logger.info("Compiling model (torch.compile)...")
        model = torch.compile(model)

    # Load optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate_base, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=args.weight_decay)

    # Load checkpoint
    if args.load_path is not None:
        iteration = load_checkpoint(args.load_path, model, optimizer, device=device)
    else:
        iteration = 1

    # Parallel Data Loading
    train_dataset = TokenDataset(train_data, args.context_length)
    val_dataset = TokenDataset(val_data, args.context_length)
    
    # Workers for speedup
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=4, 
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=4, 
        pin_memory=(device.type == "cuda")
    )
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Mixed Precision Setup
    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision and device.type == "cuda" else None

    # Async I/O uses 1 thread for saving checkpoints/metrics to avoid halting the main thread
    executor = ThreadPoolExecutor(max_workers=1) if args.async_io else None

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

    start_time = time.perf_counter()
    
    # Training Loop
    for step in range(iteration, args.num_steps + 1):
        
        # Training
        optimizer.zero_grad()
        
        # Pull next batch from background workers
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)

        # Mixed Precision Forward Pass
        # autocast allows matmuls to run in float16/bf16 for speed and memory savings
        with torch.amp.autocast("cuda", enabled=(args.mixed_precision and device.type == "cuda")):
            y_pred = model(x, rope=rope)
            loss = cross_entropy(y_pred, y)

        if scaler:
            # Scale loss by the scaler factor to prevent gradients from underflowing in float16.
            # Then unscale them before the optimizer step so that the math stays correct
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.max_grad_norm is not None:
                gradient_clipping(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.max_grad_norm is not None:
                gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()

        lr = learning_rate_schedule(step, args.learning_rate_base, args.learning_rate_min, args.steps_for_warmup, args.steps_for_cosine)
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Validation
        with torch.no_grad():
            try:
                x_val, y_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_val, y_val = next(val_iter)
            
            x_val, y_val = x_val.to(device), y_val.to(device)
            
            with torch.amp.autocast("cuda", enabled=(args.mixed_precision and device.type == "cuda")):
                y_pred_val = model(x_val, rope=rope)
                val_loss = cross_entropy(y_pred_val, y_val)

        # Log metrics every step (streaming to JSONL), with asynchronous logging
        metrics = {
            "step": step,
            "wall_time": time.perf_counter() - start_time,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "lr": lr,
        }
        
        def save_metrics(path, data):
            with open(path, "a") as f:
                f.write(json.dumps(data) + "\n")

        if executor:
            executor.submit(save_metrics, metrics_path, metrics)
        else:
            save_metrics(metrics_path, metrics)

        if step % args.print_every == 0:
            logger.info(f"Step {step}/{args.num_steps} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if step % args.save_every == 0:
            # Clear CUDA cache before checkpointing to ensure contiguous memory for saving
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint_path = os.path.join(RUN_DIR, f"checkpoint_{step:06d}.pt")
            
            # Asynchronous Checkpointing
            if executor:
                # We need to deepcopy the state dict, because it yields references to the live model weights, 
                # and so the next iteration might change them in the next step while the executor thread is still writing them
                model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                optim_state = pickle.loads(pickle.dumps(optimizer.state_dict())) # Deep copy
                executor.submit(save_checkpoint, model_state, optim_state, step, checkpoint_path, config=vars(args))
                logger.info(f"Started async checkpoint save to {checkpoint_path}")
            else:
                save_checkpoint(model, optimizer, step, checkpoint_path, config=vars(args))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Checkpoint Rotation
            if args.keep_last != -1:
                # Rotation logic: keep only the last N checkpoints
                ckpt_files = sorted([f for f in os.listdir(RUN_DIR) if f.startswith("checkpoint_") and f.endswith(".pt")])
                if len(ckpt_files) > args.keep_last:
                    for old_ckpt in ckpt_files[:-args.keep_last]:
                        os.remove(os.path.join(RUN_DIR, old_ckpt))
                        logger.info(f"Deleted old checkpoint {old_ckpt} (Rotation policy)")

        # Explicitly delete large tensor graphs before the next iteration.
        # This prevents PyTorch's allocator from experiencing peak overlap, 
        # when allocating the forward pass of step N+1 while the backward pass graphs of step N are still in scope
        del x, y, y_pred, loss
        del x_val, y_val, y_pred_val, val_loss

    if executor:
        executor.shutdown(wait=True)
    logger.info("Training complete.")
        
                        