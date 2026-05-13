import torch
import argparse
import json
import os
import time
import logging
from pathlib import Path
import torch
from datetime import datetime

from cs336_basics.softmax import softmax
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.data_loading import load_data

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

def get_config_from_checkpoint(checkpoint_path):
    checkpoint_path = PROJECT_ROOT / checkpoint_path
    parent = checkpoint_path.parent
    config_path = parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    run_slug = parent.name
    return config, run_slug

def decode(model, x, rope, eot_token_id, max_length, temp=None, top_p=None, device=None):
    """
    Decodes the given input sequence using the given model and parameters.
    
    Args:
        model: The model to use for decoding.
        x: The input sequence to decode. Shape (batch_size, context_length)
        rope: The rotary positional embedding to use.
        eot_token_id: The end-of-text token ID.
        max_length: The maximum length of the output sequence.
        temp: The temperature to use for sampling.
        top_p: The top-p value to use for sampling.
        device: The device to use for decoding.
    Returns:
        The decoded sequence.
    """
    total_tokens_generated = 0
    with torch.no_grad():
        while x[0, -1] != eot_token_id and total_tokens_generated < max_length:
            # Get logits
            logits = model(x, rope=rope) # Shape (batch_size, context_length, vocab_size)

            # Get next token logits
            next_token_logits = logits[0, -1]

            if temp is not None:
                next_token_logits = next_token_logits / temp

            # Apply softmax
            next_token_probs = softmax(next_token_logits)

            if top_p is not None:
                sorted_next_token_probs, sorted_next_token_indices = torch.sort(next_token_probs, dim=-1, descending=True)
                cum_sum = torch.cumsum(sorted_next_token_probs, dim=-1)
                
                # In the index where cum_sum exceeds top_p, set all indices after that to 0
                index_where_exceeds = torch.nonzero(cum_sum > top_p)
                if len(index_where_exceeds) > 0:
                    index_where_exceeds = index_where_exceeds[0] # First element where cum_sum exceeds top_p
                    sorted_next_token_probs[index_where_exceeds+1:] = 0
                    sorted_next_token_probs = sorted_next_token_probs / torch.sum(sorted_next_token_probs)
                
                # Re-map back to original indices
                next_token_probs = torch.zeros_like(next_token_probs)
                next_token_probs[sorted_next_token_indices] = sorted_next_token_probs

            # Sample from the distribution
            next_token = torch.multinomial(next_token_probs, num_samples=1) # Shape (1,)
            
            # Append next_token to x
            x = torch.cat((x, next_token.unsqueeze(0)), dim=-1)
            total_tokens_generated += 1

    return x[0]

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()

    # Decoding args
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    # Paths
    parser.add_argument("--vocab_path", type=str, default="./data/TinyStoriesV2-GPT4-train_vocab_10000.json")
    parser.add_argument("--merges_path", type=str, default="./data/TinyStoriesV2-GPT4-train_merges_10000.json")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--load_path", type=str, required=True)
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

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=['<|endoftext|>'])

    # Get config from load_path
    config, run_slug = get_config_from_checkpoint(args.load_path)

    # Load model
    model = TransformerLM(config["vocab_size"], config["context_length"], config["num_layers"], config["d_model"], config["num_heads"], config["d_ff"], device=device)
    model.to(device)

    # Load checkpoint
    _ = load_checkpoint(args.load_path, model, device=device)

    # ID for output run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    OUT_DIR = os.path.join(BASE_DIR, args.out_dir)
    RUN_DIR = os.path.join(OUT_DIR, run_slug)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    
    logger.info(f"Starting decoding from checkpoint {run_slug}. Outputting to {RUN_DIR}")

    # Initialize Rope
    rope = RotaryPositionalEmbedding(config["rope_theta"], config["d_model"] // config["num_heads"], config["context_length"], device=device)

    start_time = time.perf_counter()

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_tokens], device=device)
    
    # Get EOT token ID
    eot_token_id = tokenizer.reverse_vocab[b'<|endoftext|>']

    # Decode
    generated_tokens = decode(model, x, rope, eot_token_id, args.max_length, args.temp, args.top_p, device)
    generated_text = tokenizer.decode(generated_tokens.tolist())
    
    # Save output
    output_path = os.path.join(RUN_DIR, f"decoded_text_{timestamp}.txt")
    with open(output_path, "w") as f:
        f.write(generated_text)
    
    # Save decoding config
    config_path = os.path.join(RUN_DIR, f"decoding_config_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump({
            "prompt": args.prompt,
            "max_length": args.max_length,
            "temp": args.temp,
            "top_p": args.top_p
        }, f)
    
    logger.info(f"Decoding complete. Output saved to {output_path}")