import cProfile
import __main__
import regex as re
from cs336_basics.bpe_utils import init_vocab, merge_tokens_for_pre_token_dict, pre_tokenize_chunks
import cProfile
import pstats

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], use_profiler=False):
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    if use_profiler:
        import time
        start_time = time.perf_counter()

    # Get text from path
    with input_path.open("r", encoding="utf-8") as f:
        text = f.read()

    if use_profiler:
        print(f"File read: {time.perf_counter() - start_time:.4f}s\n")

    # Init vocab with list of special tokens and the 256 bytes values and merges
    chunks, vocab = init_vocab(text, special_tokens)
    merges = []

    if use_profiler:
        print(f"Vocab init done: {time.perf_counter() - start_time:.4f}s\n")

    # Pre-tokenize chunks
    per_chunk_tokens_after_pre_tokenization = pre_tokenize_chunks(chunks) # This will have for chunk i, ['word1', 'word2', ...] according to PAT

    if use_profiler:
        print(f"Initial Pretokenization done: {time.perf_counter() - start_time:.4f}s\n")
    
    # Count
    pre_tokenization_with_count = {} # This will have {low: 5, lower: 2, ...}
    for pre_tokenized_chunk in per_chunk_tokens_after_pre_tokenization:
        for token in pre_tokenized_chunk:
            if token in pre_tokenization_with_count.keys():
                pre_tokenization_with_count[token] += 1  
            else:
                pre_tokenization_with_count[token] = 1

    if use_profiler:
        print(f"Pre tokenization count complete: {time.perf_counter() - start_time:.4f}s\n")

    pre_tokenization_dict = {}
    # Represent this as dict[tuple[bytes], int], like {(l, o, w): 5, ...}
    for token, count in pre_tokenization_with_count.items():
        byte_tuple = tuple(token.encode('utf-8'))
        pre_tokenization_dict[byte_tuple] = count

    if use_profiler:
        print(f"Changed representation to get dict[tuple[bytes], int]: {time.perf_counter() - start_time:.4f}s\n")

    # Create index of counts for byte pairs, so that we dont iterate through all every time
    byte_pair_counts = {}
    for byte_tuple, word_count in pre_tokenization_dict.items():
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i+1])
            if pair in byte_pair_counts.keys():
                byte_pair_counts[pair] += word_count
            else:
                byte_pair_counts[pair] = word_count

    if use_profiler:
        print(f"Initial index created. About to start loop: {time.perf_counter() - start_time:.4f}s\n")

    # until len(vocab) == vocab_size, apply merge
    while len(vocab) < vocab_size:
        # find the highest count on byte_pair_counts, and merge those
        # Merge means putting vocab[len(vocab)] = byte_pair_with_max_count
        # Then adding the tuple of elements of vocab to merges
        # Then find overlapping byte pairs and update the counts, including the new pairs involving the new token
        if use_profiler:
            print_condition = len(merges) % 500 == 0

        if use_profiler and print_condition:
            print(f"\nNumber of merges: {len(merges)}")
            print(f"Current elapsed time: {time.perf_counter() - start_time:.4f}s")

        # 1) Find the best pair
        max_count = max(byte_pair_counts.values())
        byte_pairs_with_max_count = [pair for pair, count in byte_pair_counts.items() if count == max_count]
        chosen_pair = max(
            byte_pairs_with_max_count, 
            key=lambda p: (vocab[p[0]], vocab[p[1]]) # Tie-break by comparing the actual bytes of the tokens
        ) # Lexicographic order is different from id order! Doing max(byte_pairs_with_max_count) would be wrong

        if use_profiler and print_condition:
            print(f"Pair found: {time.perf_counter() - start_time:.4f}s")

        # 2) Record merge
        merges.append((vocab[chosen_pair[0]], vocab[chosen_pair[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[chosen_pair[0]] + vocab[chosen_pair[1]] # So that this has bytes type

        if use_profiler and print_condition:
            print(f"Record Merged: {time.perf_counter() - start_time:.4f}s")

        # 3) Update counts
        del byte_pair_counts[chosen_pair]
        new_pre_tokenization_dict = {}

        for byte_tuple, word_count in pre_tokenization_dict.items():
            if chosen_pair[0] in byte_tuple and chosen_pair[1] in byte_tuple: # Check if word needs merging
                # 1) Remove all old pairs from this word from the index
                for i in range(len(byte_tuple) - 1):
                    old_pair = (byte_tuple[i], byte_tuple[i+1])
                    if old_pair in byte_pair_counts: # Was not deleted at the start
                        byte_pair_counts[old_pair] -= word_count
                        if byte_pair_counts[old_pair] <= 0: # Extra optimization
                            del byte_pair_counts[old_pair]

                # 2) Merge the word in the pre_tokenizer_dict
                updated_tuple = merge_tokens_for_pre_token_dict(byte_tuple, chosen_pair, new_id)
                new_pre_tokenization_dict[updated_tuple] = word_count

                # 3) Add all new pairs becasue of the new word
                for i in range(len(updated_tuple) - 1):
                    new_pair = (updated_tuple[i], updated_tuple[i+1])
                    byte_pair_counts[new_pair] = byte_pair_counts.get(new_pair, 0) + word_count
            
            else:
                new_pre_tokenization_dict[byte_tuple] = word_count # Just forward the old count to the new dict

        pre_tokenization_dict = new_pre_tokenization_dict

        if use_profiler and print_condition:
            print(f"Counts updated: {time.perf_counter() - start_time:.4f}s")

    return vocab, merges

def train_bpe_tinystories():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    tinystories_path = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_characters = ['<|endoftext|>']
    return train_bpe(tinystories_path, vocab_size, special_characters, use_profiler=True)

def train_bpe_expts_owt():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    tinystories_path = PROJECT_ROOT / "data/owt_valid.txt"
    vocab_size = 32000
    special_characters = ['<|endoftext|>']
    return train_bpe(tinystories_path, vocab_size, special_characters, use_profiler=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Which dataset to use?')
    parser.add_argument('--dataset', type=str, default='owt', help='owt or tiny')
    args = parser.parse_args()

    if args.dataset == 'tiny':
        tiny_stories_vocab_10000, tiny_stories_merges_10000 = train_bpe_tinystories()

        longest_token = max(tiny_stories_vocab_10000.values(), key=len)
        print(f"Longest token (bytes): {longest_token}")
        print(f"Length in bytes: {len(longest_token)}")
        print(f"Longest token as string: {longest_token.decode('utf-8', errors='replace')}")

        # Save vocab and merges in JSON format
        import json
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        
        vocab_path = PROJECT_ROOT / 'data/tinystories_vocab_10000.json'
        json_vocab = {str(k): list(v) for k, v in tiny_stories_vocab_10000.items()}
        with open(vocab_path, 'w') as f:
            json.dump(json_vocab, f, indent=2)
            
        merges_path = PROJECT_ROOT / 'data/tinystories_merges_10000.json'
        json_merges = [[list(m[0]), list(m[1])] for m in tiny_stories_merges_10000]
        with open(merges_path, 'w') as f:
            json.dump(json_merges, f, indent=2)
        
    elif args.dataset == 'owt':
        owt_vocab_32000, owt_merges_32000 = train_bpe_expts_owt()

        longest_token = max(owt_vocab_32000.values(), key=len)
        print(f"Longest token (bytes): {longest_token}")
        print(f"Length in bytes: {len(longest_token)}")
        print(f"Longest token as string: {longest_token.decode('utf-8', errors='replace')}")

        # Save vocab and merges in JSON format
        import json
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        
        vocab_path = PROJECT_ROOT / 'data/owt_vocab_32000.json'
        json_vocab = {str(k): list(v) for k, v in owt_vocab_32000.items()}
        with open(vocab_path, 'w') as f:
            json.dump(json_vocab, f, indent=2)
            
        merges_path = PROJECT_ROOT / 'data/owt_merges_32000.json'
        json_merges = [[list(m[0]), list(m[1])] for m in owt_merges_32000]
        with open(merges_path, 'w') as f:
            json.dump(json_merges, f, indent=2)
