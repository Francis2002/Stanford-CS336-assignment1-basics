import regex as re
from cs336_basics.bpe_utils import PAT, CHUNK_SIZE, init_vocab, build_special_token_pattern, find_safe_chunk_boundary, merge_pair_in_word

def count_pretokens_in_fragment(text, special_pat, compiled_pat, counter):
    """Split a text fragment on special tokens, apply PAT regex, update counter."""
    fragments = special_pat.split(text) if special_pat else [text]
    for fragment in fragments:
        for m in compiled_pat.finditer(fragment):
            token = m.group(0)
            if token in counter:
                counter[token] += 1
            else:
                counter[token] = 1


def stream_pretokenize_counts(input_path, special_tokens, use_profiler=False, _timer=None):
    """Stream the file in 1 MB chunks, pretokenize on the fly, return word counts.

    Never loads the entire file into memory, only accumulates the dict of
    unique pre-tokens

    Returns:
        dict[str, int]: mapping from pre-token string to its count in the corpus.
    """
    special_pat = build_special_token_pattern(special_tokens)
    compiled_pat = re.compile(PAT)

    counts = {}  # {pre_token_string: count}
    with open(input_path, "r", encoding="utf-8") as f:
        leftover = ""
        bytes_read = 0

        while True:
            raw = f.read(CHUNK_SIZE)
            if not raw:
                # Flush any remaining leftover at end of file
                if leftover:
                    count_pretokens_in_fragment(leftover, special_pat, compiled_pat, counts)
                break

            text = leftover + raw
            bytes_read += len(raw)

            # Find a safe place to split so no PAT token spans across chunks
            split_pos = find_safe_chunk_boundary(text)
            if split_pos == -1:
                leftover = text  # No safe boundary, accumulate more
                continue

            process, leftover = text[:split_pos], text[split_pos:]
            count_pretokens_in_fragment(process, special_pat, compiled_pat, counts)

            # Progress logging every ~100 MB
            if use_profiler and bytes_read % (100 * CHUNK_SIZE) == 0:
                mb = bytes_read / (1024 * 1024)
                print(f"  Streamed {mb:.0f} MB, {len(counts)} unique pre-tokens "
                      f"({_timer():.1f}s)")

    return counts

def build_pair_index(pre_tok_dict: dict[tuple, int]):
    """
    Builds an inverted index of byte pairs to avoid O(V) scans during merges.

    Returns:
        pair_counts:    Maps a byte pair to its local frequency: {(id_a, id_b): total_count}
        pair_to_words:  Maps a byte pair to the set of words (tuples) 
                        that contain it: {(id_a, id_b): set of word tuples containing this pair}.
                        When a pair is merged, we only update words in this set
    """
    pair_counts = {}
    pair_to_words = {}  # inverted index: pair -> set of word tuples
    for word, word_count in pre_tok_dict.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + word_count
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)
    return pair_counts, pair_to_words

def find_best_pair(pair_counts: dict[tuple, int], vocab: dict[int, bytes]) -> tuple:
    """Return the pair with the highest count.

    Ties are broken by lexicographic order of the actual byte values of the
    two tokens (NOT by token ID -- those are different orderings!).
    """
    max_count = max(pair_counts.values())
    candidates = [pair for pair, count in pair_counts.items() if count == max_count]
    return max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))

def apply_merge(chosen_pair, new_id, pre_tok_dict, pair_counts, pair_to_words):
    """Merge chosen_pair into new_id only touching affected words.

    Updates pair_counts in-place and returns a new pre_tok_dict.
    Three sub-steps for each word that contains the pair:
    1. Remove old pair counts contributed by this word
    2. Replace the pair with new_id in the word's byte-tuple
    3. Add new pair counts from the updated word

    Uses the pair_to_words inverted index to skip the vast majority of words that don't contain the pair.

    Modifies pre_tok_dict, pair_counts, and pair_to_words IN-PLACE.
    """
    # Pop the set of words that contain this exact adjacent pair
    affected_words = pair_to_words.pop(chosen_pair, set())
    if chosen_pair in pair_counts:
        del pair_counts[chosen_pair]

    for word in affected_words:
        word_count = pre_tok_dict.pop(word)

        # Remove this word's contribution to pair counts & inverted index
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            if p == chosen_pair:
                continue  # already removed above
            if p in pair_counts:
                pair_counts[p] -= word_count
                if pair_counts[p] <= 0:
                    del pair_counts[p]
            if p in pair_to_words:
                pair_to_words[p].discard(word)
                if not pair_to_words[p]:
                    del pair_to_words[p]

        # Merge the pair in the word
        new_word = merge_pair_in_word(word, chosen_pair, new_id)
        pre_tok_dict[new_word] = word_count

        # Add new pair counts and inverted index entries
        for i in range(len(new_word) - 1):
            p = (new_word[i], new_word[i + 1])
            pair_counts[p] = pair_counts.get(p, 0) + word_count
            if p not in pair_to_words:
                pair_to_words[p] = set()
            pair_to_words[p].add(new_word)

def train_bpe(input_path, vocab_size: int, special_tokens: list[str], use_profiler=False):
    """Train a BPE tokenizer on a text corpus.

    Streams the corpus from disk (memory-safe for files of any size), then
    merges the most frequent byte-pair until vocab_size is reached.

    Args:
        input_path: Path to the training corpus (.txt file).
        vocab_size: Desired total vocabulary size (including 256 bytes + special tokens).
        special_tokens: Strings that should never be split (e.g. '<|endoftext|>').
        use_profiler: If True, print timing information during training.

    Returns:
        (vocab, merges) where:
          vocab:  dict[int, bytes] -- token ID -> token bytes
          merges: list[tuple[bytes, bytes]] -- ordered list of merges
    """
    if use_profiler:
        import time
        start_time = time.perf_counter()
        timer = lambda: time.perf_counter() - start_time
    else:
        timer = None

    # Initialize vocabulary
    vocab = init_vocab(special_tokens)
    merges = []

    # Stream and count pre-tokens 
    string_counts = stream_pretokenize_counts(
        input_path, special_tokens, use_profiler, timer
    )
    if use_profiler:
        print(f"Streaming pre-tokenization done: {timer():.4f}s "
              f"({len(string_counts)} unique pre-tokens)\n")

    # Map strings/words to tuples of raw bytes
    # pre_tok_dict looks like { (b'h', b'e', b'l', b'l', b'o'): 150, ... }
    #                       = {(104, 101, 108, 108 111): 150}
    pre_tok_dict = {
        tuple(token.encode("utf-8")): count
        for token, count in string_counts.items()
    }

    del string_counts  # Free the string-keyed dict

    if use_profiler:
        print(f"Converted to byte-tuple repr: {timer():.4f}s\n")

    # Build initial pair-count index + inverted index (pair -> words)
    pair_counts, pair_to_words = build_pair_index(pre_tok_dict)

    if use_profiler:
        print(f"Initial pair index built: {timer():.4f}s  "
              f"({len(pair_counts)} unique pairs, "
              f"{sum(len(s) for s in pair_to_words.values())} inverted-index entries)\n")

    # Iteratively merge until vocab is full
    while len(vocab) < vocab_size:
        if use_profiler and len(merges) % 500 == 0:
            print(f"\nMerge #{len(merges):>5d}  |  vocab {len(vocab)}  |  "
                  f"{len(pair_counts)} active pairs  |  "
                  f"{len(pre_tok_dict)} unique words  |  elapsed: {timer():.1f}s")

        # Find the best pair
        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        chosen_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))

        # Record the merge (as bytes, not IDs)
        merges.append((vocab[chosen_pair[0]], vocab[chosen_pair[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[chosen_pair[0]] + vocab[chosen_pair[1]]

        # Apply the merge only to affected words (via inverted index)
        apply_merge(chosen_pair, new_id, pre_tok_dict, pair_counts, pair_to_words)

    return vocab, merges

def train_bpe_tinystories():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    tinystories_path = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-train.txt"
    return train_bpe(tinystories_path, vocab_size=10000,
                     special_tokens=['<|endoftext|>'], use_profiler=True)


def train_bpe_expts_owt():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    owt_path = PROJECT_ROOT / "data/owt_train.txt"
    return train_bpe(owt_path, vocab_size=32000,
                     special_tokens=['<|endoftext|>'], use_profiler=True)

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser("Train BPE tokenizer")
    parser.add_argument("--dataset", type=str, default="owt", help="owt or tiny")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if args.dataset == "tiny":
        vocab, merges_list = train_bpe_tinystories()
        vocab_path = PROJECT_ROOT / "data/tinystories_vocab_10000.json"
        merges_path = PROJECT_ROOT / "data/tinystories_merges_10000.json"

    elif args.dataset == "owt":
        vocab, merges_list = train_bpe_expts_owt()
        vocab_path = PROJECT_ROOT / "data/owt_train_vocab_32000.json"
        merges_path = PROJECT_ROOT / "data/owt_train_merges_32000.json"

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Print longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token (bytes): {longest_token}")
    print(f"Length in bytes: {len(longest_token)}")
    print(f"Longest token as string: {longest_token.decode('utf-8', errors='replace')}")

    # Save vocab and merges as JSON
    json_vocab = {str(k): list(v) for k, v in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(json_vocab, f, indent=2)

    json_merges = [[list(m[0]), list(m[1])] for m in merges_list]
    with open(merges_path, "w") as f:
        json.dump(json_merges, f, indent=2)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
