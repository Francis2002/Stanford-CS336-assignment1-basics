from cs336_basics.bpe_utils import pretokenize_with_special_tokens
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache
import pickle

class Tokenizer():

    def __init__(self, vocab, merges, special_tokens=None):
        """
            Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
            This function should accept the following parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {value: key for key, value in vocab.items()}
        # Pre-index merge ranks for O(1) priority lookup
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Instantiates a Tokenizer from already serialized vocabulary and merge rules.
        """
        import pickle
        import json
        import os

        # Support both JSON and Pickle (Legacy)
        if str(vocab_filepath).endswith('.json'):
            with open(vocab_filepath, 'r') as f:
                json_vocab = json.load(f)
            # JSON keys are strings, values are lists of byte ints
            vocab = {int(k): bytes(v) for k, v in json_vocab.items()}
        else:
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
        
        if str(merges_filepath).endswith('.json'):
            with open(merges_filepath, 'r') as f:
                json_merges = json.load(f)
            # Convert list of lists of byte ints back to list of tuples of bytes
            merges = [(bytes(m[0]), bytes(m[1])) for m in json_merges]
        else:
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)

        return cls(vocab, merges, special_tokens=special_tokens)

    # 16384 because it is a power of 2 and it is a common cache size for LRU caches
    @lru_cache(maxsize=16384)
    def _encode_word(self, word: bytes) -> list[int]:
        """Apply BPE merges to a single pre-tokenized word.

        Starts with one token per byte, then repeatedly applies the
        highest-priority (lowest-rank) available merge until no more apply.
        """
        # Start: each byte is its own token
        ids = [self.reverse_vocab[bytes([b])] for b in word]

        while len(ids) >= 2:
            # Find all mergeable adjacent pairs and their priorities
            best_rank = None
            best_idx = None
            for i in range(len(ids) - 1):
                pair = (self.vocab[ids[i]], self.vocab[ids[i + 1]])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i

            if best_idx is None:
                break  # No more merges possible

            # Perform the best merge
            merged_bytes = self.vocab[ids[best_idx]] + self.vocab[ids[best_idx + 1]]
            merged_id = self.reverse_vocab[merged_bytes]
            ids = ids[:best_idx] + [merged_id] + ids[best_idx + 2:]

        return ids

    def encode(self, text: str, logging=False) -> list[int]:
        """
        Encode a string into a sequence of token IDs.

        Data Transformation Pipeline:
        1. str               : "Hello, <|endoftext|> world!"
        2. list[str]         : ["Hello, ", "<|endoftext|>", " world!"]                 (Split by specials)
        3. list[str]         : ["Hello", ",", " <|endoftext|>", " world", "!"]         (Pre-tokenized via Regex)
        4. bytes             : b"Hello" -> b"H", b"e", b"l", b"l", b"o"                (UTF-8 encoding per word)
        5. list[int]         : [40, 69, 76, 76, 79] -> [12053]
        
        """
        if logging:
            import time
            t0 = time.perf_counter()
            print("Started encoding...")

        # Split on special tokens and apply GPT-2 PAT regex
        pre_tokens = pretokenize_with_special_tokens(text, self.special_tokens)

        if logging:
            print(f"Pre-tokenization: {len(pre_tokens)} pre-tokens "
                  f"({time.perf_counter() - t0:.3f}s)")

        # Encode each pre-token
        token_ids = []
        for pre_token in pre_tokens:
            if self.special_tokens and pre_token in self.special_tokens:
                # Special token => single ID, no BPE merging
                token_ids.append(self.reverse_vocab[pre_token.encode('utf-8')])
            else:
                # Regular word => apply BPE merges
                token_ids.extend(self._encode_word(pre_token.encode('utf-8')))

        if logging:
            print(f"Encoding done: {len(token_ids)} token IDs "
                  f"({time.perf_counter() - t0:.3f}s)")

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
            Given an iterable of
            strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
            required for memory-eﬀicient tokenization of large files that we cannot directly load into
            memory.
        """
        for element in iterable:
            yield from self.encode(element)

    def decode(self, ids: list[int]) -> str:
        """ 
            Decode a sequence of token IDs into text.
        """
        if not ids:
            return ''
        byte_seq = bytearray()
        for token_id in ids:
            byte_seq += self.vocab[token_id]
        return byte_seq.decode('utf-8', errors='replace')

if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # Use JSON formats
    vocab_path = PROJECT_ROOT / 'data/tinystories_vocab_10000.json'
    merges_path = PROJECT_ROOT / 'data/tinystories_merges_10000.json'
    special_tokens = ['<|endoftext|>']
    my_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # Load and convert .txt to str to pass as text to encode()
    tinystories_path = PROJECT_ROOT / "data/TinyStoriesV2-GPT4-valid.txt"
    
    with open(tinystories_path, 'r') as f:
        text = f.read()

    token_ids = my_tokenizer.encode(text, logging=True)

    # Save as raw binary uint16 for performance
    token_ids_path = PROJECT_ROOT / 'data/tinystories_token_ids_10000.bin'
    np.array(token_ids, dtype=np.uint16).tofile(token_ids_path)
    print(f"Saved {len(token_ids)} tokens to {token_ids_path}")
