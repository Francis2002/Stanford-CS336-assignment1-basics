from cs336_basics.bpe_utils import pre_tokenize_chunks_with_specials, split_with_specials
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
            Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            (in the same format that your BPE training code output) and (optionally) a list of special
            tokens. This method should accept the following additional parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
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
        # Initial state: list of single-byte token IDs
        word_token_ids = [self.reverse_vocab[bytes([b])] for b in word]
        
        while len(word_token_ids) >= 2:
            # Find all possible merges and their ranks
            pairs = []
            for i in range(len(word_token_ids) - 1):
                pair = (self.vocab[word_token_ids[i]], self.vocab[word_token_ids[i+1]])
                if pair in self.merge_ranks:
                    pairs.append((self.merge_ranks[pair], i))
            
            if not pairs:
                break
                
            # Perform the highest priority merge (lowest rank)
            best_rank, best_index = min(pairs)
            
            p1 = self.vocab[word_token_ids[best_index]]
            p2 = self.vocab[word_token_ids[best_index + 1]]
            new_id = self.reverse_vocab[p1 + p2]
            
            # Create new list with merged token
            new_ids = word_token_ids[:best_index] + [new_id] + word_token_ids[best_index + 2:]
            word_token_ids = new_ids
            
        return word_token_ids

    def encode(self, text: str, logging=False) -> list[int]:
        """
            Encode an input text into a sequence of token IDs.
        """

        if logging:
            import time
            start_time = time.perf_counter()
            print("Started encoding")

        # Split text into chunks with specials in between like ['hello, I', '<special>', 'would not', ...]
        chunks = split_with_specials(text, self.special_tokens)
        chunks = pre_tokenize_chunks_with_specials(chunks, self.special_tokens)

        if logging:
            print("Pretokenization complete")
            total_pre_tokens = sum(
                len(chunk) if isinstance(chunk, Sequence) else 1
                for chunk in chunks
            )
            print(f"Number of pre_tokens: {total_pre_tokens}")
            pre_token_cnt = -1

        sequence_of_token_ids = []
        for chunk in chunks:
            for pre_token in chunk:
                if self.special_tokens and pre_token in self.special_tokens:
                    sequence_of_token_ids.append(self.reverse_vocab[pre_token.encode('utf-8')])
                    continue
                
                # Use cached word-level encoding
                word_bytes = pre_token.encode('utf-8')
                sequence_of_token_ids.extend(self._encode_word(word_bytes))

        return sequence_of_token_ids

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

        byte_sequence = bytearray()
        for token_id in ids:
            if byte_sequence is None:
                byte_sequence = self.vocab[token_id] # First case
            else:
                byte_sequence += self.vocab[token_id]
        return byte_sequence.decode('utf-8', errors='replace')

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
