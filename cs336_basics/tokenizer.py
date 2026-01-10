from cs336_basics.bpe_utils import pre_tokenize_chunks_with_specials, split_with_specials
from collections.abc import Iterable, Iterator
from .tokenizer_BPE import train_bpe
import regex as re

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

        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
            Encode an input text into a sequence of token IDs.
        """

        # Split text into chunks with specials in between like ['hello, I', '<special>', 'would not', ...]
        chunks = split_with_specials(text, self.special_tokens)
        chunks = pre_tokenize_chunks_with_specials(chunks, self.special_tokens)

        sequence_of_token_ids = []
        for chunk in chunks:
            for pre_token in chunk:
                if self.special_tokens and pre_token in self.special_tokens:
                    sequence_of_token_ids.append(self.reverse_vocab[pre_token.encode('utf-8')])
                    continue
                word_token_ids = [self.reverse_vocab[bytes([b])] for b in pre_token.encode('utf-8')] 
                # Here we have something like [self.reverse_vocab[b't'], self.reverse_vocab[b'h'], self.reverse_vocab[b'e'], ...]

                for pair_bytes in self.merges:
                    id1 = self.reverse_vocab[pair_bytes[0]] # Here we have like 32
                    id2 = self.reverse_vocab[pair_bytes[1]] # here we have like 45
                    new_id = self.reverse_vocab[pair_bytes[0] + pair_bytes[1]] # Here we have something like self.reverse_vocab[b'th'] = 276

                    new_ids = []
                    i = 0
                    while i < len(word_token_ids):
                        if i < len(word_token_ids) - 1 and word_token_ids[i] == id1 and word_token_ids[i+1] == id2:
                            new_ids.append(new_id)
                            i += 2
                        else:
                            new_ids.append(word_token_ids[i])
                            i += 1
                    word_token_ids = new_ids
                
                sequence_of_token_ids.extend(word_token_ids)

        return sequence_of_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
            Given an iterable of
            strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
            required for memory-eﬀicient tokenization of large files that we cannot directly load into
            memory.
        """

        for element in iterable:
            sequence_of_token_ids = self.encode(element)
            for token_id in sequence_of_token_ids:
                yield token_id

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