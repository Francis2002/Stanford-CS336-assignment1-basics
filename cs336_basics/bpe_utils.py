import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(text: str, special_tokens: list[str]):
    # 1. Base 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}

    # 2. Add special tokens immediately after bytes
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    
    chunks = split_into_chunks(text, special_tokens)

    return chunks, vocab

def merge_tokens_for_pre_token_dict(byte_tuple, pair, new_id):
    new_tuple = []
    i = 0
    while i < len(byte_tuple):
        # Check if current and next element match the pair
        if i < len(byte_tuple) - 1 and byte_tuple[i] == pair[0] and byte_tuple[i+1] == pair[1]:
            new_tuple.append(new_id)
            i += 2  # Skip both elements of the pair
        else:
            new_tuple.append(byte_tuple[i])
            i += 1
    return tuple(new_tuple)

def split_into_chunks(text, special_tokens):
    if special_tokens:
        safe_special_tokens = map(re.escape, special_tokens)
        chunks = re.split("|".join(safe_special_tokens), text)
    else:
        chunks = [text]
    return chunks

# In bpe_utils.py or tokenizer.py
def split_with_specials(text, special_tokens):
    if not special_tokens:
        return [text]
    
    # Sort by length (descending) so longest matches win
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    # Use capturing parentheses () to keep the delimiters in the list
    # Use '|'.join with re.escape to handle characters like '|' or '['
    pattern = f"({'|'.join(re.escape(t) for t in sorted_specials)})"
    
    # Filter out empty strings result from split
    return [c for c in re.split(pattern, text) if c]

def pre_tokenize_chunks(chunks):
    # Pre-tokenize chunks
    per_chunk_tokens_after_pre_tokenization = [] # This will have for chunk i, ['word1', 'word2', ...] according to PAT
    for chunk in chunks:
        tokens_after_pre_tokenization = [match.group(0) for match in re.finditer(PAT, chunk)]
        per_chunk_tokens_after_pre_tokenization.append(tokens_after_pre_tokenization)
    return per_chunk_tokens_after_pre_tokenization

def pre_tokenize_chunks_with_specials(chunks, special_tokens):
    # Pre-tokenize chunks
    per_chunk_tokens_after_pre_tokenization = [] # This will have for chunk i, ['word1', 'word2', ...] according to PAT
    for chunk in chunks:
        if special_tokens and chunk in special_tokens:
            per_chunk_tokens_after_pre_tokenization.append([chunk])
            continue
        tokens_after_pre_tokenization = [match.group(0) for match in re.finditer(PAT, chunk)]
        per_chunk_tokens_after_pre_tokenization.append(tokens_after_pre_tokenization)
    return per_chunk_tokens_after_pre_tokenization