import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# How many bytes to read at a time when streaming large files
CHUNK_SIZE = 1024 * 1024  # 1 MB

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Create base vocabulary: 256 single-byte tokens + special tokens."""
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab

def build_special_token_pattern(special_tokens: list[str]) -> re.Pattern | None:
    """Compile a regex that matches any of the special tokens.

    Sorts by length descending so longest matches win (e.g. '<|endoftext|>'
    is matched before '<|end' if both were special tokens).
    Returns None if special_tokens is empty/None.
    """
    if not special_tokens:
        return None
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    return re.compile("|".join(re.escape(t) for t in sorted_specials))


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens, DISCARDING the special tokens themselves.

    Used during BPE training: special tokens are vocabulary entries but we
    don't want to learn merges across/inside them.
    """
    if not special_tokens:
        return [text]
    pat = build_special_token_pattern(special_tokens)
    return pat.split(text)


def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens, KEEPING them as separate list elements.

    Used during encoding: we need to emit the special token's ID and also
    tokenize the text between specials.
    """
    if not special_tokens:
        return [text]
    pat = build_special_token_pattern(special_tokens)

    pattern_with_capture = f"({pat.pattern})"
    return [c for c in re.split(pattern_with_capture, text) if c]

def pretokenize(text: str, compiled_pat: re.Pattern | None = None) -> list[str]:
    """Apply GPT-2 PAT regex to split text into pre-tokens.

    Returns a flat list of matched strings.
    """
    if compiled_pat is None:
        compiled_pat = re.compile(PAT)
    return [m.group(0) for m in compiled_pat.finditer(text)]


def pretokenize_with_special_tokens(
    text: str,
    special_tokens: list[str],
    compiled_pat: re.Pattern | None = None,
) -> list[str]:
    """Split text on special tokens, then pretokenize non-special chunks.

    Returns a flat list where special tokens appear as-is among the
    pretokenized words. Used in Tokenizer.encode().
    """
    if compiled_pat is None:
        compiled_pat = re.compile(PAT)

    chunks = split_with_special_tokens(text, special_tokens)
    result = []
    special_set = set(special_tokens) if special_tokens else set()
    for chunk in chunks:
        if chunk in special_set:
            result.append(chunk)  # Keep special token as-is
        else:
            result.extend(pretokenize(chunk, compiled_pat))
    return result

def merge_pair_in_word(word: tuple, pair: tuple, new_id: int) -> tuple:
    """Replace all adjacent occurrences of `pair` in `word` with `new_id`."""
    result = []
    i = 0
    while i < len(word):
        # Check if current + next element match the pair
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(new_id)
            i += 2  # Skip both elements of the pair
        else:
            result.append(word[i])
            i += 1
    return tuple(result)

def find_safe_chunk_boundary(text: str) -> int:
    """Find a position to split `text` so that no PAT token is broken.

    Strategy:
    1. Find the last newline (guarantees we don't split inside a word).
    2. Backtrack past any whitespace before (and including) the newline.
    3. The split point is the first non-whitespace char.

    Returns:
        Position to split at.
        Returns -1 if no safe boundary exists (caller should accumulate more).
    """
    last_nl = text.rfind('\n')
    if last_nl == -1:
        return -1  # No newline at all

    # Walk backwards past whitespace (including the \n itself)
    split_pos = last_nl
    while split_pos > 0 and text[split_pos - 1].isspace():
        split_pos -= 1

    if split_pos == 0:
        return -1  # Everything before the \n is whitespace

    return split_pos