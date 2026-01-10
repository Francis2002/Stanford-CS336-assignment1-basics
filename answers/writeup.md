# Assignment 1 - Basics

## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does chr(0) return?

It returns the null character ('\x00').

### (b) How does this character’s string representation (__repr__()) differ from its printed representation?

The string representation is just '\x00', while the printed representation is "no character", as if it was not there

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

Since the printed representation is essentially nothing, and repre is '\x00', the behaviour is expected, since we see the '\x00' in the repr, but in the printed version, instead of '\x00' we see no character (which is different from the space character ' ', which is not nothing).

## Problem (unicode2): Unicode Encodings (3 points)

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

UTF-8 represents the test strings with less bytes, and the others have a lot of zeros.

### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

The function tries to decode each individual byte as a charqcter, which is not how Unicode works. A character may be represented by more than one byte. For example the char "ç" throws UnicodeDecodeError

### (c) Give a two byte sequence that does not decode to any Unicode character(s).

b'\x14\xa2' doesn't decode to anything. 

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?

On the validation set:

Longest token (bytes): b' accomplishment'
Length in bytes: 15
Longest token as string:  accomplishment

### (b) Profile your code. What part of the tokenizer training process takes the most time?

Updating the counts of token pairs, because this involves adding new token pairs, and updating counts of the overlapped tokens

### (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

On validation set:

Longest token (bytes): b'----------------------------------------------------------------'
Length in bytes: 64
Longest token as string: ----------------------------------------------------------------