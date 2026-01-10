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

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

### (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?

Even using validation tiny-stories, this takes a lot of time, but trend is approaching 4.10/4.11


### (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

No time for this

### (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?

Trend approximates 1350 bytes/second. 825GB / 1350 bytes/s = 825 * 2^30 /1350 = 7594,62 days = 20,79 years hahaha crazy

### (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?

Token ids are positive integers, so unsigned int makes sense. The maximum id value is vocab_size. With uint16 we get 2^16 = 65536 different values, which is appropriate for a 32000 vocab size, and would actualy be fit for a vocab size up to 65536