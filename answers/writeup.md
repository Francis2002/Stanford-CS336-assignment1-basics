# Assignment 1 - Basics

## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does `chr(0)` return?

It returns the null character (`'\x00'`).

### (b) How does this character’s string representation (`__repr__()`) differ from its printed representation?

The string representation is just `'\x00'`, while the printed representation is "no character", as if it was not there.

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

Since the printed representation is essentially nothing, and its `repr()` is `'\x00'`, the behavior is as expected: we see the `'\x00'` in the `repr`, but in the printed version, instead of `'\x00'` we see no character (which is different from the space character `' '`, which is not nothing).

---

## Problem (unicode2): Unicode Encodings (3 points)

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

UTF-8 represents the test strings with fewer bytes, and the others have many zeros (padding).

### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

The function tries to decode each individual byte as a character, which is not how Unicode works. A character may be represented by more than one byte. For example, the character `"ç"` throws a `UnicodeDecodeError` if decoded byte-by-byte.

### (c) Give a two byte sequence that does not decode to any Unicode character(s).

`b'\x14\xa2'` doesn't decode to anything.

---

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?

On the validation set:

- **Longest token (bytes):** `b' accomplishment'`
- **Length in bytes:** 15
- **Longest token as string:** ` accomplishment`

### (b) Profile your code. What part of the tokenizer training process takes the most time?

Updating the counts of token pairs, because this involves searching for and merging pairs, and updating counts of the affected tokens.

---

## Problem (train_bpe_openwebtext): BPE Training on OpenWebText

### (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

On validation set:

- **Longest token (bytes):** `b'----------------------------------------------------------------'`
- **Length in bytes:** 64
- **Longest token as string:** `----------------------------------------------------------------`

---

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

### (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?

Even using the validation set for TinyStories, this takes a lot of time, but the trend is approaching **4.10 / 4.11**.

### (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

*No time for this analysis.*

### (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?

- **Trend approximates:** 1,350 bytes/second
- **Calculation:**
  > 825 GB / 1,350 bytes/s = (825 * 2^30) / 1,350 ≈ 656,013,101 seconds
  > 656,013,101 / 86,400 ≈ 7,592 days
  > 7,592 / 365.25 ≈ **20.79 years**

### (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype `uint16`. Why is `uint16` an appropriate choice?

Token IDs are non-negative integers, so an unsigned integer (`uint`) makes sense. The maximum ID value is the `vocab_size`. With `uint16`, we get $2^{16} = 65,536$ different values, which is appropriate for a 32,000 vocab size, and would actually accommodate a vocab size up to 65,536.

---

## Problem (transformer_accounting): Transformer LM resource accounting (5 points)

### Consider GPT-2 XL, which has the following configuration:
- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 6,400

### (a) Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

1 head:

Total Params = d_model * vocab_size + d_model + num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model) + vocab_size * d_model = 
50,257 * 1,600 + 1,600 + 48 * (4 * 1,600 * 1,600 + 3 * 1,600 * 6,400 + 2 * 1,600) + 50,257 * 1,600 = 80,411,200 + 1,600 + 48 * (10,240,000 + 30,720,000 + 3,200) + 80,411,200 = 80,411,200 + 1,600 + 48 * 40,963,200 + 80,411,200 = 80,411,200 + 1,600 + 1,966,233,600 + 80,411,200 = 2,127,057,600

Memory Required = Total Params * 4 bytes = 7.9239GB

### (b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.

Matrix Multiplies:
- Linear Head: (vocab_size x d_model) * d_model
- Per Layer:
    - Feed-Forward: 2 * (d_ff x d_model) * d_model + (d_model x d_ff) * d_ff
    - MHA:
        - Projections: 4 * (d_model x d_model) * d_model
        - Self-Attention: (context_length x d_model) * (d_model x context_length) + (context_length x context_length) * (context_length x d_model)

Total FLOPs = vocab_size * d_model + num_layers * (3 * d_ff * d_model + 4 * d_model * d_model + 2 * context_length * d_model * context_length) = 50,257 * 1,600 + 48 * (3 * 6,400 * 1,600 + 4 * 1,600 * 1,600 + 2 * 1,024 * 1,600 * 1,024) = 80,411,200 + 48 * (30,720,000 + 10,240,000 + 3,355,443,200) = 80,411,200 + 48 * 3,396,403,200 = 80,411,200 + 163,027,353,600 = 163,107,764,800 FLOPs

### (c) Based on your analysis above, which parts of the model require the most FLOPs?

Self-Attention requires the most FLOPs.

### (d) Repeat your analysis with GPT-2 small (12 layers, 768 `d_model`, 12 heads), GPT-2 medium (24 layers, 1024 `d_model`, 16 heads), and GPT-2 large (36 layers, 1280 `d_model`, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

*Skipped.*

### (e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

This will increase the FLOPs of the MHA component by a factor of 16,384 / 1,024 = 16 squared, or 256. The total FLOPs will approximately increase by a factor of 256.

## Problem (learning_rate_tuning): Tuning the learning rate (1 point)

Obvious. Already know from past. Skip.

## Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)

### (a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.

For simplicity, when calculating memory usage of activations, consider only the following components:
• Transformer block
– RMSNorm(s)
– Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax, weighted sum of values, output projection.
– Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
• final RMSNorm
• output embedding
• cross-entropy on logits

1 head:

Parameters: d_model * vocab_size + d_model + num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model) + vocab_size * d_model

... ok we know how much its going to be.

### (b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

skip

### (c) How many FLOPs does running one step of AdamW take?

skip

### (d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022], assume that the backward pass has twice the FLOPs of the forward pass.

skip