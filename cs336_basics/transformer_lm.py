import torch
from torch import nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.softmax import softmax

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, device=None, dtype=None):
        """
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
        embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of
        the position embedding matrix.
        num_layers: int The number of Transformer blocks to use
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = context_length
        self.vocab_size = vocab_size

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x, rope=None, token_positions=None):
        x = self.token_embedding(x)        
        for block in self.transformer_blocks:
            x = block(x, rope, token_positions)
        x = self.norm(x)
        x = self.linear(x)
        return x