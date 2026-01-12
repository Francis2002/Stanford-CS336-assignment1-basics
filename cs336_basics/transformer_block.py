import torch
from torch import nn
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer
        """
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.attention = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, rope=None, token_positions=None):
        x = x + self.attention(self.norm1(x), rope, token_positions) # (Norm -> Attn) + residual
        x = x + self.feed_forward(self.norm2(x)) # (Norm -> FF) + residual
        return x
        