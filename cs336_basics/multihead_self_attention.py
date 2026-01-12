import torch 
from torch import nn
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear
import einops

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, device=None, dtype=None):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.device = device
        self.dtype = dtype

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, rope=None, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, 'b ... s (h d) -> b ... h s d', h=self.n_heads) # Each head should get something of shape (..., seq_len, d_k)
        k = einops.rearrange(k, 'b ... s (h d) -> b ... h s d', h=self.n_heads)
        v = einops.rearrange(v, 'b ... s (h d) -> b ... h s d', h=self.n_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(q.shape[-2], device=q.device) # q.shape[-2] is the sequence length
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # this should look like [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        mask = ~torch.ones(q.shape[:-2] + (q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device).triu(diagonal=1)

        attention = scaled_dot_product_attention(q, k, v, mask)

        # Concat all
        attention = einops.rearrange(attention, 'b ... h s d -> b ... s (h d)', h=self.n_heads)

        return self.out_proj(attention)


        

    