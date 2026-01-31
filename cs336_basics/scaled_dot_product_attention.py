import torch
import einops
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Implements the scaled dot-product attention function. The implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v). See section 3.3 for a discussion on batch-like dimensions.
    The implementation should also support an optional user-provided boolean mask of shape (seq_len,
    seq_len). The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    """
    d_k = Q.shape[-1]

    qk = einops.einsum(Q, K, 'b ... seq_q d_k, b ... seq_k d_k -> b ... seq_q seq_k')
    before_softmax = qk / (d_k ** 0.5)
    if mask is not None:
        # Use masked_fill instead of boolean indexing.
        # before_softmax[mask == 0] = -float('inf') can trigger expensive copies.
        before_softmax = before_softmax.masked_fill(~mask, float('-inf'))
    before_V = softmax(before_softmax, dim=-1)
    return einops.einsum(before_V, V, 'b ... seq_q seq_k, b ... seq_k d_v -> b ... seq_q d_v')
    
