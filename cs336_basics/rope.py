from torch import nn
import torch
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct
        the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """

        super().__init__()

        indexes = torch.arange(max_seq_len, device=device).unsqueeze(-1) # Unsqueeze makes shape of this [seq_len, 1]
        theta_exps = torch.arange(0, d_k, 2, device=device) / d_k # Shape of this is [d_k // 2]

        inv_thetas = 1.0 / (theta ** theta_exps)
        final_thetas = indexes * inv_thetas # Shape of this is [seq_len, d_k // 2]

        cosines = torch.cos(final_thetas)
        sines = torch.sin(final_thetas)

        self.register_buffer('cosines', cosines, persistent=False)
        self.register_buffer('sines', sines, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """

        cos_pos = self.cosines[token_positions] # Shapes here are [..., seq_len]
        sin_pos = self.sines[token_positions]

        x_2ks = x[..., ::2] # Shape here are [..., seq_len, d_k // 2]
        x_2ks1 = x[..., 1::2]

        rotated_2ks = x_2ks * cos_pos - x_2ks1 * sin_pos # Shapes here are [..., seq_len, d_k // 2]
        rotated_2ks1 = x_2ks * sin_pos + x_2ks1 * cos_pos

        # We have to reintertwine the 2ks and 2k1s
        # Stack then rearrange to interleave
        result = rearrange(torch.stack([rotated_2ks, rotated_2ks1], dim=-1), '... n i -> ... (n i)')

        return result


