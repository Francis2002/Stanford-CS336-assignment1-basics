from math import sqrt
from einops import einsum
import torch
from torch import nn
import einops

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
            Construct a linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__()
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = sqrt(2/(in_features + out_features))
        self.W = nn.Parameter(nn.init.trunc_normal_(W, std=std, a = -3*std, b= 3*std))

    def forward(self, x: torch.Tensor):
        """
            Apply the linear transformation to the input
        """

        return einsum(
            x, self.W,
            "... d_in, d_out d_in -> ... d_out"
        )


