from torch import nn, sqrt
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x_sq = torch.mean(x**2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms_x_sq + self.eps)
        result = x_norm * self.g

        # Return the result in the original dtype
        return result.to(in_dtype)