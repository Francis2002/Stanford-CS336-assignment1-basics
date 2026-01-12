import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Takes two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. 
    The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution.
    """
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    