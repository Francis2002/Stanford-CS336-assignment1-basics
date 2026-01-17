import torch

def gradient_clipping(parameters, max_norm, eps=1e-6):
    """
    Given a list of parameters and a maximum norm,
    modify the gradients of the parameters in-place to have a maximum total norm of max_norm.
    """
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(param.grad, ord=2) for param in parameters if param.grad is not None]), ord=2)
    if total_norm > max_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data *= (max_norm /  (total_norm + eps))

