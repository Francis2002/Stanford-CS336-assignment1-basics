import torch

def cross_entropy(logits, targets):
    """
    Args:
        logits (torch.Tensor): Logits with shape (..., vocab_size)
        targets (torch.Tensor): Target indices with shape (...)
    """
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    sum_exp = torch.exp(shifted_logits).sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    log_probs = shifted_logits - log_sum_exp # Here we have the log probabilities of each token for all time steps

    # Here we want only the probability assigned to the correct token. We must index using targets, but targets need an extra dimension
    log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return -log_probs.mean() # Mean across every dimension

    

    
    
    