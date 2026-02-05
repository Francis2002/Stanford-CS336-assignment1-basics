import torch

def cross_entropy(logits, targets):
    """
    Args:
        logits (torch.Tensor): Logits with shape (..., vocab_size)
        targets (torch.Tensor): Target indices with shape (...)
    """
    # torch.logsumexp for numerical stability and memory efficiency.
    # Also avoids materializing massive [B, T, V] intermediate exp/sum matrices.
    log_sum_exp = torch.logsumexp(logits, dim=-1)

    # We only need the logits corresponding to the target labels.
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Loss = log(sum(exp(logits))) - logit_target
    return (log_sum_exp - target_logits).mean()

    

    
    
    