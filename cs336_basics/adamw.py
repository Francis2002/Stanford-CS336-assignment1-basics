from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data # Get the gradient of loss with respect to p.
                state["m"] = beta1 * m + (1 - beta1) * grad # m ← β1m + (1 − β1)g
                state["v"] = beta2 * v + (1 - beta2) * grad * grad # v ← β2v + (1 − β2)g^2
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t) #αt ← lr * sqrt(1 − β2^t) / 1 − β1^t
                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss