from torch import sigmoid, nn
from cs336_basics.linear import Linear

def silu_activation(x):
    return x * sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        # Combining operations avoids storing named intermediate tensors in the local scope
        return self.w2(silu_activation(self.w1(x)) * self.w3(x))
