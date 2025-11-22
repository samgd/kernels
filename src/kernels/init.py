import math

import torch
from jaxtyping import Float


def glorot_uniform_(weight: Float[torch.Tensor, "output_size hidden_size"]):
    """In-place Glorot uniform initialization."""
    output_size, hidden_size = weight.shape
    bound = math.sqrt(6 / (output_size + hidden_size))
    with torch.no_grad():
        # rand in [0, 1], shift by 0.5 to [-0.5, 0.5], multiply by 2 to [-1, 1], scale by bound to give glorot
        weight.data = bound * 2 * (torch.rand(weight.shape, dtype=weight.dtype, device=weight.device) - 0.5)


def zero_(tensor: Float[torch.Tensor, "..."]):
    """In-place zero initialization."""
    with torch.no_grad():
        tensor.zero_()
