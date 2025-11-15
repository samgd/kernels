import torch
from jaxtyping import Float


def sigmoid(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return 1.0 / (1 + torch.exp(-x))


def swish(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return x * sigmoid(x)
