import torch
from jaxtyping import Float, Integer


def cross_entropy(
    logits: Float[torch.Tensor, "batch vocab"], targets: Integer[torch.Tensor, " batch"]
) -> Float[torch.Tensor, " batch"]:
    m = logits.max(dim=1).values
    lse = m + torch.log(torch.sum(torch.exp(logits - m[:, None]), dim=1))
    li = logits[torch.arange(logits.shape[0]), targets]
    return -li + lse
