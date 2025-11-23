import torch
from jaxtyping import Float, Integer


def cross_entropy(
    logits: Float[torch.Tensor, "batch vocab"], targets: Integer[torch.Tensor, " batch"]
) -> Float[torch.Tensor, " batch"]:
    """Compute numerically-stable cross entropy between logits and class targets.

    Args:
        logits: Unnormalized logit tensor.
        targets: Integer class indices.

    Returns:
        Per-example cross entropy losses.
    """
    m = logits.max(dim=1).values
    lse = m + torch.log(torch.sum(torch.exp(logits - m[:, None]), dim=1))
    li = logits[torch.arange(logits.shape[0]), targets]
    return -li + lse
