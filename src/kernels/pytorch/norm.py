import einx
import torch
from jaxtyping import Float


class RMSNorm(torch.nn.Module):
    """Root-mean-square layer normalization with learnable scaling.

    Args:
        hidden_size: Size of the last dimension to normalize over.
        eps: Added for numerical stability.
        device: Optional device for parameters.
        dtype: Optional dtype for parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... hidden_size"]) -> Float[torch.Tensor, "... hidden_size"]:
        return rms_norm(x, self.weight, self.eps)


def rms_norm(
    x: Float[torch.Tensor, "... hidden_size"],
    weight: Float[torch.Tensor, " hidden_size"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "... hidden_size"]:
    """Apply root-mean-square layer normalization.

    Args:
        x: Input tensor.
        weight: Scale parameter applied after normalization.
        eps: Added to the mean of squares for stability.

    Returns:
        Tensor normalized by the root-mean-square statistic and scaled by ``weight``.
    """
    input_dtype = x.dtype
    x = x.float()
    mean_sq = einx.mean("... [hidden_size]", x.square(), keepdims=True)
    irms = torch.rsqrt(mean_sq + eps)
    return (weight.float() * (irms * x)).to(input_dtype)
