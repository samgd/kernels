import einx
import torch
from jaxtyping import Float


class RMSNorm(torch.nn.Module):
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
    input_dtype = x.dtype
    x = x.float()
    mean_sq = einx.mean("... [hidden_size]", x.square(), keepdims=True)
    irms = torch.rsqrt(mean_sq + eps)
    return (weight.float() * (irms * x)).to(input_dtype)
