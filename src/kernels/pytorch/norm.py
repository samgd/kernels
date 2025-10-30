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
        input_dtype = x.dtype
        x = x.float()
        mean_sq = einx.mean("... [hidden_size]", x.square(), keepdims=True)
        irms = torch.rsqrt(self.eps + mean_sq)
        return (self.weight.float() * (x * irms)).to(input_dtype)
