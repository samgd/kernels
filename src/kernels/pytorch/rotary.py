import torch
from jaxtyping import Float, Int


def angular_freqs(
    head_dim: int, base: float = 10_000.0, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> Float[torch.Tensor, "half_head_dim"]:
    j = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    return (base ** ((-2 / head_dim) * j)).to(dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Model head dimension must be divisible by 2"
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("freqs", angular_freqs(self.head_dim, self.base, device, dtype))

    def forward(
        self,
        x: Float[torch.Tensor, "batch ... head_dim"],
        offsets: Int[torch.Tensor, " batch"],
    ) -> Float[torch.Tensor, "... head_dim"]:
        pass
