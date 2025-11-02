import torch
from jaxtyping import Float


def angular_freqs(
    head_dim: int, base: float = 10_000.0, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> Float[torch.Tensor, " half_head_dim"]:
    j = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    return (base ** ((-2 / head_dim) * j)).to(dtype)


def rotate_half(x: Float[torch.Tensor, "... 2 half_head_dim"]) -> Float[torch.Tensor, "... 2 half_head_dim"]:
    x1 = x[..., 0, :]
    y1 = x[..., 1, :]
    return torch.stack([-y1, x1], dim=-2)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Model head dimension must be divisible by 2"
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.freqs = angular_freqs(self.head_dim, self.base, device, dtype=torch.float32)
        self._cos = None
        self._sin = None

    def _get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Float[torch.Tensor, "seq_len 1 1 half_head_dim"], Float[torch.Tensor, "seq_len 1 1 half_head_dim"]]:
        if self._cos is not None and self._cos.dtype == dtype and self._cos.shape[0] >= seq_len:
            if device is not None:
                self._cos = self._cos.to(device)
                self._sin = self._sin.to(device)  # type: ignore
            return self._cos[:seq_len], self._sin[:seq_len]
        self.max_seq_len = seq_len if self.max_seq_len is None else max(seq_len, self.max_seq_len)
        if device is not None:
            self.freqs = self.freqs.to(device)
        pos = torch.arange(self.max_seq_len, dtype=self.freqs.dtype, device=self.freqs.device)
        self._cos = torch.cos(self.freqs[None, :] * pos[:, None]).to(dtype)[:, None, None, :]
        self._sin = torch.sin(self.freqs[None, :] * pos[:, None]).to(dtype)[:, None, None, :]
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    ) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
        cos, sin = self._get_cos_sin(x.shape[-3], device=x.device, dtype=x.dtype)
        x = x.unflatten(dim=-1, sizes=(2, -1))
        out = x * cos + rotate_half(x) * sin
        return out.flatten(start_dim=-2)
