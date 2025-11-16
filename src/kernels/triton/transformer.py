from typing import cast

import einx
import torch
from jaxtyping import Float, Integer

from kernels.triton.activation import swish
from kernels.triton.attention import scaled_dot_product_attention
from kernels.triton.linear import Linear
from kernels.triton.norm import RMSNorm
from kernels.triton.rotary import RotaryEmbedding


class Embedding(torch.nn.Module):
    def __init__(
        self, num_embeddings: int, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, d_model, dtype=dtype, device=device))

    def forward(self, idxs: Integer[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        return self.weight[idxs]


class CausalMHA(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.d_model % self.n_head == 0, f"{d_model=} must be divisible by {n_head=}"
        self.head_dim = self.d_model // self.n_head
        self.rope = RotaryEmbedding(self.head_dim, device=device)
        self.Wqkv = Linear(d_model, 3 * d_model, bias=False, device=device, dtype=dtype)
        self.Wout = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        B, L, H = x.shape
        q, k, v = self.Wqkv(x).view(B, L, 3, self.n_head, self.head_dim).unbind(dim=2)
        q = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", self.rope(q)))
        k = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", self.rope(k)))
        v = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", v))
        h = scaled_dot_product_attention(q, k, v, is_causal=True)
        h = einx.rearrange("(B nh) L hd -> B L (nh hd)", h, B=B, L=L, nh=self.n_head, hd=self.head_dim)
        h = self.Wout(h)
        return h


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = Linear(d_model, 2 * d_ff, bias=False, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        a, gate = self.fc1(x).split(self.d_ff, dim=2)
        h = a * swish(gate)
        return self.fc2(h)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, n_head: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = CausalMHA(d_model, n_head, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        h = x + self.mha(self.norm1(x))
        o = h + self.ff(self.norm2(h))
        return o


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_vocab: int,
        depth: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embed = Embedding(n_vocab, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.Sequential(
            *[TransformerBlock(d_model, n_head, d_ff, device=device, dtype=dtype) for _ in range(depth)]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.proj = Linear(d_model, n_vocab, bias=False, device=device, dtype=dtype)

    def forward(self, idxs: Integer[torch.Tensor, "batch seq_len"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        return self.proj(self.norm(self.layers(self.embed(idxs))))
