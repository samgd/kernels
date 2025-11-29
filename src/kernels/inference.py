from typing import cast

import einx
import torch
from jaxtyping import Float, Integer

from kernels.transformer import CausalMHA, Impl
from kernels.pytorch.rotary import rotate_half


def rotary_embedding(
    x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    cos: Float[torch.Tensor, "seq_len half_head_dim"],
    sin: Float[torch.Tensor, "seq_len half_head_dim"],
) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
    x = x.unflatten(dim=-1, sizes=(2, -1))
    out = x * cos + rotate_half(x) * sin
    return out.flatten(start_dim=-2)


class CausalMHAInfer(CausalMHA):
    """CausalMHA with KVCache forward pass.

    Args:
        impl: Implementation backend (PyTorch or Triton).
        d_model: Model hidden dimension.
        n_head: Number of attention heads.
        device: Optional device to initialise on.
        dtype: Optional dtype to use.
    """

    def __init__(
        self,
        impl: Impl,
        d_model: int,
        n_head: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(impl, d_model, n_head, device, dtype)
        self.cur_seq_lens: Integer[torch.Tensor, " batch"]
        self.register_buffer("cur_seq_lens", None)
        self.register_buffer("k_cache", None)

    @torch.inference_mode()
    def forward(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], prefill: bool = True
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        if prefill:
            return self.prefill(x)
        return self.decode(x)

    @torch.inference_mode()
    def prefill(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        B, L, H = x.shape
        assert B == 1, "Only batch_size=1 supported."
        q, k, v = self.Wqkv(x).view(B, L, 3, self.n_head, self.head_dim).unbind(dim=2)
        q = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", self.rope(q)))
        k = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", self.rope(k)))
        v = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", v))
        h = self.sdpa(q, k, v, is_causal=True)
        h = einx.rearrange("(B nh) L hd -> B L (nh hd)", h, B=B, L=L, nh=self.n_head, hd=self.head_dim)

        # New sequence, flush sequence lengths and kv cache.
        self.cur_seq_lens = torch.tensor([L], dtype=torch.int32, device=x.device)
        self.k_cache = k
        self.v_cache = v

        return self.Wout(h)

    @torch.inference_mode()
    def decode(self, x: Float[torch.Tensor, "batch 1 d_model"]) -> Float[torch.Tensor, "batch 1 d_model"]:
        B, L, H = x.shape
        assert B == 1, "Only batch_size=1 supported."
        assert L == 1, "'Input to `decode` must be current token only (seq_len == 1).'"

        tok_q, tok_k, tok_v = self.Wqkv(x).view(B, 1, 3, self.n_head, self.head_dim).unbind(dim=2)

        # TODO: move this into the RotaryEmbedding class
        # TODO: Generate many of these each time and cache, only generate next bunch when used up.
        freqs: Float[torch.Tensor, "1 half_head_dim"] = self.rope.freqs[None, :] * self.cur_seq_lens[:, None]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        tok_q = rotary_embedding(tok_q, cos, sin)
        tok_k = rotary_embedding(tok_k, cos, sin)

        tok_q = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", tok_q))
        tok_k = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", tok_k))
        tok_v = cast(torch.Tensor, einx.rearrange("B L nh hd -> (B nh) L hd", tok_v))

        # TODO: Increase KV cache size by 2**L and assign rather than cat by 1 each time.
        self.cur_seq_lens += 1
        self.k_cache = torch.cat([self.k_cache, tok_k], dim=1)
        self.v_cache = torch.cat([self.v_cache, tok_v], dim=1)

        h = self.sdpa(tok_q, self.k_cache, self.v_cache)
        h = einx.rearrange("(B nh) L hd -> B L (nh hd)", h, B=1, L=1, nh=self.n_head, hd=self.head_dim)

        return self.Wout(h)
