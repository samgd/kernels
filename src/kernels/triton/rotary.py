from math import prod

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from kernels.pytorch.rotary import angular_freqs


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
    ) -> tuple[Float[torch.Tensor, "seq_len half_head_dim"], Float[torch.Tensor, "seq_len half_head_dim"]]:
        if self._cos is not None and self._cos.dtype == dtype and self._cos.shape[0] >= seq_len:
            if device is not None:
                self._cos = self._cos.to(device)
                self._sin = self._sin.to(device)  # type: ignore
            return self._cos[:seq_len], self._sin[:seq_len]
        self.max_seq_len = seq_len if self.max_seq_len is None else max(seq_len, self.max_seq_len)
        if device is not None:
            self.freqs = self.freqs.to(device)
        pos = torch.arange(self.max_seq_len, dtype=self.freqs.dtype, device=self.freqs.device)
        self._cos = torch.cos(self.freqs[None, :] * pos[:, None]).to(dtype)
        self._sin = torch.sin(self.freqs[None, :] * pos[:, None]).to(dtype)
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    ) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
        cos, sin = self._get_cos_sin(x.shape[-3], device=x.device, dtype=x.dtype)
        return rotary_embedding(x, cos, sin)


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={"HALF_HEAD_DIM_TILE_SIZE": HALF_HEAD_DIM_TILE_SIZE},
            num_warps=num_warps,
            num_stages=1,
        )
        for HALF_HEAD_DIM_TILE_SIZE in [64, 128, 256]
        for num_warps in [2, 4, 8]
    ],
    key=["HALF_HEAD_DIM"],
)
@triton.jit
def rotary_embedding_kernel(
    X_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_xbs,
    stride_xsl,
    stride_xnh,
    stride_x2d,
    stride_xhd,
    stride_csl,
    stride_chd,
    stride_ssl,
    stride_shd,
    stride_obs,
    stride_osl,
    stride_onh,
    stride_o2d,
    stride_ohd,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEAD: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM_TILE_SIZE: tl.constexpr,
):
    BSH = tl.program_id(0)
    BH = BATCH * N_HEAD
    seq_len_idx = BSH // BH
    batch_head_idx = BSH % BH
    batch_idx = batch_head_idx // N_HEAD
    head_idx = batch_head_idx % N_HEAD

    tile_idx = tl.program_id(1) * HALF_HEAD_DIM_TILE_SIZE

    X1_block_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(BATCH, SEQ_LEN, N_HEAD, 2, HALF_HEAD_DIM),
        strides=(stride_xbs, stride_xsl, stride_xnh, stride_x2d, stride_xhd),
        offsets=(batch_idx, seq_len_idx, head_idx, 0, tile_idx),
        block_shape=(1, 1, 1, 1, HALF_HEAD_DIM_TILE_SIZE),
        order=(4, 3, 2, 1, 0),
    )
    Y1_block_ptr = X1_block_ptr.advance((0, 0, 0, 1, 0))

    cos_block_ptr = tl.make_block_ptr(
        base=cos_ptr,
        shape=(SEQ_LEN, HALF_HEAD_DIM),
        strides=(stride_csl, stride_chd),
        offsets=(seq_len_idx, tile_idx),
        block_shape=(1, HALF_HEAD_DIM_TILE_SIZE),
        order=(1, 0),
    )

    sin_block_ptr = tl.make_block_ptr(
        base=sin_ptr,
        shape=(SEQ_LEN, HALF_HEAD_DIM),
        strides=(stride_ssl, stride_shd),
        offsets=(seq_len_idx, tile_idx),
        block_shape=(1, HALF_HEAD_DIM_TILE_SIZE),
        order=(1, 0),
    )

    X2_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(BATCH, SEQ_LEN, N_HEAD, 2, HALF_HEAD_DIM),
        strides=(stride_obs, stride_osl, stride_onh, stride_o2d, stride_ohd),
        offsets=(batch_idx, seq_len_idx, head_idx, 0, tile_idx),
        block_shape=(1, 1, 1, 1, HALF_HEAD_DIM_TILE_SIZE),
        order=(4, 3, 2, 1, 0),
    )
    Y2_block_ptr = X2_block_ptr.advance((0, 0, 0, 1, 0))

    cos = tl.load(cos_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)
    sin = tl.load(sin_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)

    x1 = tl.load(X1_block_ptr, boundary_check=(4,), padding_option="zero").to(tl.float32)
    y1 = tl.load(Y1_block_ptr, boundary_check=(4,), padding_option="zero").to(tl.float32)

    x2 = cos * x1 - sin * y1
    y2 = sin * x1 + cos * y1

    tl.store(X2_block_ptr, x2.to(out_ptr.type.element_ty), boundary_check=(4,))
    tl.store(Y2_block_ptr, y2.to(out_ptr.type.element_ty), boundary_check=(4,))


def _rotary_embedding(
    x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    cos: Float[torch.Tensor, "seq_len half_head_dim"],
    sin: Float[torch.Tensor, "seq_len half_head_dim"],
) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
    *dims, seq_len, n_head, head_dim = x.shape
    batch = prod(dims)

    x = x.reshape(batch, seq_len, n_head, 2, head_dim // 2)
    out = torch.empty_like(x)

    stride_xbs, stride_xsl, stride_xnh, stride_x2d, stride_xhd = x.stride()
    stride_csl, stride_chd = cos.stride()
    stride_ssl, stride_shd = sin.stride()
    stride_obs, stride_osl, stride_onh, stride_o2d, stride_ohd = out.stride()

    def grid(meta):
        return (
            meta["BATCH"] * meta["SEQ_LEN"] * meta["N_HEAD"],
            triton.cdiv(meta["HALF_HEAD_DIM"], meta["HALF_HEAD_DIM_TILE_SIZE"]),
        )

    rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        out,
        stride_xbs,
        stride_xsl,
        stride_xnh,
        stride_x2d,
        stride_xhd,
        stride_csl,
        stride_chd,
        stride_ssl,
        stride_shd,
        stride_obs,
        stride_osl,
        stride_onh,
        stride_o2d,
        stride_ohd,
        batch,
        seq_len,
        n_head,
        head_dim // 2,
    )

    return out.reshape(*dims, seq_len, n_head, head_dim)


class _RotaryEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Float[torch.Tensor, "... seq_len n_head head_dim"],
        cos: Float[torch.Tensor, "seq_len half_head_dim"],
        sin: Float[torch.Tensor, "seq_len half_head_dim"],
    ) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
        ctx.save_for_backward(cos, sin)
        return _rotary_embedding(x, cos, sin)

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: Float[torch.Tensor, "... seq_len n_head head_dim"]
    ) -> tuple[Float[torch.Tensor, "... seq_len n_head head_dim"], None, None]:
        cos, sin = ctx.saved_tensors
        return _rotary_embedding(grad_output, cos, -sin), None, None


def rotary_embedding(
    x: Float[torch.Tensor, "... seq_len n_head head_dim"],
    cos: Float[torch.Tensor, "seq_len half_head_dim"],
    sin: Float[torch.Tensor, "seq_len half_head_dim"],
) -> Float[torch.Tensor, "... seq_len n_head head_dim"]:
    return _RotaryEmbedding.apply(x, cos, sin)  # type: ignore
