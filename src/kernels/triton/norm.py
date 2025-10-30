from math import prod

import torch
import triton
import triton.language as tl
from jaxtyping import Float


@triton.jit
def rms_norm_kernel(
    X_ptr,
    W_ptr,
    O_ptr,
    stride_xb,
    stride_xh,
    stride_wh,
    stride_ob,
    stride_oh,
    HIDDEN_SIZE: tl.constexpr,
    BATCH_TOTAL: tl.constexpr,
    BATCH_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
):
    batch_index = tl.program_id(0)

    X_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(BATCH_TOTAL, HIDDEN_SIZE),
        strides=(stride_xb, stride_xh),
        offsets=(batch_index * HIDDEN_SIZE, 0),
        block_shape=(BATCH_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )

    W_block_ptr = tl.make_block_ptr(
        W_ptr,
        shape=(HIDDEN_SIZE,),
        strides=(stride_wh,),
        offsets=(0,),
        block_shape=(HIDDEN_TILE_SIZE,),
        order=(0,),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr,
        shape=(BATCH_TOTAL, HIDDEN_SIZE),
        strides=(stride_ob, stride_oh),
        offsets=(batch_index * HIDDEN_SIZE, 0),
        block_shape=(BATCH_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )


def rms_norm(
    x: Float[torch.Tensor, "... hidden_size"], weight: Float[torch.Tensor, "hidden_size"]
) -> Float[torch.Tensor, "... hidden_size"]:
    *dims, hidden_size = x.shape
    batch_total = prod(dims)
    x = x.flatten(end_dim=-2)
    o = torch.empty_like(x)

    stride_xb, stride_xh = x.stride()
    stride_wh = weight.stride()
    stride_ob, stride_oh = o.stride()

    def grid(meta):
        return triton.cdiv(meta["BATCH_TOTAL"], meta["B_SIZE"])

    BATCH_TILE_SIZE = 8
    HIDDEN_TILE_SIZE = 64
    rms_norm_kernel[grid](
        x,
        weight,
        o,
        stride_xb,
        stride_xh,
        stride_wh,
        stride_ob,
        stride_oh,
        hidden_size,  # type: ignore
        batch_total,  # type: ignore
        BATCH_TILE_SIZE,  # type: ignore
        HIDDEN_TILE_SIZE,  # type: ignore
    )

    return o
