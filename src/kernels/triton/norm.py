from math import prod

import torch
import triton
import triton.language as tl
from jaxtyping import Float


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "ROW_TILE_SIZE": ROW_TILE_SIZE,
                "HIDDEN_TILE_SIZE": HIDDEN_TILE_SIZE,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for ROW_TILE_SIZE in [4, 8, 16]
        for HIDDEN_TILE_SIZE in [64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [2]
    ],
    key=["HIDDEN_SIZE", "ROWS"],
)
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
    eps: float,
    HIDDEN_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
):
    row_index = tl.program_id(0)
    row_tile_index = row_index * ROW_TILE_SIZE

    X_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_xb, stride_xh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
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
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_ob, stride_oh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )

    # First pass over x: compute root mean square
    sq = tl.zeros(shape=(ROW_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(HIDDEN_SIZE, HIDDEN_TILE_SIZE)):
        x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        sq += tl.sum(x * x, axis=-1)
        X_block_ptr = X_block_ptr.advance((0, HIDDEN_TILE_SIZE))
    sq /= HIDDEN_SIZE
    rms = tl.rsqrt(sq + eps)

    # Reset X pointer
    X_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_xb, stride_xh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )

    # Second pass over x: apply rms and weight
    for i in range(tl.cdiv(HIDDEN_SIZE, HIDDEN_TILE_SIZE)):
        x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        w = tl.load(W_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        # w:   [HIDDEN_SIZE,]
        # rms: [ROWS,]
        # x:   [ROWS, HIDDEN_SIZE]
        o = w * rms.expand_dims(axis=-1) * x

        tl.store(O_block_ptr, value=o.to(O_ptr.type.element_ty), boundary_check=(0, 1))

        X_block_ptr = X_block_ptr.advance((0, HIDDEN_TILE_SIZE))
        W_block_ptr = W_block_ptr.advance((HIDDEN_TILE_SIZE,))
        O_block_ptr = O_block_ptr.advance((0, HIDDEN_TILE_SIZE))


def _rms_norm(
    x: Float[torch.Tensor, "... hidden_size"], weight: Float[torch.Tensor, " hidden_size"], eps: float = 1e-5
) -> Float[torch.Tensor, "... hidden_size"]:
    *dims, hidden_size = x.shape
    batch_total = prod(dims)
    x = x.reshape(batch_total, hidden_size)
    o = torch.empty_like(x)

    stride_xb, stride_xh = x.stride()
    stride_wh, *_ = weight.stride()
    stride_ob, stride_oh = o.stride()

    def grid(meta):
        return (triton.cdiv(meta["ROWS"], meta["ROW_TILE_SIZE"]),)

    rms_norm_kernel[grid](
        x,
        weight,
        o,
        stride_xb,
        stride_xh,
        stride_wh,
        stride_ob,
        stride_oh,
        eps,
        hidden_size,  # type: ignore
        batch_total,  # type: ignore
    )

    return o.reshape(*dims, hidden_size)


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "ROW_TILE_SIZE": ROW_TILE_SIZE,
                "HIDDEN_TILE_SIZE": HIDDEN_TILE_SIZE,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for ROW_TILE_SIZE in [4, 8, 16]
        for HIDDEN_TILE_SIZE in [64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [2]
    ],
    key=["HIDDEN_SIZE", "ROWS"],
    reset_to_zero=["GW_ptr"],
)
@triton.jit
def _rms_norm_backward_kernel(
    X_ptr,
    W_ptr,
    GO_ptr,
    GI_ptr,
    GW_ptr,
    stride_xb,
    stride_xh,
    stride_wh,
    stride_gob,
    stride_goh,
    stride_gib,
    stride_gih,
    stride_gwh,
    eps: float,
    HIDDEN_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
):
    row_index = tl.program_id(0)
    row_tile_index = row_index * ROW_TILE_SIZE

    X_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_xb, stride_xh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
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

    GO_block_ptr = tl.make_block_ptr(
        GO_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_gob, stride_goh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )

    GI_block_ptr = tl.make_block_ptr(
        GI_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_gib, stride_gih),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )

    # First pass: compute mean square and grad_output * x * w
    sq = tl.zeros(shape=(ROW_TILE_SIZE,), dtype=tl.float32)
    dot = tl.zeros(shape=(ROW_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(HIDDEN_SIZE, HIDDEN_TILE_SIZE)):
        x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        w = tl.load(W_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32).expand_dims(0)
        go = tl.load(GO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        sq += tl.sum(x * x, axis=-1)
        dot += tl.sum(x * w * go, axis=-1)

        X_block_ptr = X_block_ptr.advance((0, HIDDEN_TILE_SIZE))
        W_block_ptr = W_block_ptr.advance((HIDDEN_TILE_SIZE,))
        GO_block_ptr = GO_block_ptr.advance((0, HIDDEN_TILE_SIZE))

    sq /= HIDDEN_SIZE
    r = tl.sqrt(sq + eps)
    inv_r = 1.0 / r
    inv_r3 = inv_r * inv_r * inv_r

    inv_r = tl.expand_dims(inv_r, -1)
    inv_r3 = tl.expand_dims(inv_r3, -1)

    coef = tl.expand_dims(dot / HIDDEN_SIZE, -1) * inv_r3

    # Reset pointers
    X_block_ptr = tl.make_block_ptr(
        X_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_xb, stride_xh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
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

    GO_block_ptr = tl.make_block_ptr(
        GO_ptr,
        shape=(ROWS, HIDDEN_SIZE),
        strides=(stride_gob, stride_goh),
        offsets=(row_tile_index, 0),
        block_shape=(ROW_TILE_SIZE, HIDDEN_TILE_SIZE),
        order=(1, 0),
    )

    # Use manual indexing instead of make_block_ptr for dQ as it requires atomic_add
    GW_base = GW_ptr
    GW_offset = tl.arange(0, HIDDEN_TILE_SIZE)

    # Second pass: compute gradient for input and weight
    for i in range(tl.cdiv(HIDDEN_SIZE, HIDDEN_TILE_SIZE)):
        x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        w = tl.load(W_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32).expand_dims(0)
        go = tl.load(GO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        gi = (go * w) * inv_r - x * coef
        gw = tl.sum(go * (x * inv_r), axis=0)

        tl.store(GI_block_ptr, gi, boundary_check=(0, 1))
        tl.atomic_add(GW_base + GW_offset * stride_gwh, gw, mask=GW_offset < HIDDEN_SIZE)

        X_block_ptr = X_block_ptr.advance((0, HIDDEN_TILE_SIZE))
        W_block_ptr = W_block_ptr.advance((HIDDEN_TILE_SIZE,))
        GO_block_ptr = GO_block_ptr.advance((0, HIDDEN_TILE_SIZE))
        GI_block_ptr = GI_block_ptr.advance((0, HIDDEN_TILE_SIZE))
        GW_offset += HIDDEN_TILE_SIZE


def _rms_norm_backward(
    grad_output: Float[torch.Tensor, "... hidden_size"],
    x: Float[torch.Tensor, "... hidden_size"],
    weight: Float[torch.Tensor, " hidden_size"],
    eps: float,
) -> tuple[Float[torch.Tensor, "... hidden_size"], Float[torch.Tensor, "... hidden_size"]]:
    *dims, hidden_size = x.shape
    batch_total = prod(dims)
    x = x.reshape(batch_total, hidden_size)
    grad_output = grad_output.reshape(batch_total, hidden_size)
    grad_input = torch.empty_like(x, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)

    stride_xb, stride_xh = x.stride()
    stride_wh, *_ = weight.stride()
    stride_gob, stride_goh = grad_output.stride()
    stride_gib, stride_gih = grad_input.stride()
    stride_gwh, *_ = grad_weight.stride()

    def grid(meta):
        return (triton.cdiv(meta["ROWS"], meta["ROW_TILE_SIZE"]),)

    _rms_norm_backward_kernel[grid](
        x,
        weight,
        grad_output,
        grad_input,
        grad_weight,
        stride_xb,
        stride_xh,
        stride_wh,
        stride_gob,
        stride_goh,
        stride_gib,
        stride_gih,
        stride_gwh,
        eps,
        hidden_size,  # type: ignore
        batch_total,  # type: ignore
    )

    torch.cuda.synchronize()

    return grad_input.reshape(*dims, hidden_size), grad_weight


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: Float[torch.Tensor, "... hidden_size"], weight: Float[torch.Tensor, " hidden_size"], eps: float = 1e-5
    ) -> Float[torch.Tensor, "... hidden_size"]:
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return _rms_norm(x, weight, eps)

    @staticmethod
    def backward(  # type: ignore
        ctx,
        grad_output: Float[torch.Tensor, "... hidden_size"],
    ) -> tuple[Float[torch.Tensor, "... hidden_size"], Float[torch.Tensor, "... hidden_size"], None]:
        x, weight = ctx.saved_tensors
        gx, gw = _rms_norm_backward(grad_output, x, weight, ctx.eps)
        return gx, gw, None


def rms_norm(
    x: Float[torch.Tensor, "... hidden_size"], weight: Float[torch.Tensor, " hidden_size"], eps: float = 1e-5
) -> Float[torch.Tensor, "... hidden_size"]:
    return _RMSNorm.apply(x, weight, eps)  # type: ignore
