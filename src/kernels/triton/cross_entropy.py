from typing import cast

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Integer


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={"VOCAB_TILE_SIZE": vts},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for vts in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["VOCAB_SIZE"],
)
@triton.jit
def cross_entropy_forward_kernel(
    logits_ptr,
    targets_ptr,
    out_ptr,
    lse_ptr,
    stride_lb: tl.int64,  # type: ignore
    stride_lv: tl.int64,  # type: ignore
    stride_tb: tl.int64,  # type: ignore
    BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_TILE_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Find maximum value in row.
    max_logits_ptr = logits_ptr + batch_idx * stride_lb
    m = tl.full((), -float("inf"), dtype=tl.float32)
    offsets = tl.arange(start=0, end=VOCAB_TILE_SIZE)
    for _ in range(tl.cdiv(VOCAB_SIZE, VOCAB_TILE_SIZE)):
        mask = offsets < VOCAB_SIZE
        logits = tl.load(max_logits_ptr + offsets * stride_lv, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(logits, axis=0))
        offsets += VOCAB_TILE_SIZE

    # Compute logsumexp over row and store for backwards.
    lse_logits_ptr = logits_ptr + batch_idx * stride_lb
    lse = tl.full((), 0.0, dtype=tl.float32)
    offsets = tl.arange(start=0, end=VOCAB_TILE_SIZE)
    for _ in range(tl.cdiv(VOCAB_SIZE, VOCAB_TILE_SIZE)):
        mask = offsets < VOCAB_SIZE
        logits = tl.load(lse_logits_ptr + offsets * stride_lv, mask=mask, other=m).to(tl.float32)
        lse += tl.sum(tl.where(mask, tl.exp(logits - m), 0.0), axis=0)
        offsets += VOCAB_TILE_SIZE
    lse = m + tl.log(lse)
    tl.store(lse_ptr + batch_idx, lse)

    # Compute and store output.
    target_idx = tl.load(targets_ptr + batch_idx * stride_tb)
    target_logit = tl.load(logits_ptr + batch_idx * stride_lb + target_idx * stride_lv).to(tl.float32)
    out = (-target_logit + lse).to(logits_ptr.type.element_ty)
    tl.store(out_ptr + batch_idx, out)


def cross_entropy_forward(
    logits: Float[torch.Tensor, "batch vocab"], targets: Integer[torch.Tensor, " batch"]
) -> tuple[Float[torch.Tensor, " batch"], Float[torch.Tensor, " batch"]]:
    batch_size, vocab_size = logits.shape

    out = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    lse = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    stride_lb, stride_lv = logits.stride()
    (stride_tb,) = targets.stride()

    def grid(meta):
        return (meta["BATCH_SIZE"],)

    cross_entropy_forward_kernel[grid](
        logits, targets, out, lse, stride_lb, stride_lv, stride_tb, batch_size, vocab_size
    )

    return out, lse


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={"VOCAB_TILE_SIZE": vts},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for vts in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["VOCAB_SIZE"],
)
@triton.jit
def cross_entropy_backward_kernel(
    logits_ptr,
    targets_ptr,
    lse_ptr,
    grad_logits_ptr,
    grad_output_ptr,
    stride_lb: tl.int64,  # type: ignore
    stride_lv: tl.int64,  # type: ignore
    stride_tb: tl.int64,  # type: ignore
    stride_gb: tl.int64,  # type: ignore
    stride_gv: tl.int64,  # type: ignore
    stride_ob: tl.int64,  # type: ignore
    BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_TILE_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    _logits_ptr = logits_ptr + batch_idx * stride_lb
    _grad_logits_ptr = grad_logits_ptr + batch_idx * stride_gb

    target_idx = tl.load(targets_ptr + batch_idx * stride_tb)
    lse = tl.load(lse_ptr + batch_idx)
    grad_output = tl.load(grad_output_ptr + batch_idx * stride_ob)

    offsets = tl.arange(start=0, end=VOCAB_TILE_SIZE)
    for _ in range(tl.cdiv(VOCAB_SIZE, VOCAB_TILE_SIZE)):
        mask = offsets < VOCAB_SIZE
        logits = tl.load(_logits_ptr + offsets * stride_lv, mask=mask, other=0.0).to(tl.float32)
        probs = tl.exp(logits - lse)
        grad = grad_output * (probs - (offsets == target_idx).to(tl.float32))
        tl.store(_grad_logits_ptr + offsets * stride_gv, grad, mask=mask)
        offsets += VOCAB_TILE_SIZE


def cross_entropy_backward(
    logits: Float[torch.Tensor, "batch vocab"],
    targets: Integer[torch.Tensor, " batch"],
    lse: Float[torch.Tensor, " batch"],
    grad_output: Float[torch.Tensor, " batch"],
) -> Float[torch.Tensor, "batch vocab"]:
    batch_size, vocab_size = logits.shape

    grad_logits = torch.empty((batch_size, vocab_size), device=logits.device, dtype=torch.float32)
    stride_lb, stride_lv = logits.stride()
    (stride_tb,) = targets.stride()
    stride_gb, stride_gv = grad_logits.stride()
    (stride_ob,) = grad_output.stride()

    def grid(meta):
        return (meta["BATCH_SIZE"],)

    cross_entropy_backward_kernel[grid](
        logits,
        targets,
        lse,
        grad_logits,
        grad_output,
        stride_lb,
        stride_lv,
        stride_tb,
        stride_gb,
        stride_gv,
        stride_ob,
        batch_size,
        vocab_size,
    )

    return grad_logits


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits: Float[torch.Tensor, "batch vocab"], targets: Integer[torch.Tensor, " batch"]
    ) -> Float[torch.Tensor, "batch vocab"]:
        out, lse = cross_entropy_forward(logits, targets)
        ctx.save_for_backward(logits, targets, lse)
        return out

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: Float[torch.Tensor, " batch"]
    ) -> tuple[Float[torch.Tensor, "batch vocab"], None]:
        logits, targets, lse = ctx.saved_tensors
        return cross_entropy_backward(logits, targets, lse, grad_output), None


def cross_entropy(
    logits: Float[torch.Tensor, "batch vocab"], targets: Integer[torch.Tensor, " batch"]
) -> Float[torch.Tensor, "batch vocab"]:
    return cast(Float[torch.Tensor, "batch vocab"], _CrossEntropy.apply(logits, targets))
