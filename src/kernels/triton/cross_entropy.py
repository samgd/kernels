import torch
import triton
import triton.language as tl
from jaxtyping import Float, Integer


@triton.jit
def cross_entropy_forward_kernel(
    logits_ptr,
    targets_ptr,
    out_ptr,
    stride_lb,
    stride_lv,
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

    # Compute logsumexp over row.
    lse_logits_ptr = logits_ptr + batch_idx * stride_lb
    lse = tl.full((), 0.0, dtype=tl.float32)
    offsets = tl.arange(start=0, end=VOCAB_TILE_SIZE)
    for _ in range(tl.cdiv(VOCAB_SIZE, VOCAB_TILE_SIZE)):
        mask = offsets < VOCAB_SIZE
        logits = tl.load(lse_logits_ptr + offsets * stride_lv, mask=mask, other=m).to(tl.float32)
        lse += tl.sum(tl.where(mask, tl.exp(logits - m), 0.0), axis=0)
        offsets += VOCAB_TILE_SIZE
    lse = m + tl.log(lse)

    # Compute and store output.
    target_idx = tl.load(targets_ptr + batch_idx)
    target_logit = tl.load(logits_ptr + batch_idx * stride_lb + target_idx * stride_lv).to(tl.float32)
    out = -target_logit + lse
    tl.store(out_ptr + batch_idx, out)


def cross_entropy_forward(
    logits: Float[torch.Tensor, "batch vocab"], targets: Integer[torch.Tensor, " batch"]
) -> Float[torch.Tensor, " batch"]:
    batch_size, vocab_size = logits.shape

    out = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)
    stride_lb, stride_lv = logits.shape

    def grid(meta):
        return (meta["BATCH_SIZE"],)

    cross_entropy_forward_kernel[grid](logits, targets, out, stride_lb, stride_lv, batch_size, vocab_size, 64)

    return out
