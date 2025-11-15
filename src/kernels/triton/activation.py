from typing import cast

import torch
import triton
import triton.language as tl
from jaxtyping import Float


@triton.jit
def sigmoid_forward_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offset < N

    x = tl.load(x_ptr + offset, mask=mask).to(tl.float32)
    h = (1.0 / (1.0 + tl.exp(-x))).to(out_ptr.type.element_ty)
    tl.store(out_ptr + offset, h, mask=mask)


def sigmoid_forward(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    x = x.contiguous()
    N = x.numel()

    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(meta["N"], meta["BLOCK_SIZE"]),)

    sigmoid_forward_kernel[grid](x, out, N, 1024)  # type: ignore

    return out


@triton.jit
def sigmoid_backward_kernel(x_ptr, grad_output_ptr, grad_input_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offset < N

    # Recompute sigmoid(x) to avoid precision loss if forward output isn't fp32.
    x = tl.load(x_ptr + offset, mask=mask).to(tl.float32)
    g = tl.load(grad_output_ptr + offset, mask=mask).to(tl.float32)
    h = 1.0 / (1.0 + tl.exp(-x))
    grad = g * h * (1 - h)
    tl.store(grad_input_ptr + offset, grad, mask=mask)


def sigmoid_backward(
    x: Float[torch.Tensor, " ..."], grad_output: Float[torch.Tensor, " ..."]
) -> Float[torch.Tensor, " ..."]:
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    N = x.numel()

    grad_input = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(meta["N"], meta["BLOCK_SIZE"]),)

    sigmoid_backward_kernel[grid](x, grad_output, grad_input, N, 1024)  # type: ignore

    return grad_input


class _Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        ctx.save_for_backward(x)
        return sigmoid_forward(x)

    @staticmethod
    def backward(ctx, grad_output: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:  # type: ignore
        (x,) = ctx.saved_tensors
        return sigmoid_backward(x, grad_output)


def sigmoid(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return cast(Float[torch.Tensor, " ..."], _Sigmoid.apply(x))
