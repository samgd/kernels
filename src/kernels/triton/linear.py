from math import prod

import torch
import triton.language as tl
import triton
from jaxtyping import Float


@triton.jit
def linear_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    M,
    K,
    N,
):
    pass


def linear(
    A: Float[torch.Tensor, "... K"],
    B: Float[torch.Tensor, "K N"],
) -> Float[torch.Tensor, "... N"]:
    *dims, K = A.shape
    B_K, N = B.shape

    assert K == B_K, f"A {K=} and B {B_K=} must match"
    assert A.dtype == B.dtype, f"{A.dtype=} and {B.dtype=} must be the same"
    assert A.device == B.device, f"{A.device=} and {B.device=} must be the same"

    M = prod(dims)
    A = A.reshape(M, K)

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_om, stride_on = C.stride()

    linear_kernel(
        A,
        B,
        C,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_om,
        stride_on,
        M,
        N,
        K,
    )

    C = C.reshape(*dims, K)

    return C
