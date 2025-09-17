import torch
import triton
import triton.language as tl
from jaxtyping import Float


@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, o_ptr,
    x_stride_m, x_stride_k,
    y_stride_k, y_stride_n,
    o_stride_m, o_stride_n,
    M, K, N,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = tl.arange(0, BLOCK_SIZE)
    rn = tl.arange(0, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + rm
    offs_n = pid_n * BLOCK_SIZE + rn

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_SIZE):
        rk = tl.arange(0, BLOCK_SIZE)
        offs_k = k0 + rk

        x_ptrs = x_ptr + (offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k)
        y_ptrs = y_ptr + (offs_k[:, None] * y_stride_k + offs_n[None, :] * y_stride_n)

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        y_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y_tile = tl.load(y_ptrs, mask=y_mask, other=0.0)

        acc += tl.dot(x_tile.to(tl.float32), y_tile.to(tl.float32))

    o_ptrs = o_ptr + (offs_m[:, None] * o_stride_m + offs_n[None, :] * o_stride_n)
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


def matmul(x: Float[torch.Tensor, "M K"], y: Float[torch.Tensor, "K N"], block_size: int = 128) -> Float[torch.Tensor, "M N"]:
    m, k = x.shape
    k2, n = y.shape
    assert k2 == k
    o = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE"]),
        triton.cdiv(n, meta["BLOCK_SIZE"]),
    )

    matmul_kernel[grid](
        x, y, o,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        o.stride(0), o.stride(1),
        m, k, n,
        BLOCK_SIZE=block_size,
        num_warps=2,
        num_stages=2,
    )
    return o


@triton.jit
def matmul_kernel_l2(
    x_ptr, y_ptr, o_ptr,
    x_stride_m, x_stride_k,
    y_stride_k, y_stride_n,
    o_stride_m, o_stride_n,
    M, K, N,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    n_block_n = tl.cdiv(N, BLOCK_SIZE)

    blocks_per_group = GROUP_SIZE_M * n_block_n
    group_id        = pid // blocks_per_group
    idx_in_group    = pid % blocks_per_group

    pid_n  = idx_in_group // GROUP_SIZE_M
    local_m = idx_in_group % GROUP_SIZE_M

    pid_m = group_id * GROUP_SIZE_M + local_m

    rm = tl.arange(0, BLOCK_SIZE)
    rn = tl.arange(0, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + rm
    offs_n = pid_n * BLOCK_SIZE + rn

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_SIZE):
        rk = tl.arange(0, BLOCK_SIZE)
        offs_k = k0 + rk

        x_ptrs = x_ptr + (offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_k)
        y_ptrs = y_ptr + (offs_k[:, None] * y_stride_k + offs_n[None, :] * y_stride_n)

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        y_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y_tile = tl.load(y_ptrs, mask=y_mask, other=0.0)

        acc = tl.dot(x_tile, y_tile, acc)

    o_ptrs = o_ptr + (offs_m[:, None] * o_stride_m + offs_n[None, :] * o_stride_n)
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


def matmul_l2(x: Float[torch.Tensor, "M K"], y: Float[torch.Tensor, "K N"], block_size: int = 128) -> Float[torch.Tensor, "M N"]:
    m, k = x.shape
    k2, n = y.shape
    assert k2 == k
    o = torch.empty((m, n), device=x.device, dtype=x.dtype)

    n_block_m = triton.cdiv(m, block_size)
    n_block_n = triton.cdiv(n, block_size)

    grid = lambda meta: (n_block_m * n_block_n,)

    matmul_kernel_l2[grid](
        x, y, o,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        o.stride(0), o.stride(1),
        m, k, n,
        BLOCK_SIZE=block_size,
        GROUP_SIZE_M=8,
        num_warps=2,
        num_stages=2,
    )
    return o