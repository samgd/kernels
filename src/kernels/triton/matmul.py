from math import prod

import torch
import triton.language as tl
import triton
from jaxtyping import Float


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "M_TILE_SIZE": M_TILE_SIZE,
                "K_TILE_SIZE": K_TILE_SIZE,
                "N_TILE_SIZE": N_TILE_SIZE,
                "GROUP_SIZE_M": GROUP_SIZE_M,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for M_TILE_SIZE in [32, 64, 128]
        for K_TILE_SIZE in [32, 64, 128]
        for N_TILE_SIZE in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
        for GROUP_SIZE_M in [4, 8, 16]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M,
    K,
    N,
    M_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    N_TILE_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Swizzle program_ids to improve L2 cache hits.
    #
    # Desired ordering of program_ids:
    # [ 0,  2,  4,  6,  8, 10, 12, 14]
    # [ 1,  3,  5,  7,  9, 11, 13, 15]
    # [16, 18, 20, 22, 24, 26, 28, 30]
    # [17, 19, 21, 23, 25, 27, 29, 31]
    #
    # Compute program_id -> [m_index, n_index] mapping. Example with GROUP_SIZE_M=2, M=4, N=8:
    #
    # group_id:
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 1,  1,  1,  1,  1,  1,  1,  1]
    # [ 1,  1,  1,  1,  1,  1,  1,  1]
    #
    # pid_in_group:
    # [ 0,  2,  4,  6,  8, 10, 12, 14]
    # [ 1,  3,  5,  7,  9, 11, 13, 15]
    # [ 0,  2,  4,  6,  8, 10, 12, 14]
    # [ 1,  3,  5,  7,  9, 11, 13, 15]
    #
    # first_m:
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 2,  2,  2,  2,  2,  2,  2,  2]
    # [ 2,  2,  2,  2,  2,  2,  2,  2]
    #
    # pid_in_group % GROUP_SIZE_M
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 1,  1,  1,  1,  1,  1,  1,  1]
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 1,  1,  1,  1,  1,  1,  1,  1]
    #
    # m_index = first_m + (pid_in_group % GROUP_SIZE_M)
    # [ 0,  0,  0,  0,  0,  0,  0,  0]
    # [ 1,  1,  1,  1,  1,  1,  1,  1]
    # [ 2,  2,  2,  2,  2,  2,  2,  2]
    # [ 3,  3,  3,  3,  3,  3,  3,  3]
    #
    # n_index = pid_in_group // GROUP_SIZE_M
    # [ 0,  1,  2,  3,  4,  5,  6,  7]
    # [ 0,  1,  2,  3,  4,  5,  6,  7]
    # [ 0,  1,  2,  3,  4,  5,  6,  7]
    # [ 0,  1,  2,  3,  4,  5,  6,  7]
    n_pid_n = tl.cdiv(N, N_TILE_SIZE)
    S = GROUP_SIZE_M * n_pid_n

    group_id = pid // S
    pid_in_group = pid % S
    first_m = group_id * GROUP_SIZE_M
    m_index = first_m + (pid_in_group % GROUP_SIZE_M)
    n_index = pid_in_group // GROUP_SIZE_M

    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(m_index * M_TILE_SIZE, 0),
        block_shape=(M_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0),
    )

    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, n_index * N_TILE_SIZE),
        block_shape=(K_TILE_SIZE, N_TILE_SIZE),
        order=(1, 0),
    )

    C_block_ptr = tl.make_block_ptr(
        C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_index * M_TILE_SIZE, n_index * N_TILE_SIZE),
        block_shape=(M_TILE_SIZE, N_TILE_SIZE),
        order=(1, 0),
    )

    acc = tl.zeros(shape=(M_TILE_SIZE, N_TILE_SIZE), dtype=tl.float32)

    for k in range(tl.cdiv(K, K_TILE_SIZE)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")

        acc = tl.dot(a, b, acc=acc)

        A_block_ptr = A_block_ptr.advance((0, K_TILE_SIZE))
        B_block_ptr = B_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(C_block_ptr, acc.to(C_ptr.type.element_ty), boundary_check=(0, 1))


def matmul(
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
    stride_cm, stride_cn = C.stride()

    def grid(meta):
        return (triton.cdiv(meta["M"], meta["M_TILE_SIZE"]) * triton.cdiv(meta["N"], meta["N_TILE_SIZE"]),)

    matmul_kernel[grid](
        A,
        B,
        C,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        M,
        N,
        K,
    )

    C = C.reshape(*dims, K)

    return C
