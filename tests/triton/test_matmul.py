import torch
import hypothesis.strategies as st
from jaxtyping import Float
from hypothesis import given, note

from kernels.triton.matmul import matmul


@st.composite
def matmul_examples(
    draw,
    min_d=16,
    max_d=300,
) -> tuple[
    Float[torch.Tensor, "M K"],
    Float[torch.Tensor, "K N"],
]:
    M = draw(st.integers(min_d, max_d))
    K = draw(st.integers(min_d, max_d))
    N = draw(st.integers(min_d, max_d))

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    note(f"{M=}, {K=}, {N=}, {dtype=}, {seed=}")

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    A = torch.randn(M, K, generator=rng, dtype=dtype, device=torch.device("cuda"))
    B = torch.randn(K, N, generator=rng, dtype=dtype, device=torch.device("cuda"))

    return A, B


@given(matmul_examples())
def test_triton_matmul(example):
    A, B = example
    exp = A @ B
    act = matmul(A, B)
    max_abs_diff = (exp - act).abs().max()
    if A.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"
