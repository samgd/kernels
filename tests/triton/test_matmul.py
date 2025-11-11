import torch
import hypothesis.strategies as st
from jaxtyping import Float
from hypothesis import given, note

from kernels.triton.matmul import matmul


@st.composite
def matmul_examples(
    draw,
    bias: bool,
    min_d=16,
    max_d=300,
) -> tuple[Float[torch.Tensor, "M K"], Float[torch.Tensor, "K N"], Float[torch.Tensor, " N"] | None]:
    M = draw(st.integers(min_d, max_d))
    K = draw(st.integers(min_d, max_d))
    N = draw(st.integers(min_d, max_d))

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    note(f"{bias=}, {M=}, {K=}, {N=}, {dtype=}, {seed=}")

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    A = torch.randn(M, K, generator=rng, dtype=dtype, device=torch.device("cuda")) * 0.1
    B = torch.randn(K, N, generator=rng, dtype=dtype, device=torch.device("cuda")) * 0.1
    if bias:
        o = torch.randn((N,), dtype=dtype, device=torch.device("cuda")) * 0.1
    else:
        o = None

    return A, B, o


def run_test(A, B, bias):
    exp = A @ B
    if bias is not None:
        exp += bias
    act = matmul(A, B, bias)
    max_abs_diff = (exp - act).abs().max()
    atol = 2e-2
    rtol = 3e-2
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"


@given(matmul_examples(bias=False))
def test_triton_matmul(example):
    run_test(*example)


@given(matmul_examples(bias=True))
def test_triton_matmul_bias(example):
    run_test(*example)
