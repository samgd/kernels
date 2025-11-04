import torch
import hypothesis.strategies as st
from jaxtyping import Float
from hypothesis import given, note

from kernels.pytorch.norm import rms_norm as pytorch_rms_norm
from kernels.triton.norm import rms_norm as triton_rms_norm


@st.composite
def tensor_examples(
    draw,
    min_dims=1,
    max_dims=4,
    min_dim=8,
    max_dim=257,
) -> tuple[Float[torch.Tensor, "... hidden_size"], Float[torch.Tensor, "... hidden_size"]]:
    dims = draw(st.lists(st.integers(min_dim, max_dim), min_size=min_dims, max_size=max_dims))

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    x = torch.randn(*dims, generator=rng, dtype=dtype, device=torch.device("cuda"))
    weight = torch.randn(dims[-1], generator=rng, dtype=dtype, device=torch.device("cuda"))

    note(f"{dims=}, {dtype=}, {seed=}")

    return x, weight


@given(tensor_examples())
def test_triton_rms_norm_forward(example):
    x, weight = example
    exp = pytorch_rms_norm(x, weight)
    act = triton_rms_norm(x, weight)
    max_abs_diff = (exp - act).abs().max()
    if x.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"


@given(tensor_examples())
def test_triton_rms_norm_backward(example):
    x, w = example

    x_pyt = x.clone().requires_grad_()
    w_pyt = w.clone().requires_grad_()
    exp = pytorch_rms_norm(x_pyt, w_pyt)
    l_pyt = exp.sum()
    l_pyt.backward()

    x_trt = x.clone().requires_grad_()
    w_trt = w.clone().requires_grad_()
    act = triton_rms_norm(x_trt, w_trt)
    l_trt = act.sum()
    l_trt.backward()

    for typ, act, exp in [("x", x_trt.grad, x_pyt.grad), ("w", w_trt.grad, w_pyt.grad)]:
        max_abs_diff = (act - exp).abs().max()
        if x.dtype in [torch.float16, torch.bfloat16]:
            atol = 2e-2
            rtol = 3e-2
        else:
            atol = 1e-5
            rtol = 1e-4
        assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{typ=}, {max_abs_diff=}"
