import torch
import hypothesis.strategies as st
from jaxtyping import Float
from hypothesis import given, note

from kernels.pytorch.activation import sigmoid as pytorch_sigmoid
from kernels.triton.activation import sigmoid as triton_sigmoid


@st.composite
def tensor_examples(
    draw,
    min_n_dims=1,
    max_n_dims=3,
    min_dim=8,
    max_dim=257,
) -> Float[torch.Tensor, " ..."]:
    dims = draw(st.lists(st.integers(min_dim, max_dim), min_size=min_n_dims, max_size=max_n_dims))

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    x = torch.randn(dims, generator=rng, dtype=dtype, device=torch.device("cuda"))

    note(f"{dims=}, {dtype=}, {seed=}")

    return x


@given(tensor_examples())
def test_triton_sigmoid_forward(x):
    exp = pytorch_sigmoid(x)
    act = triton_sigmoid(x)
    max_abs_diff = (exp - act).abs().max()
    if x.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"


@given(tensor_examples())
def test_triton_sigmoid_backward(x):
    x_pyt = x.clone().requires_grad_()
    out_pyt = pytorch_sigmoid(x_pyt)
    l_pyt = out_pyt.sum()
    l_pyt.backward()
    act = x_pyt.grad

    x_trt = x.clone().requires_grad_()
    out_trt = triton_sigmoid(x_trt)
    l_trt = out_trt.sum()
    l_trt.backward()
    exp = x_trt.grad

    max_abs_diff = (act - exp).abs().max()
    if x.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"
