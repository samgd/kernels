import torch
import hypothesis.strategies as st

from jaxtyping import Float
from hypothesis import given, note

from kernels.triton.linear import Linear


@st.composite
def linear_examples(
    draw,
    min_d=16,
    max_d=300,
) -> tuple[Linear, torch.nn.Linear, Float[torch.Tensor, "... hidden_size"]]:
    in_features = draw(st.integers(min_d, max_d))
    out_features = draw(st.integers(min_d, max_d))
    bias = draw(st.booleans())

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    note(f"{in_features=}, {out_features=} {dtype=}, {seed=}")

    device = torch.device("cuda")

    rng = torch.Generator(device=device)
    rng = rng.manual_seed(seed)

    m_trt = Linear(in_features, out_features, bias, device=device, dtype=dtype)
    m_pyt = torch.nn.Linear(in_features, out_features, bias, device=device, dtype=dtype)
    with torch.no_grad():
        m_pyt.weight.data = m_trt.weight.data.detach().clone()
        if bias:
            assert m_trt.bias is not None
            m_pyt.bias.data = m_trt.bias.data.detach().clone()

    x = torch.randn((16, in_features), device=device, dtype=dtype)

    return m_trt, m_pyt, x


@given(linear_examples())
def test_triton_linear_forward(example):
    m_trt, m_pyt, x = example
    exp = m_pyt(x)
    act = m_trt(x)
    max_abs_diff = (exp - act).abs().max()
    atol = 2e-2
    rtol = 3e-2
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"


@given(linear_examples())
def test_triton_linear_backward(example):
    m_trt, m_pyt, x = example

    m_trt.train()
    m_pyt.train()

    x_pyt = x.detach().clone().requires_grad_()
    exp = m_pyt(x_pyt)
    exp.sum().backward()

    x_trt = x.detach().clone().requires_grad_()
    act = m_trt(x_trt)
    act.sum().backward()

    for typ, pyt, trt in [
        ("x", x_pyt, x_trt),
        ("weight", m_pyt.weight, m_trt.weight),
        ("bias", m_pyt.bias, m_trt.bias),
    ]:
        if typ == "bias" and pyt is None:
            continue
        max_abs_diff = (trt.grad - pyt.grad).abs().max()
        atol = 2e-2
        rtol = 3e-2
        assert torch.allclose(trt.grad, pyt.grad, atol=atol, rtol=rtol), f"{typ=}, {max_abs_diff=}"
