import torch
import hypothesis.strategies as st
from jaxtyping import Float, Integer
from hypothesis import given, note

from kernels.pytorch.cross_entropy import cross_entropy as pytorch_cross_entropy
from kernels.triton.cross_entropy import cross_entropy as triton_cross_entropy


@st.composite
def tensor_examples(
    draw,
    min_dim=8,
    max_dim=257,
) -> tuple[Float[torch.Tensor, "batch vocab_size"], Integer[torch.Tensor, " batch"]]:
    batch, vocab_size = draw(st.lists(st.integers(min_dim, max_dim), min_size=2, max_size=2))

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    logits = torch.randn((batch, vocab_size), generator=rng, dtype=dtype, device=torch.device("cuda"))
    targets = torch.randint(low=0, high=vocab_size, size=(batch,), generator=rng, device=torch.device("cuda"))

    note(f"{batch=}, {vocab_size=}, {dtype=}, {seed=}")

    return logits, targets


@given(tensor_examples())
def test_triton_cross_entropy_forward(example):
    logits, targets = example
    exp = pytorch_cross_entropy(logits, targets)
    act = triton_cross_entropy(logits, targets)
    max_abs_diff = (exp - act).abs().max()
    if logits.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"


@given(tensor_examples())
def test_triton_cross_entropy_backward(example):
    logits, targets = example

    logits_pyt = logits.clone().requires_grad_()
    out_pyt = pytorch_cross_entropy(logits_pyt, targets)
    l_pyt = out_pyt.sum()
    l_pyt.backward()
    act = logits_pyt.grad

    logits_trt = logits.clone().requires_grad_()
    out_trt = triton_cross_entropy(logits_trt, targets)
    l_trt = out_trt.sum()
    l_trt.backward()
    exp = logits_trt.grad

    max_abs_diff = (act - exp).abs().max()
    if logits.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"
