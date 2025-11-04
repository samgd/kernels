import torch
import hypothesis.strategies as st
from jaxtyping import Float
from hypothesis import given, note

from kernels.pytorch.rotary import RotaryEmbedding as PyTorchRotaryEmbedding
from kernels.triton.rotary import RotaryEmbedding as TritonRotaryEmbedding


@st.composite
def tensor_examples(
    draw, min_dims=3, max_dims=5, min_dim=1, max_dim=32, min_head_dim=16, max_head_dim=258
) -> tuple[Float[torch.Tensor, "... seq_len n_head head_dim"], float]:
    dims = draw(st.lists(st.integers(min_dim, max_dim), min_size=min_dims, max_size=max_dims - 1))

    head_dim = 2 * (draw(st.integers(min_head_dim, max_head_dim)) // 2)  # ensure head_dim is a multiple of 2
    dims = dims + [head_dim]

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    base = draw(
        st.floats(min_value=100.0, max_value=20_000, allow_nan=False, allow_infinity=False, allow_subnormal=False)
    )

    note(f"{dims=}, {dtype=}, {seed=}, {base=}")

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    x = torch.randn(*dims, generator=rng, dtype=dtype, device=torch.device("cuda"))

    return x, base


@given(tensor_examples())
def test_triton_rotary_embedding_forward(example):
    x, base = example
    *_, head_dim = x.shape
    exp = PyTorchRotaryEmbedding(head_dim, base=base)(x)
    act = TritonRotaryEmbedding(head_dim, base=base)(x)
    max_abs_diff = (exp - act).abs().max()
    if x.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"


@given(tensor_examples())
def test_triton_rotary_embedding_backward(example):
    x, base = example
    *_, head_dim = x.shape

    x_pyt = x.clone().requires_grad_()
    exp = PyTorchRotaryEmbedding(head_dim, base=base)(x_pyt)
    l_pyt = exp.sum()
    l_pyt.backward()

    x_trt = x.clone().requires_grad_()
    act = TritonRotaryEmbedding(head_dim, base=base)(x_trt)
    l_trt = act.sum()
    l_trt.backward()

    max_abs_diff = (act - exp).abs().max()
    if x.dtype in [torch.float16, torch.bfloat16]:
        atol = 2e-2
        rtol = 3e-2
    else:
        atol = 1e-5
        rtol = 1e-4
    assert torch.allclose(act, exp, atol=atol, rtol=rtol), f"{max_abs_diff=}"
