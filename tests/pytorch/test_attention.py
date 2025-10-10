import torch
import hypothesis.strategies as st
from jaxtyping import Float
from hypothesis import given, note

from kernels.pytorch.attention import scaled_dot_product_attention as pytorch_attention
from kernels.triton.attention import scaled_dot_product_attention as triton_attention


@st.composite
def attention_examples(
    draw,
    min_batch=1, max_batch=4,
    min_q=1, max_q=257,
    min_k=1, max_k=257,
    min_d=16, max_d=256,
) -> tuple[
        Float[torch.Tensor, "batch q_seq_len head_dim"],
        Float[torch.Tensor, "batch kv_seq_len head_dim"],
        Float[torch.Tensor, "batch kv_seq_len head_dim"],
        bool
]:
    batch = draw(st.integers(min_batch, max_batch))
    n_q = draw(st.integers(min_q, max_q))
    n_k = draw(st.integers(min_k, max_k))
    D = draw(st.integers(min_d, max_d))

    dtype = draw(st.sampled_from([torch.float16, torch.bfloat16, torch.float32]))

    seed = draw(st.integers(min_value=0, max_value=1_000_000))

    rng = torch.Generator(device=torch.device("cuda"))
    rng = rng.manual_seed(seed)

    Q = torch.randn(batch, n_q, D, generator=rng, dtype=dtype, device=torch.device("cuda"))
    K = torch.randn(batch, n_k, D, generator=rng, dtype=dtype, device=torch.device("cuda"))
    V = torch.randn(batch, n_k, D, generator=rng, dtype=dtype, device=torch.device("cuda"))

    is_causal = draw(st.booleans())

    note(f"{batch=}, {n_q=}, {n_k=}, {D=}, {dtype=}, {seed=}, {is_causal=}")

    return Q, K, V, is_causal


@given(attention_examples())
def test_triton_scaled_dot_product_attention(example):
    q, k, v, is_causal = example
    exp = pytorch_attention(q, k, v, is_causal=is_causal)
    act = triton_attention(q, k, v, is_causal=is_causal)
    max_abs_diff = (exp - act).abs().max()
    assert torch.allclose(exp, act, atol=1e-2, rtol=2e-2), f"{max_abs_diff=}"