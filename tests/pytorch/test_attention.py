import torch
import hypothesis_torch
import hypothesis.strategies as st
from hypothesis import given, HealthCheck, settings

from kernels.pytorch.attention import scaled_dot_product_attention


@st.composite
def attention_examples(
    draw,
    min_batch=1, max_batch=4,
    min_q=1, max_q=257,
    min_k=1, max_k=257,
    min_d=16, max_d=257,
):
    batch = draw(st.integers(min_batch, max_batch))
    n_q = draw(st.integers(min_q, max_q))
    n_k = draw(st.integers(min_k, max_k))
    D = draw(st.integers(min_d, max_d))

    elem = st.floats(-1e2, 1e2, allow_nan=False, allow_infinity=False, width=16)

    Q = draw(
        hypothesis_torch.tensor_strategy(
            dtype=torch.float32,
            shape=(batch, n_q, D),
            layout=torch.strided,
            device=torch.device("cuda"),
            elements=elem
        ),
    )
    K = draw(
        hypothesis_torch.tensor_strategy(
            dtype=torch.float32,
            shape=(batch, n_k, D),
            layout=torch.strided,
            device=torch.device("cuda"),
            elements=elem
        ),
    )
    V = draw(
        hypothesis_torch.tensor_strategy(
            dtype=torch.float32,
            shape=(batch, n_k, D),
            layout=torch.strided,
            device=torch.device("cuda"),
            elements=elem
        ),
    )

    return Q, K, V


@settings(deadline=500, suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow])
@given(attention_examples())
def test_scaled_dot_product_attention(example):
    q, k, v = example
    exp = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    act = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(exp, act)