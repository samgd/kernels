import time
from collections.abc import Callable
from typing import Annotated

import numpy as np
import torch
import typer
from jaxtyping import Float

from kernels.pytorch.attention import scaled_dot_product_attention as naive_pytorch_sdpa
from kernels.triton.attention import scaled_dot_product_attention as triton_sdpa


str_to_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def attention_function(
    attention: str,
) -> Callable[
    [
        Float[torch.Tensor, "batch q_seq_len d"],
        Float[torch.Tensor, "batch kv_seq_len d"],
        Float[torch.Tensor, "batch kv_seq_len d"],
        bool,
    ],
    Float[torch.Tensor, "batch q_seq_len d"],
]:
    if attention == "naive_pytorch":
        return naive_pytorch_sdpa
    if attention == "triton":
        return triton_sdpa
    raise NotImplementedError(f"unknown {attention=}")


def main(
    attention: Annotated[str, typer.Option()],
    warmup_steps: Annotated[int, typer.Option()],
    benchmark_steps: Annotated[int, typer.Option()],
    batch: Annotated[int, typer.Option()],
    q_seq_len: Annotated[int, typer.Option()],
    kv_seq_len: Annotated[int, typer.Option()],
    hidden_size: Annotated[int, typer.Option()],
    is_causal: Annotated[bool, typer.Option()],
    dtype_str: Annotated[str, typer.Option()],
    device_str: Annotated[str, typer.Option()],
):
    dtype = str_to_dtype[dtype_str]
    device = torch.device(device_str)

    q = torch.randn((batch, q_seq_len, hidden_size), dtype=dtype, device=device)
    k = torch.randn((batch, kv_seq_len, hidden_size), dtype=dtype, device=device)
    v = torch.randn((batch, kv_seq_len, hidden_size), dtype=dtype, device=device)

    attn_fn = attention_function(attention)

    for _ in range(warmup_steps):
        attn_fn(q, k, v, is_causal)
    torch.cuda.synchronize()

    times = []
    for i in range(benchmark_steps):
        start = time.perf_counter_ns()
        with torch.cuda.nvtx.range(f"{attention} iter_{i}"):
            attn_fn(q, k, v, is_causal)
            torch.cuda.synchronize()
        times.append(time.perf_counter_ns() - start)
    fwd_mu_ms = np.mean(times) / 1e6
    fwd_std_ms = np.std(times) / 1e6
    print(f"Forward pass: mu={fwd_mu_ms:5.3f}ms std={fwd_std_ms:5.3f}ms")


if __name__ == "__main__":
    typer.run(main)
