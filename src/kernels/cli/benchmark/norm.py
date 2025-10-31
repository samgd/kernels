import time
from collections.abc import Callable
from typing import Annotated

import numpy as np
import torch
import typer
from jaxtyping import Float

from kernels.pytorch.norm import rms_norm as pytorch_rms_norm
from kernels.triton.norm import rms_norm as triton_norm


str_to_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def rms_norm_function(
    rms_norm: str,
) -> Callable[
    [
        Float[torch.Tensor, "... hidden_size"],
        Float[torch.Tensor, "hidden_size"],
    ],
    Float[torch.Tensor, "... hidden_size"],
]:
    if rms_norm == "pytorch":
        return pytorch_rms_norm
    if rms_norm == "triton":
        return triton_norm
    raise NotImplementedError(f"unknown {rms_norm=}")


def main(
    rms_norm: Annotated[str, typer.Option()],
    warmup_steps: Annotated[int, typer.Option()],
    benchmark_steps: Annotated[int, typer.Option()],
    batch: Annotated[int, typer.Option()],
    effective_batch_size: Annotated[int, typer.Option()],
    hidden_size: Annotated[int, typer.Option()],
    dtype_str: Annotated[str, typer.Option()],
    device_str: Annotated[str, typer.Option()],
):
    dtype = str_to_dtype[dtype_str]
    device = torch.device(device_str)

    x = torch.randn((effective_batch_size, hidden_size), dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)

    fn = rms_norm_function(rms_norm)

    for i in range(warmup_steps):
        print(f"warmup: {i} start", flush=True)
        fn(x, weight)
        print(f"warmup: {i} end", flush=True)
    torch.cuda.synchronize()

    times = []
    for i in range(benchmark_steps):
        print(f"bench: {i} start", flush=True)
        start = time.perf_counter_ns()
        with torch.cuda.nvtx.range(f"{rms_norm} iter_{i}"):
            fn(x, weight)
            torch.cuda.synchronize()
        times.append(time.perf_counter_ns() - start)
        print(f"bench: {i} end", flush=True)
    fwd_mu_ms = np.mean(times) / 1e6
    fwd_std_ms = np.std(times) / 1e6
    print(f"Forward pass: mu={fwd_mu_ms:5.3f}ms std={fwd_std_ms:5.3f}ms")


if __name__ == "__main__":
    typer.run(main)
