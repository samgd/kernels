from math import prod

import torch
from jaxtyping import Float

from kernels.triton.matmul import matmul


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Float[torch.Tensor, "... hidden_size"],
        weight: Float[torch.Tensor, "output_size hidden_size"],
        bias: Float[torch.Tensor, " output_size"] | None = None,
    ) -> Float[torch.Tensor, "... output_size"]:
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        return matmul(x, weight.T.contiguous(), bias)

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: Float[torch.Tensor, "... output_size"]
    ) -> tuple[
        Float[torch.Tensor, "... hidden_size"],
        Float[torch.Tensor, "output_size hidden_size"],
        Float[torch.Tensor, " output_size"] | None,
    ]:
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None

        grad_input = matmul(grad_output, weight)

        *dims, output_size = grad_output.shape
        grad_output = grad_output.reshape(prod(dims), output_size)
        grad_weight = matmul(grad_output.T, x.reshape(prod(dims), -1))

        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias  # type: ignore


def linear(
    x: Float[torch.Tensor, "... hidden_size"],
    weight: Float[torch.Tensor, "output_size hidden_size"],
    bias: Float[torch.Tensor, " output_size"] | None = None,
) -> Float[torch.Tensor, "... output_size"]:
    return _Linear.apply(x, weight, bias)  # type: ignore


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._bias = bias

        self.weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        self.bias = torch.empty(out_features, device=device, dtype=dtype) if self._bias else None

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        return linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self._bias}"
