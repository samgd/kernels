import math

import einx
import torch
from jaxtyping import Float

def scaled_dot_product_attention(
    q: Float[torch.Tensor, "... q_seq_len d"],
    k: Float[torch.Tensor, "... kv_seq_len d"],
    v: Float[torch.Tensor, "... kv_seq_len d"],
    is_causal: bool = False
) -> Float[torch.Tensor, "... q_seq_len d"]:
    s = einx.dot("... q_seq_len d, ... kv_seq_len d -> ... q_seq_len kv_seq_len", q, k)

    s = s / math.sqrt(q.shape[-1])

    m = einx.max("... q_seq_len [kv_seq_len]", s).values
    s = (s - m.unsqueeze(-1)).exp()
    d = einx.sum("... q_seq_len [kv_seq_len]", s, keepdims=True)

    p = s / d

    return einx.dot("... q_seq_len kv_seq_len, ... kv_seq_len d -> ... q_seq_len d", p, v)