import torch
from jaxtyping import Float

from kernels.util import cdiv

def flash_attention_2(
    Q: Float[torch.Tensor, "q_seq_len hidden"],
    K: Float[torch.Tensor, "kv_seq_len hidden"],
    V: Float[torch.Tensor, "kv_seq_len hidden"],
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_M: int,
) -> Float[torch.Tensor, "q_seq_len hidden"]:

    N, H = Q.shape
    M = K.shape[0]
    o = torch.empty((N, H))

    for i in range(cdiv(N, BLOCK_SIZE_N)):

        m_cur = torch.full((BLOCK_SIZE_N,), fill_value=-float("inf"))
        s_cur = torch.zeros((BLOCK_SIZE_N, H))
        l_cur = torch.zeros((BLOCK_SIZE_N,))

        Qb = Q[i*BLOCK_SIZE_N:(i+1)*BLOCK_SIZE_N]

        for b in range(cdiv(M, BLOCK_SIZE_M)):
            Kb = K[b*BLOCK_SIZE_M:(b+1)*BLOCK_SIZE_M] # [BLOCK_SIZE_M, H]
            Vb = V[b*BLOCK_SIZE_M:(b+1)*BLOCK_SIZE_M] # [BLOCK_SIZE_M, H]

            QKb = Qb @ Kb.T                       # [BLOCK_SIZE_N, BLOCK_SIZE_M]

            m_b = QKb.max(dim=-1).values          # [BLOCK_SIZE_N]
            m = torch.maximum(m_cur, m_b)         # [BLOCK_SIZE_N]

            f_b = torch.exp(QKb - m[:, None])     # [BLOCK_SIZE_N, BLOCK_SIZE_M]
            s_b = f_b @ Vb                        # [BLOCK_SIZE_N, H]
            l_b = f_b.sum(dim=-1)                 # [BLOCK_SIZE_N]

            alpha = torch.exp(m_cur - m)          # [BLOCK_SIZE_N]
            beta = torch.exp(m_b - m)             # [BLOCK_SIZE_N]

            m_cur = m
            s_cur = alpha[:, None] * s_cur + s_b
            l_cur = alpha * l_cur + l_b

        o[i*BLOCK_SIZE_N:(i+1)*BLOCK_SIZE_N] = s_cur / l_cur[:, None]

    return o
