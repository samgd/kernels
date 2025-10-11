import math

import torch
import triton
import triton.language as tl
from jaxtyping import Float


def get_padded_D(D: int) -> int:
    if D not in [32, 64, 128, 256]:
        if D > 256:
            raise ValueError(f"Head Dimension {D} must be <= 256")
        PADDED_D = None
        for d in [32, 64, 128, 256]:
            if d > D:
                PADDED_D = d
                break
        assert PADDED_D is not None
    else:
        PADDED_D = D
    return PADDED_D


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "Q_TILE_SIZE": Q_TILE_SIZE,
                "K_TILE_SIZE": K_TILE_SIZE,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for Q_TILE_SIZE in [16, 32, 64, 128]
        for K_TILE_SIZE in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["N_QUERIES", "N_KEYS", "is_causal", "PADDED_D"],
)
@triton.jit
def triton_flash_attention_2_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    PADDED_D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    o_block = tl.zeros((Q_TILE_SIZE, PADDED_D), dtype=tl.float32)
    l_block = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_block = tl.full((Q_TILE_SIZE,), value=-float("inf"), dtype=tl.float32)

    q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_start = query_index * Q_TILE_SIZE

    k_cols = tl.arange(0, K_TILE_SIZE)
    q_rows = tl.arange(0, Q_TILE_SIZE)

    q_pos = (q_start + q_rows)[:, None]

    if is_causal:
        n_keys = tl.minimum(N_KEYS, (query_index + 1) * Q_TILE_SIZE)
    else:
        n_keys = N_KEYS

    for ki in range(tl.cdiv(n_keys, K_TILE_SIZE)):
        k_start = ki * K_TILE_SIZE
        k_pos = k_start + k_cols
        col_mask = (k_pos < N_KEYS)[None, :]

        if is_causal:
            col_mask &= k_pos[None, :] <= q_pos

        k_part = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_part = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        s_part = tl.dot(q_block, tl.trans(k_part))
        s_part = scale * s_part
        s_part = tl.where(col_mask, s_part, -float("inf"))
        m_part = tl.max(s_part, axis=1)
        m_new = tl.maximum(m_block, m_part)
        p_part = tl.exp(s_part - m_new[:, None])
        l_part = tl.sum(p_part, axis=1)

        sc = tl.exp(m_block - m_new)
        m_block = m_new
        l_block *= sc
        l_block += l_part

        p_part = p_part.to(V_ptr.type.element_ty)
        o_part = tl.dot(p_part, v_part)
        o_block *= sc[:, None]
        o_block += o_part

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = (o_block / l_block[:, None]).to(O_ptr.type.element_ty)
    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, m_block + tl.log(l_block), boundary_check=(0,))


def triton_flash_attention_2(Q, K, V, is_causal=False):
    B, NQ, D = Q.shape
    _, NK, _ = K.shape

    PADDED_D = get_padded_D(D)

    O = torch.empty_like(Q)
    L = torch.empty(B, NQ, dtype=torch.float32, device=Q.device)

    stride_qb, stride_qq, stride_qd = Q.stride()
    stride_kb, stride_kk, stride_kd = K.stride()
    stride_vb, stride_vk, stride_vd = V.stride()
    stride_ob, stride_oq, stride_od = O.stride()
    stride_lb, stride_lq = L.stride()

    scale = 1 / math.sqrt(D)

    def grid(meta):
        return triton.cdiv(NQ, meta["Q_TILE_SIZE"]), B

    triton_flash_attention_2_kernel[grid](
        Q,
        K,
        V,
        O,
        L,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        NQ,
        NK,
        scale,
        is_causal,
        D,
        PADDED_D,
    )

    return O, L


@triton.autotune(
    configs=[
        triton.Config(kwargs={"Q_TILE_SIZE": Q_TILE_SIZE}, num_warps=num_warps, num_stages=num_stages)
        for Q_TILE_SIZE in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["N_QUERIES", "PADDED_D"],
)
@triton.jit
def triton_backward_dO_O_dot(
    O_ptr,
    dO_ptr,
    E_ptr,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_eb,
    stride_eq,
    N_QUERIES,
    D: tl.constexpr,
    PADDED_D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    query_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    E_block_ptr = tl.make_block_ptr(
        E_ptr + batch_index * stride_eb,
        shape=(N_QUERIES,),
        strides=(stride_eq,),
        offsets=(query_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_block = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dO_block = dO_block.to(tl.float32)
    O_block = O_block.to(tl.float32)

    e = dO_block * O_block
    e = tl.sum(e, axis=1)

    tl.store(E_block_ptr, e, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "Q_TILE_SIZE": Q_TILE_SIZE,
                "K_TILE_SIZE": K_TILE_SIZE,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for Q_TILE_SIZE in [16, 32, 64, 128]
        for K_TILE_SIZE in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["N_QUERIES", "N_KEYS", "PADDED_D"],
    reset_to_zero=["dQ_ptr"],
)
@triton.jit
def triton_flash_attention_2_backward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    E_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    stride_eb,
    stride_eq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    PADDED_D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    key_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    if is_causal:
        q_base = (key_index * K_TILE_SIZE) // Q_TILE_SIZE
    else:
        q_base = 0

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_base * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_base * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_base * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, PADDED_D),
        order=(1, 0),
    )

    E_block_ptr = tl.make_block_ptr(
        E_ptr + batch_index * stride_eb,
        shape=(N_QUERIES,),
        strides=(stride_eq,),
        offsets=(q_base * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    k_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dK_block = tl.zeros((K_TILE_SIZE, PADDED_D), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, PADDED_D), dtype=tl.float32)

    # Use manual indexing instead of make_block_ptr for dQ as it requires atomic_add
    dQ_base = dQ_ptr + batch_index * stride_dqb

    k_start = key_index * K_TILE_SIZE
    k_pos = k_start + tl.arange(0, K_TILE_SIZE)
    k_inb = k_pos < N_KEYS

    for qi in range(q_base, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q0 = qi * Q_TILE_SIZE
        rows = q0 + tl.arange(0, Q_TILE_SIZE)[:, None]
        cols = tl.arange(0, PADDED_D)[None, :]
        row_m = rows < N_QUERIES
        dQ_ptrs = dQ_base + rows * stride_dqq + cols * stride_dqd

        q_part = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_part = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        l_part = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        e_part = tl.load(E_block_ptr, boundary_check=(0,), padding_option="zero")

        s_part = tl.dot(q_part, tl.trans(k_block))
        s_part = scale * s_part

        mask = k_inb[None, :]
        if is_causal:
            mask = mask & (k_pos[None, :] <= rows)
        s_part = tl.where(mask, s_part, -float("inf"))

        p_part = tl.exp(s_part - l_part[:, None])

        dV_block += tl.dot(tl.trans(p_part).to(dO_ptr.type.element_ty), dO_part)

        dP = tl.dot(dO_part, tl.trans(v_block))

        dS = dP - e_part[:, None]
        dS = p_part * dS

        dSp = scale * dS
        dSp = dSp.to(K_ptr.type.element_ty)

        tl.atomic_add(dQ_ptrs, tl.dot(dSp, k_block).to(tl.float32), mask=row_m)
        dK_block += tl.dot(tl.trans(dSp), q_part)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        E_block_ptr = E_block_ptr.advance((Q_TILE_SIZE,))

    dK_block = dK_block.to(K_ptr.type.element_ty)
    dV_block = dV_block.to(V_ptr.type.element_ty)

    tl.store(dK_block_ptr, dK_block, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_block, boundary_check=(0, 1))


def triton_flash_attention_2_backward(Q, K, V, O, L, dO, is_causal=False):
    B, NQ, D = Q.shape
    _, NK, _ = K.shape

    PADDED_D = get_padded_D(D)

    # PADDED_D for dQ, many global load/stores so power of 2 beneficial
    dQ = torch.zeros((*Q.shape[:-1], PADDED_D), dtype=torch.float32, device=Q.device)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    E = torch.empty((B, NQ), dtype=torch.float32, device=Q.device)

    stride_qb, stride_qq, stride_qd = Q.stride()
    stride_kb, stride_kk, stride_kd = K.stride()
    stride_vb, stride_vk, stride_vd = V.stride()
    stride_ob, stride_oq, stride_od = O.stride()
    stride_lb, stride_lq = L.stride()

    stride_dob, stride_doq, stride_dod = dO.stride()

    stride_dqb, stride_dqq, stride_dqd = dQ.stride()
    stride_dkb, stride_dkk, stride_dkd = dK.stride()
    stride_dvb, stride_dvk, stride_dvd = dV.stride()

    stride_eb, stride_eq = E.stride()

    scale = 1 / math.sqrt(D)

    def grid_dO_O(meta):
        return triton.cdiv(NQ, meta["Q_TILE_SIZE"]), B

    triton_backward_dO_O_dot[grid_dO_O](
        O,
        dO,
        E,
        stride_ob,
        stride_oq,
        stride_od,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_eb,
        stride_eq,
        NQ,
        D,
        PADDED_D,
    )

    def grid_bwd(meta):
        return triton.cdiv(NK, meta["K_TILE_SIZE"]), B

    triton_flash_attention_2_backward_kernel[grid_bwd](
        Q,
        K,
        V,
        L,
        dO,
        dQ,
        dK,
        dV,
        E,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_lb,
        stride_lq,
        stride_dob,
        stride_doq,
        stride_dod,
        stride_dqb,
        stride_dqq,
        stride_dqd,
        stride_dkb,
        stride_dkk,
        stride_dkd,
        stride_dvb,
        stride_dvk,
        stride_dvd,
        stride_eb,
        stride_eq,
        NQ,
        NK,
        scale,
        is_causal,
        D,
        PADDED_D,
    )

    dQ = dQ[..., :D].to(Q.dtype)

    return dQ, dK, dV


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        assert Q.is_contiguous()
        assert K.is_contiguous()
        assert V.is_contiguous()

        out, L = triton_flash_attention_2(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, out, L)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, *_ = grad_outputs
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = triton_flash_attention_2_backward(Q, K, V, O, L, grad_output, ctx.is_causal)
        return dQ, dK, dV, None


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "batch q_seq_len d"],
    k: Float[torch.Tensor, "batch kv_seq_len d"],
    v: Float[torch.Tensor, "batch kv_seq_len d"],
    is_causal: bool = False,
):
    """ """
    return FlashAttention2.apply(q, k, v, is_causal)
