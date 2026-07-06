# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER page-16 sparse paged-attention helpers for MiniMax-M3 on ROCm."""

import torch

try:
    from vllm.triton_utils import tl, triton
except ModuleNotFoundError:
    import triton
    import triton.language as tl

from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE

ASM_PAGE_SIZE = 16
PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE

_FP8_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
}


def _is_fp8_kv_cache_tensor(kv_cache: torch.Tensor) -> bool:
    return kv_cache.dtype in _FP8_DTYPES


_FP8_MAX = {
    getattr(torch, "float8_e4m3fn", None): 448.0,
    getattr(torch, "float8_e4m3fnuz", None): 240.0,
    getattr(torch, "float8_e5m2", None): 57344.0,
    getattr(torch, "float8_e5m2fnuz", None): 57344.0,
}


# ---------------------------------------------------------------------------
# Fused Gemma-QKnorm + partial-NeoX-RoPE + page-16 SHUFFLE KV insert.
#
# One kernel replaces the previous three-op decode/prefill insert sequence in
# the AITER sparse-PA path (fused_minimax_m3_qknorm_rope_kv_insert (norm/rope
# only) -> aiter.reshape_and_cache(asm_layout) -> minimax_m3_insert_index_cache).
# It reads q/k/v/index_q/index_k straight out of the single fused ``qkv`` GEMM
# output, so it also avoids the ``k.contiguous()`` / ``v.contiguous()`` copies
# that reshape_and_cache required. The SHUFFLE (page-16) K/V offset math matches
# aiter.reshape_and_cache(asm_layout=True); FP8 stores divide by the (scalar) KV
# scale so the gluon PA read (which multiplies by the same scale) round-trips.
# ---------------------------------------------------------------------------
@triton.jit
def _gemma_norm_rope_head(
    row_ptr,  # this head's input row (head_dim contiguous)
    w_ptr,  # norm weight [head_dim]
    cos_ptr,  # [half] cos for this token
    sin_ptr,  # [half] sin for this token
    HEAD_DIM: tl.constexpr,
    ROT_HALF: tl.constexpr,  # rotary_dim // 2
    eps,
):
    """Gemma (1+w) RMSNorm in fp32 + partial NeoX RoPE; returns fp32 [HEAD_DIM]."""
    d = tl.arange(0, HEAD_DIM)
    vals = tl.load(row_ptr + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    var = tl.sum(vals * vals, axis=0) / HEAD_DIM
    normed = vals * tl.rsqrt(var + eps) * (1.0 + w)

    dh = tl.arange(0, HEAD_DIM)
    is_low = dh < ROT_HALF
    in_rot = dh < (2 * ROT_HALF)
    partner_idx = tl.where(is_low, dh + ROT_HALF, dh - ROT_HALF)
    pvals = tl.load(row_ptr + partner_idx, mask=in_rot, other=0.0).to(tl.float32)
    pw = tl.load(w_ptr + partner_idx, mask=in_rot, other=0.0).to(tl.float32)
    p_normed = pvals * tl.rsqrt(var + eps) * (1.0 + pw)

    j = tl.where(is_low, dh, dh - ROT_HALF)
    cos = tl.load(cos_ptr + j, mask=in_rot, other=0.0)
    sin = tl.load(sin_ptr + j, mask=in_rot, other=0.0)
    sign = tl.where(is_low, -1.0, 1.0)
    roped = normed * cos + sign * p_normed * sin
    return tl.where(in_rot, roped, normed)


@triton.jit
def _fused_qknorm_rope_shuffle_insert_kernel(
    qkv_ptr,  # [num_tokens, row_elems]
    q_norm_w_ptr,  # [head_dim]
    k_norm_w_ptr,  # [head_dim]
    iq_norm_w_ptr,  # [idx_head_dim]
    ik_norm_w_ptr,  # [idx_head_dim]
    cos_sin_ptr,  # [max_pos, rotary_dim]  (first half cos, second half sin)
    positions_ptr,  # [num_tokens] int64
    slot_mapping_ptr,  # [num_tokens] int64 (logical token slot = block*128 + off)
    index_slot_mapping_ptr,  # [num_tokens] int64 or nullptr-equivalent
    q_out_ptr,  # [num_tokens, num_heads*head_dim]
    iq_out_ptr,  # [num_tokens, num_index_heads*idx_head_dim]
    kc_ptr,  # SHUFFLE K [nph16, nkv, head_dim//x, 16, x] (contiguous)
    vc_ptr,  # SHUFFLE V [nph16, nkv, 16//x, head_dim, x] (contiguous)
    index_cache_ptr,  # index K cache, flat page-128 [-1, idx_head_dim] (contiguous)
    k_scale_ptr,  # scalar fp8 K scale or dummy
    v_scale_ptr,  # scalar fp8 V scale or dummy
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_index_heads: tl.constexpr,
    head_dim: tl.constexpr,
    idx_head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps,
    row_elems: tl.constexpr,
    x: tl.constexpr,  # 16 // itemsize of the KV cache
    ASM_PAGE: tl.constexpr,  # 16
    FP8_KV: tl.constexpr,  # divide K/V by scale + clamp before fp8 store
    FP8_MAX: tl.constexpr,
    SKIP_INDEX: tl.constexpr,  # skip index_q/index_k branch (cross-layer share)
    INSERT_INDEX_K: tl.constexpr,  # write index_k into index_cache
):
    tok = tl.program_id(0)
    half = rotary_dim // 2
    pos = tl.load(positions_ptr + tok)
    cos_row = cos_sin_ptr + pos * rotary_dim
    sin_row = cos_sin_ptr + pos * rotary_dim + half

    # qkv row: [q (nq*hd) | k (nkv*hd) | v (nkv*hd) | iq (niq*idx) | ik (idx)]
    q_base = 0
    k_base = num_heads * head_dim
    v_base = k_base + num_kv_heads * head_dim
    iq_base = v_base + num_kv_heads * head_dim
    ik_base = iq_base + num_index_heads * idx_head_dim
    row = qkv_ptr + tok * row_elems
    d = tl.arange(0, head_dim)

    # ----- (1) q heads: norm + rope -> q_out -----
    for h in tl.static_range(num_heads):
        out = _gemma_norm_rope_head(
            row + q_base + h * head_dim,
            q_norm_w_ptr,
            cos_row,
            sin_row,
            head_dim,
            half,
            eps,
        )
        tl.store(
            q_out_ptr + tok * (num_heads * head_dim) + h * head_dim + d,
            out.to(q_out_ptr.dtype.element_ty),
        )

    # ----- (2) index_q heads: norm + rope -> iq_out -----
    if not SKIP_INDEX:
        di = tl.arange(0, idx_head_dim)
        for h in tl.static_range(num_index_heads):
            out = _gemma_norm_rope_head(
                row + iq_base + h * idx_head_dim,
                iq_norm_w_ptr,
                cos_row,
                sin_row,
                idx_head_dim,
                half,
                eps,
            )
            tl.store(
                iq_out_ptr
                + tok * (num_index_heads * idx_head_dim)
                + h * idx_head_dim
                + di,
                out.to(iq_out_ptr.dtype.element_ty),
            )

    slot = tl.load(slot_mapping_ptr + tok)
    page = slot // ASM_PAGE
    s = slot % ASM_PAGE
    valid_slot = slot >= 0

    k_inv = 1.0
    v_inv = 1.0
    if FP8_KV:
        k_inv = 1.0 / tl.load(k_scale_ptr)
        v_inv = 1.0 / tl.load(v_scale_ptr)

    # ----- (3) k heads (norm+rope) -> SHUFFLE K, (4) v heads (raw) -> SHUFFLE V -----
    for h in tl.static_range(num_kv_heads):
        kout = _gemma_norm_rope_head(
            row + k_base + h * head_dim,
            k_norm_w_ptr,
            cos_row,
            sin_row,
            head_dim,
            half,
            eps,
        )
        if FP8_KV:
            kout = tl.minimum(tl.maximum(kout * k_inv, -FP8_MAX), FP8_MAX)
        k_off = (
            ((page * num_kv_heads + h) * (head_dim // x) + d // x) * (ASM_PAGE * x)
            + s * x
            + (d % x)
        )
        tl.store(kc_ptr + k_off, kout.to(kc_ptr.dtype.element_ty), mask=valid_slot)

        vvals = tl.load(row + v_base + h * head_dim + d).to(tl.float32)
        if FP8_KV:
            vvals = tl.minimum(tl.maximum(vvals * v_inv, -FP8_MAX), FP8_MAX)
        v_off = (
            ((page * num_kv_heads + h) * (ASM_PAGE // x) + s // x) * (head_dim * x)
            + d * x
            + (s % x)
        )
        tl.store(vc_ptr + v_off, vvals.to(vc_ptr.dtype.element_ty), mask=valid_slot)

    # ----- (5) index_k: norm + rope -> index_cache (page-128 flat scatter) -----
    if INSERT_INDEX_K:
        ikout = _gemma_norm_rope_head(
            row + ik_base,
            ik_norm_w_ptr,
            cos_row,
            sin_row,
            idx_head_dim,
            half,
            eps,
        )
        di2 = tl.arange(0, idx_head_dim)
        idx_slot = tl.load(index_slot_mapping_ptr + tok)
        tl.store(
            index_cache_ptr + idx_slot * idx_head_dim + di2,
            ikout.to(index_cache_ptr.dtype.element_ty),
            mask=idx_slot >= 0,
        )


@torch.no_grad()
def minimax_m3_fused_qknorm_rope_shuffle_insert(
    qkv: torch.Tensor,  # [num_tokens, row_elems]
    q_norm_weight: torch.Tensor,  # [head_dim]
    k_norm_weight: torch.Tensor,  # [head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [num_tokens] int
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    slot_mapping: torch.Tensor,  # [num_tokens] int64 logical token slots
    kv_cache_k: torch.Tensor,  # SHUFFLE K [nph16, nkv, head_dim//x, 16, x]
    kv_cache_v: torch.Tensor,  # SHUFFLE V [nph16, nkv, 16//x, head_dim, x]
    q_out: torch.Tensor,  # [num_tokens, q_size] normed+roped q
    *,
    num_index_heads: int,
    idx_head_dim: int,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    skip_index_branch: bool = False,
) -> None:
    """Fused Gemma-QKnorm + partial-NeoX-RoPE + page-16 SHUFFLE KV insert.

    Replaces the AITER sparse-PA insert triple (norm/rope, reshape_and_cache,
    index-cache insert) with one Triton kernel. K/V SHUFFLE writes match
    ``aiter.reshape_and_cache(asm_layout=True)``; FP8 caches are divided by the
    scalar KV scale before the fp8 store.
    """
    num_tokens = qkv.shape[0]
    head_dim = q_norm_weight.shape[-1]
    assert head_dim == 128, "M3 fused shuffle insert requires head_dim == 128"
    assert kv_cache_k.is_contiguous() and kv_cache_v.is_contiguous()
    x = 16 // kv_cache_k.element_size()

    fp8_kv = _is_fp8_kv_cache_tensor(kv_cache_k)
    fp8_max = _FP8_MAX.get(kv_cache_k.dtype, 448.0) if fp8_kv else 0.0
    if fp8_kv:
        assert k_scale is not None and v_scale is not None, (
            "fused shuffle insert with fp8 KV cache requires scalar k/v scales"
        )

    insert_index_k = (
        not skip_index_branch
        and index_cache is not None
        and index_cache.numel() > 0
        and index_slot_mapping is not None
    )
    if insert_index_k:
        assert index_cache is not None and index_cache.is_contiguous(), (
            "index cache must be contiguous"
        )

    # Dummy 1-elem tensors keep the kernel signature stable when a branch is off.
    dummy = qkv.new_empty(1)
    _fused_qknorm_rope_shuffle_insert_kernel[(num_tokens,)](
        qkv,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight if index_q_norm_weight is not None else dummy,
        index_k_norm_weight if index_k_norm_weight is not None else dummy,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping if index_slot_mapping is not None else slot_mapping,
        q_out,
        index_q_out if index_q_out is not None else dummy,
        kv_cache_k,
        kv_cache_v,
        index_cache if index_cache is not None else dummy,
        k_scale if k_scale is not None else dummy,
        v_scale if v_scale is not None else dummy,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        head_dim=head_dim,
        idx_head_dim=idx_head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        row_elems=qkv.shape[1],
        x=x,
        ASM_PAGE=ASM_PAGE_SIZE,
        FP8_KV=fp8_kv,
        FP8_MAX=fp8_max,
        SKIP_INDEX=skip_index_branch,
        INSERT_INDEX_K=insert_index_k,
    )


@triton.jit
def _build_sparse_block_table_kernel(
    topk_ptr,  # [1, batch, topk] int32, selected logical 128-block ids
    block_table_ptr,  # [batch, max_blocks] int32, logical 128-page table
    seq_lens_ptr,  # [batch] int32
    sparse_bt_ptr,  # [batch, topk * 8] int32, physical 16-page table
    sparse_ctx_ptr,  # [batch] int32
    max_topk,
    stride_topk_n,
    stride_topk_k,
    stride_bt_b,
    stride_sbt_b,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid_b)
    last_blk = (seq_len - 1) // SPARSE_BLOCK_SIZE_C

    topk_row = topk_ptr + pid_b * stride_topk_n
    bt_row = block_table_ptr + pid_b * stride_bt_b
    sbt_row = sparse_bt_ptr + pid_b * stride_sbt_b

    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_k, mask=off_t < max_topk, other=-1)
    valid = blk >= 0
    is_tail = valid & (blk == last_blk)
    is_full = valid & (blk != last_blk)

    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    slot = tl.where(is_full, earlier_full, n_full)

    logical_page = tl.load(bt_row + blk, mask=valid, other=0).to(tl.int32)
    base_phys = logical_page * PAGES_PER_BLOCK
    dst_base = slot * PAGES_PER_BLOCK

    for j in tl.static_range(PAGES_PER_BLOCK):
        tl.store(sbt_row + dst_base + j, base_phys + j, mask=valid)

    n_used = n_valid * PAGES_PER_BLOCK
    off_w = tl.arange(0, BLOCK_SIZE_T * PAGES_PER_BLOCK)
    tl.store(sbt_row + off_w, tl.zeros_like(off_w), mask=off_w >= n_used)

    tail_tokens = seq_len - last_blk * SPARSE_BLOCK_SIZE_C
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * SPARSE_BLOCK_SIZE_C + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * SPARSE_BLOCK_SIZE_C, seq_len))
    tl.store(sparse_ctx_ptr + pid_b, ctx)


@torch.no_grad()
def minimax_m3_build_sparse_block_table(
    topk_idx: torch.Tensor,  # [1, batch, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact selected logical sparse blocks into physical page-16 tables."""
    assert topk_idx.shape[0] == 1, "AITER sparse PA requires num_kv_heads == 1"
    batch = topk_idx.shape[1]
    topk = topk_idx.shape[-1]
    width = topk * PAGES_PER_SPARSE_BLOCK
    sparse_bt = torch.empty((batch, width), dtype=torch.int32, device=topk_idx.device)
    sparse_ctx = torch.empty((batch,), dtype=torch.int32, device=topk_idx.device)
    _build_sparse_block_table_kernel[(batch,)](
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        SPARSE_BLOCK_SIZE_C=SPARSE_BLOCK_SIZE,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


@triton.jit
def _build_sparse_block_table_prefill_kernel(
    topk_ptr,  # [1, total_q, topk]
    block_table_ptr,  # [batch, max_blocks]
    req_id_ptr,  # [total_q]
    abs_pos_ptr,  # [total_q]
    sparse_bt_ptr,  # [total_q, topk * 8]
    sparse_ctx_ptr,  # [total_q]
    max_topk,
    stride_topk_n,
    stride_topk_k,
    stride_bt_b,
    stride_sbt_n,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    req_id = tl.load(req_id_ptr + pid_n)
    abs_pos = tl.load(abs_pos_ptr + pid_n)
    causal_len = abs_pos + 1
    self_blk = abs_pos // SPARSE_BLOCK_SIZE_C

    topk_row = topk_ptr + pid_n * stride_topk_n
    bt_row = block_table_ptr + req_id * stride_bt_b
    sbt_row = sparse_bt_ptr + pid_n * stride_sbt_n

    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_k, mask=off_t < max_topk, other=-1)
    valid = (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)

    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    slot = tl.where(is_full, earlier_full, n_full)

    logical_page = tl.load(bt_row + blk, mask=valid, other=0).to(tl.int32)
    base_phys = logical_page * PAGES_PER_BLOCK
    dst_base = slot * PAGES_PER_BLOCK

    for j in tl.static_range(PAGES_PER_BLOCK):
        tl.store(sbt_row + dst_base + j, base_phys + j, mask=valid)

    n_used = n_valid * PAGES_PER_BLOCK
    off_w = tl.arange(0, BLOCK_SIZE_T * PAGES_PER_BLOCK)
    tl.store(sbt_row + off_w, tl.zeros_like(off_w), mask=off_w >= n_used)

    tail_tokens = causal_len - self_blk * SPARSE_BLOCK_SIZE_C
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * SPARSE_BLOCK_SIZE_C + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * SPARSE_BLOCK_SIZE_C, causal_len))
    tl.store(sparse_ctx_ptr + pid_n, ctx)


@torch.no_grad()
def minimax_m3_build_sparse_block_table_prefill(
    topk_idx: torch.Tensor,  # [1, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    query_req_id: torch.Tensor,  # [total_q]
    query_abs_pos: torch.Tensor,  # [total_q]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one page-16 sparse block table row per prefill query token."""
    assert topk_idx.shape[0] == 1, "AITER sparse PA requires num_kv_heads == 1"
    total_q = topk_idx.shape[1]
    topk = topk_idx.shape[-1]
    width = topk * PAGES_PER_SPARSE_BLOCK
    sparse_bt = torch.empty((total_q, width), dtype=torch.int32, device=topk_idx.device)
    sparse_ctx = torch.empty((total_q,), dtype=torch.int32, device=topk_idx.device)
    _build_sparse_block_table_prefill_kernel[(total_q,)](
        topk_idx,
        block_table,
        query_req_id,
        query_abs_pos,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        SPARSE_BLOCK_SIZE_C=SPARSE_BLOCK_SIZE,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


@triton.jit
def _insert_index_cache_kernel(
    index_k_ptr,  # [num_tokens, head_dim]
    index_cache_ptr,  # [num_blocks, block_size, head_dim]
    slot_mapping_ptr,  # [num_tokens]
    stride_k_t,
    stride_k_d,
    stride_c_b,
    stride_c_t,
    stride_c_d,
    stride_slot_t,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    slot = tl.load(slot_mapping_ptr + pid_t * stride_slot_t)
    block_id = slot // CACHE_BLOCK_SIZE
    block_off = slot - block_id * CACHE_BLOCK_SIZE

    src = index_k_ptr + pid_t * stride_k_t + offs_d * stride_k_d
    dst = (
        index_cache_ptr
        + block_id * stride_c_b
        + block_off * stride_c_t
        + offs_d * stride_c_d
    )
    mask = (slot >= 0) & (offs_d < HEAD_DIM)
    value = tl.load(src, mask=offs_d < HEAD_DIM, other=0.0)
    tl.store(dst, value, mask=mask)


@torch.no_grad()
def minimax_m3_insert_index_cache(
    index_k: torch.Tensor,
    index_cache: torch.Tensor,
    index_slot_mapping: torch.Tensor,
) -> None:
    """Scatter index keys into MiniMax-M3's key-only side cache."""
    if index_k.numel() == 0 or index_cache.numel() == 0:
        return
    if index_k.dim() != 2 or index_cache.dim() != 3:
        raise ValueError("MiniMax-M3 index cache insert expects [N,D] and [B,T,D]")
    if index_k.shape[1] != index_cache.shape[2]:
        raise ValueError("MiniMax-M3 index key dim must match index cache head dim")
    if index_slot_mapping.dim() != 1 or index_slot_mapping.shape[0] != index_k.shape[0]:
        raise ValueError("MiniMax-M3 index slot mapping must be a length-N vector")
    if index_cache.stride(2) != 1:
        raise ValueError("MiniMax-M3 index cache requires contiguous head dimension")

    head_dim = index_k.shape[1]
    _insert_index_cache_kernel[(index_k.shape[0],)](
        index_k,
        index_cache,
        index_slot_mapping,
        index_k.stride(0),
        index_k.stride(1),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(2),
        index_slot_mapping.stride(0),
        CACHE_BLOCK_SIZE=index_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def _gluon_scale_arg(
    scale: torch.Tensor | None,
    *,
    num_phys_pages: int,
    num_kv_heads: int,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.numel() == 1:
        return scale
    if scale.dim() != 2 or scale.shape[0] != num_kv_heads:
        raise ValueError(
            "MiniMax-M3 AITER sparse PA supports scalar KV scales or "
            "[num_kv_heads, max_kv_tokens] scales"
        )
    max_tokens = num_phys_pages * ASM_PAGE_SIZE
    if scale.shape[1] < max_tokens:
        raise ValueError(
            "MiniMax-M3 AITER sparse PA KV scale tensor is smaller than the "
            f"cache token capacity ({scale.shape[1]} < {max_tokens})"
        )
    scale = scale[:, :max_tokens]
    return (
        scale.transpose(0, 1)
        .contiguous()
        .view(num_phys_pages, ASM_PAGE_SIZE, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
        .unsqueeze(-1)
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode_aiter(
    q: torch.Tensor,  # [batch, num_heads, head_dim]
    k_cache: torch.Tensor,  # [phys16, num_kv_heads, head_dim // x, 16, x]
    v_cache: torch.Tensor,  # [phys16, num_kv_heads, 16 // x, head_dim, x]
    topk_idx: torch.Tensor,  # [1, batch, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [batch, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    sparse_bt: torch.Tensor | None = None,
    sparse_ctx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The compacted 16-page block_table only depends on the shared top-k
    # selection + block_table + seq_lens, all identical across the sparse layers
    # that reuse a top-k group (ATOM index_topk_freq). Callers can prebuild it
    # once per group and pass it in; otherwise build here. Returns it for reuse.
    if sparse_bt is None or sparse_ctx is None:
        sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table(
            topk_idx, block_table, seq_lens
        )
    _run_gluon_decode(
        q,
        k_cache,
        v_cache,
        sparse_bt,
        sparse_ctx,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )
    return sparse_bt, sparse_ctx


@torch.no_grad()
def minimax_m3_sparse_attn_prefill_aiter(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,  # [1, total_q, topk]
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    sparse_bt: torch.Tensor | None = None,
    sparse_ctx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sparse_bt is None or sparse_ctx is None:
        total_q = q.shape[0]
        pos = torch.arange(total_q, dtype=torch.int32, device=q.device)
        query_req_id = torch.searchsorted(
            cu_seqlens_q[1:].contiguous(), pos, right=True
        )
        query_req_id = query_req_id.to(torch.int32)
        query_abs_pos = (
            prefix_lens[query_req_id] + (pos - cu_seqlens_q[query_req_id])
        ).to(torch.int32)
        sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
            topk_idx, block_table, query_req_id, query_abs_pos
        )
    _run_gluon_decode(
        q,
        k_cache,
        v_cache,
        sparse_bt,
        sparse_ctx,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )
    return sparse_bt, sparse_ctx


def _run_gluon_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )

    if not q.is_contiguous():
        q = q.contiguous()
    if not output.is_contiguous():
        raise ValueError("MiniMax-M3 AITER sparse PA output must be contiguous")

    total_q, num_q_heads, head_size = q.shape
    if head_size != 128:
        raise ValueError("MiniMax-M3 AITER sparse PA requires head_dim == 128")
    group_size = num_q_heads // num_kv_heads

    nphys16, hkv = k_cache.shape[0], k_cache.shape[1]
    k_cache_view = k_cache.view(nphys16 * hkv, 1, *k_cache.shape[2:])
    v_cache_view = v_cache.view(nphys16 * hkv, 1, *v_cache.shape[2:])
    q_view = q.view(total_q * num_kv_heads, group_size, head_size)
    out_view = output.view(total_q * num_kv_heads, group_size, head_size)

    num_seqs = total_q * num_kv_heads
    max_context_partition_num = get_recommended_splits(num_seqs, 1)
    context_partition_size = 256
    intermediate_shape = (num_seqs, 1, max_context_partition_num, group_size)
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        *intermediate_shape, head_size, dtype=q.dtype, device=q.device
    )

    is_fp8 = _is_fp8_kv_cache_tensor(k_cache)
    compute_type = aiter_dtypes.fp8 if is_fp8 else q.dtype
    if is_fp8:
        k_scale_arg = _gluon_scale_arg(
            k_scale, num_phys_pages=nphys16, num_kv_heads=hkv
        )
        v_scale_arg = _gluon_scale_arg(
            v_scale, num_phys_pages=nphys16, num_kv_heads=hkv
        )
    else:
        k_scale_arg = v_scale_arg = None

    pa_decode_gluon(
        output=out_view,
        query=q_view,
        key_cache=k_cache_view,
        value_cache=v_cache_view,
        context_lengths=sparse_ctx,
        block_tables=sparse_bt,
        softmax_scale=sm_scale,
        query_length=1,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=compute_type,
        query_scale=None,
        key_scale=k_scale_arg,
        value_scale=v_scale_arg,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
        sliding_window=-1,
        ps=True,
    )
