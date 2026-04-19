# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gluon-based unified attention kernel for AMD CDNA3/CDNA4 (gfx9xx).

This file has two kernels:

* Phase 1: ``kernel_gluon_unified_attention_2d_v1`` — single-warp, no LDS
  pipelining, direct global→register loads. Correct but slow. Kept as
  a reference / fallback.

* Phase 2: ``kernel_gluon_unified_attention_2d_v2`` — four-warp MFMA
  (``warps_per_cta=[1,4]``), explicit LDS staging via
  ``gl.amd.cdna4.async_copy.buffer_load_to_shared``, two-stage
  prefetch/wait pipeline, and the MFMA output accumulator kept in its
  native layout across iterations.

Both support the causal paged-attention common case (bf16/fp16, GQA,
sliding window, no FP8/alibi/softcap/sinks/qq_bias/mm_prefix). The
backend wrapper falls back to Triton ``unified_attention`` for anything
outside that.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: gl.constexpr):
    left: gl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


# =============================================================================
# Phase 1 kernel (single-warp reference)
# =============================================================================

@gluon.jit
def kernel_gluon_unified_attention_2d_v1(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_query_heads: gl.constexpr,
    num_queries_per_kv: gl.constexpr,
    block_table_stride: gl.int64,
    query_stride_0: gl.int64,
    query_stride_1: gl.int64,
    output_stride_0: gl.int64,
    output_stride_1: gl.int64,
    stride_k_cache_0: gl.int64,
    stride_k_cache_1: gl.int64,
    stride_k_cache_2: gl.int64,
    stride_k_cache_3: gl.constexpr,
    stride_v_cache_0: gl.int64,
    stride_v_cache_1: gl.int64,
    stride_v_cache_2: gl.int64,
    stride_v_cache_3: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    HEAD_SIZE_PADDED: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    num_seqs: gl.int32,
):
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )
    q_block_start_idx = gl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = gl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = gl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = (
        cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 1],
    )

    workLayoutQ: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    workLayoutS: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )

    offs_m_q = gl.arange(0, BLOCK_M, gl.SliceLayout(1, workLayoutQ))
    offs_d_q = gl.arange(0, HEAD_SIZE_PADDED, gl.SliceLayout(0, workLayoutQ))
    offs_m_s = gl.arange(0, BLOCK_M, gl.SliceLayout(1, workLayoutS))
    offs_t_s = gl.arange(0, TILE_SIZE, gl.SliceLayout(0, workLayoutS))

    query_pos_q = q_block_local_idx * BLOCK_Q + offs_m_q // num_queries_per_kv
    query_offset_0_q = cur_batch_in_all_start_index + query_pos_q
    query_offset_1_q = (
        kv_head_idx * num_queries_per_kv + offs_m_q % num_queries_per_kv
    )
    query_offset = (
        query_offset_0_q[:, None] * query_stride_0
        + query_offset_1_q[:, None] * query_stride_1
        + offs_d_q[None, :]
    )

    dim_mask_q = offs_d_q < HEAD_SIZE
    query_mask_0_q = query_pos_q < cur_batch_query_len
    query_mask_1_q = query_offset_1_q < num_query_heads

    Q = gl.load(
        query_ptr + query_offset,
        mask=(
            dim_mask_q[None, :]
            & query_mask_0_q[:, None]
            & query_mask_1_q[:, None]
        ),
        other=0.0,
    )

    query_pos_s = q_block_local_idx * BLOCK_Q + offs_m_s // num_queries_per_kv
    query_offset_1_s = (
        kv_head_idx * num_queries_per_kv + offs_m_s % num_queries_per_kv
    )
    query_mask_0_s = query_pos_s < cur_batch_query_len
    query_mask_1_s = query_offset_1_s < num_query_heads

    block_table_offset = seq_idx * block_table_stride

    M = gl.full(
        [BLOCK_M], float("-inf"), gl.float32,
        layout=gl.SliceLayout(1, workLayoutS),
    )
    L = gl.full(
        [BLOCK_M], 1.0, gl.float32,
        layout=gl.SliceLayout(1, workLayoutS),
    )
    acc = gl.zeros(
        [BLOCK_M, HEAD_SIZE_PADDED], gl.float32, workLayoutQ,
    )

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = gl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed = context_len + qpos_hi
        tile_start = gl.maximum(0, first_allowed // TILE_SIZE)
        tile_end = gl.minimum((last_allowed // TILE_SIZE) + 1, num_tiles)

    for j in range(tile_start, tile_end):
        seq_offset_s = j * TILE_SIZE + offs_t_s
        tile_mask_s = seq_offset_s < max_seq_prefix_len

        k_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[8, 8],
            warps_per_cta=[1, 1],
            order=[0, 1],
        )
        offs_d_k = gl.arange(0, HEAD_SIZE_PADDED, gl.SliceLayout(1, k_layout))
        offs_t_k = gl.arange(0, TILE_SIZE, gl.SliceLayout(0, k_layout))
        seq_offset_k = j * TILE_SIZE + offs_t_k
        tile_mask_k = seq_offset_k < max_seq_prefix_len
        physical_block_idx_k = gl.load(
            block_tables_ptr
            + block_table_offset
            + seq_offset_k // BLOCK_SIZE
        ).to(gl.int64)
        k_offset = (
            physical_block_idx_k[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d_k[:, None] * stride_k_cache_3
            + (seq_offset_k % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        K = gl.load(
            key_cache_ptr + k_offset,
            mask=(offs_d_k < HEAD_SIZE)[:, None] & tile_mask_k[None, :],
            other=0.0,
        )

        v_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[16, 4],
            warps_per_cta=[1, 1],
            order=[1, 0],
        )
        offs_t_v = gl.arange(0, TILE_SIZE, gl.SliceLayout(1, v_layout))
        offs_d_v = gl.arange(0, HEAD_SIZE_PADDED, gl.SliceLayout(0, v_layout))
        seq_offset_v = j * TILE_SIZE + offs_t_v
        tile_mask_v = seq_offset_v < max_seq_prefix_len
        physical_block_idx_v = gl.load(
            block_tables_ptr
            + block_table_offset
            + seq_offset_v // BLOCK_SIZE
        ).to(gl.int64)
        v_offset = (
            physical_block_idx_v[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset_v % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        V = gl.load(
            value_cache_ptr + v_offset,
            mask=tile_mask_v[:, None] & (offs_d_v < HEAD_SIZE)[None, :],
            other=0.0,
        )

        dotOpA: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mfmaLayout, k_width=16,
        )
        dotOpB: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=mfmaLayout, k_width=16,
        )
        Q_mfma = gl.convert_layout(Q, layout=dotOpA)
        K_mfma = gl.convert_layout(K, layout=dotOpB)
        S_acc = gl.zeros([BLOCK_M, TILE_SIZE], gl.float32, mfmaLayout)
        S_acc = gl.amd.cdna4.mfma(Q_mfma, K_mfma, S_acc)
        S = gl.convert_layout(S_acc, layout=workLayoutS) * scale

        query_abs_pos_s = context_len + query_pos_s[:, None]
        seq_mask = seq_offset_s[None, :] <= query_abs_pos_s
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos_s - seq_offset_s[None, :]) < SLIDING_WINDOW
            )
        S = gl.where(
            seq_mask & query_mask_1_s[:, None] & query_mask_0_s[:, None],
            S,
            float("-inf"),
        )

        m_j = gl.maximum(M, gl.max(S, axis=1))
        m_j = gl.where(m_j > float("-inf"), m_j, 0.0)
        P = gl.exp(S - m_j[:, None])
        l_j = gl.sum(P, axis=1)
        alpha = gl.exp(M - m_j)
        L = L * alpha + l_j
        M = m_j

        alpha_q = gl.convert_layout(alpha, layout=gl.SliceLayout(1, workLayoutQ))
        acc = acc * alpha_q[:, None]

        P_mfma = gl.convert_layout(P.to(V.dtype), layout=dotOpA)
        V_mfma = gl.convert_layout(V, layout=dotOpB)
        acc_mfma = gl.convert_layout(acc, layout=mfmaLayout)
        acc_mfma = gl.amd.cdna4.mfma(P_mfma, V_mfma, acc_mfma)
        acc = gl.convert_layout(acc_mfma, layout=workLayoutQ)

    L_q = gl.convert_layout(L, layout=gl.SliceLayout(1, workLayoutQ))
    acc = acc / L_q[:, None]

    output_offset = (
        query_offset_0_q[:, None] * output_stride_0
        + query_offset_1_q[:, None] * output_stride_1
        + offs_d_q[None, :]
    )
    gl.store(
        output_ptr + output_offset,
        acc.to(query_ptr.dtype.element_ty),
        mask=(
            dim_mask_q[None, :]
            & query_mask_0_q[:, None]
            & query_mask_1_q[:, None]
        ),
    )


# =============================================================================
# Phase 2 kernel (4 warps + LDS staging + 2-stage pipeline)
# =============================================================================
#
# Design choices:
# * BLOCK_M = 16, TILE_SIZE = 64 (head_size=256 path).
#   MFMA instr_shape=[16,16,32] with warps_per_cta=[1,4] tiles the
#   S = [BLOCK_M, TILE_SIZE] = [16, 64] output perfectly: each warp owns
#   one [16, 16] tile, and runs 8 K-steps to sweep HEAD_SIZE_PADDED = 256.
# * For P·V (output [BLOCK_M, HEAD_SIZE_PADDED] = [16, 256]) the same
#   layout has each warp own a [16, 64] strip along N.
# * K is staged in LDS as [HEAD_SIZE_PADDED, TILE_SIZE]; V as
#   [TILE_SIZE, HEAD_SIZE_PADDED]. Two buffers each; LDS budget =
#   2·(256·64·2 + 64·256·2) = 128 KB < 160 KB on gfx950.
# * Paged gather lookups (block_tables) are computed once per tile into
#   an int64 offset tensor, then handed to buffer_load_to_shared.

@gluon.jit
def kernel_gluon_unified_attention_2d_v2(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_query_heads: gl.constexpr,
    num_queries_per_kv: gl.constexpr,
    block_table_stride: gl.int64,
    query_stride_0: gl.int64,
    query_stride_1: gl.int64,
    output_stride_0: gl.int64,
    output_stride_1: gl.int64,
    stride_k_cache_0: gl.int64,
    stride_k_cache_1: gl.int64,
    stride_k_cache_2: gl.int64,
    stride_k_cache_3: gl.constexpr,
    stride_v_cache_0: gl.int64,
    stride_v_cache_1: gl.int64,
    stride_v_cache_2: gl.int64,
    stride_v_cache_3: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    HEAD_SIZE_PADDED: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    num_seqs: gl.int32,
):
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )
    q_block_start_idx = gl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = gl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = gl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = (
        cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # ---- MFMA layouts (4 warps along N) ----
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dotOpA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=16,
    )
    dotOpB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=16,
    )

    # ---- Work layouts (4-warp) ----
    qLoadLayout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    # K: [HEAD_SIZE_PADDED, TILE_SIZE]; HEAD is contiguous in global.
    kLoadLayout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    # V: [TILE_SIZE, HEAD_SIZE_PADDED]; HEAD is contiguous in global.
    vLoadLayout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    # ---- Q (loaded once) ----
    offs_m_q = gl.arange(0, BLOCK_M, gl.SliceLayout(1, qLoadLayout))
    offs_d_q = gl.arange(0, HEAD_SIZE_PADDED, gl.SliceLayout(0, qLoadLayout))
    query_pos_q = q_block_local_idx * BLOCK_Q + offs_m_q // num_queries_per_kv
    query_offset_0_q = cur_batch_in_all_start_index + query_pos_q
    query_offset_1_q = (
        kv_head_idx * num_queries_per_kv + offs_m_q % num_queries_per_kv
    )
    dim_mask_q = offs_d_q < HEAD_SIZE
    query_mask_0_q = query_pos_q < cur_batch_query_len
    query_mask_1_q = query_offset_1_q < num_query_heads

    query_offset = (
        query_offset_0_q[:, None] * query_stride_0
        + query_offset_1_q[:, None] * query_stride_1
        + offs_d_q[None, :]
    )
    Q_raw = gl.load(
        query_ptr + query_offset,
        mask=(
            dim_mask_q[None, :]
            & query_mask_0_q[:, None]
            & query_mask_1_q[:, None]
        ),
        other=0.0,
    )
    Q_mfma = gl.convert_layout(Q_raw, layout=dotOpA)

    # ---- Mask / offset helpers, in mfmaLayout slices ----
    offs_m_s = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfmaLayout))
    offs_t_s = gl.arange(0, TILE_SIZE, gl.SliceLayout(0, mfmaLayout))
    query_pos_s = q_block_local_idx * BLOCK_Q + offs_m_s // num_queries_per_kv
    query_offset_1_s = (
        kv_head_idx * num_queries_per_kv + offs_m_s % num_queries_per_kv
    )
    query_mask_0_s = query_pos_s < cur_batch_query_len
    query_mask_1_s = query_offset_1_s < num_query_heads

    block_table_offset = seq_idx * block_table_stride

    # ---- Online softmax state (kept in MFMA-compatible layouts) ----
    M_state = gl.full(
        [BLOCK_M], float("-inf"), gl.float32,
        layout=gl.SliceLayout(1, mfmaLayout),
    )
    L_state = gl.full(
        [BLOCK_M], 1.0, gl.float32,
        layout=gl.SliceLayout(1, mfmaLayout),
    )
    acc = gl.zeros(
        [BLOCK_M, HEAD_SIZE_PADDED], gl.float32, mfmaLayout,
    )

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = gl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed = context_len + qpos_hi
        tile_start = gl.maximum(0, first_allowed // TILE_SIZE)
        tile_end = gl.minimum((last_allowed // TILE_SIZE) + 1, num_tiles)

    offs_d_k = gl.arange(0, HEAD_SIZE_PADDED, gl.SliceLayout(1, kLoadLayout))
    offs_t_k = gl.arange(0, TILE_SIZE, gl.SliceLayout(0, kLoadLayout))
    offs_t_v = gl.arange(0, TILE_SIZE, gl.SliceLayout(1, vLoadLayout))
    offs_d_v = gl.arange(0, HEAD_SIZE_PADDED, gl.SliceLayout(0, vLoadLayout))

    # ---- Main loop (multi-warp MFMA, direct gl.load — no LDS staging) ----
    for j in range(tile_start, tile_end):
        # K tile [HEAD, TILE]
        seq_offset_k = j * TILE_SIZE + offs_t_k
        tile_mask_k = seq_offset_k < max_seq_prefix_len
        phys_k = gl.load(
            block_tables_ptr + block_table_offset + seq_offset_k // BLOCK_SIZE
        ).to(gl.int64)
        k_off = (
            phys_k[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d_k[:, None] * stride_k_cache_3
            + (seq_offset_k % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        K = gl.load(
            key_cache_ptr + k_off,
            mask=(offs_d_k < HEAD_SIZE)[:, None] & tile_mask_k[None, :],
            other=0.0,
        )

        # V tile [TILE, HEAD]
        seq_offset_v = j * TILE_SIZE + offs_t_v
        tile_mask_v = seq_offset_v < max_seq_prefix_len
        phys_v = gl.load(
            block_tables_ptr + block_table_offset + seq_offset_v // BLOCK_SIZE
        ).to(gl.int64)
        v_off = (
            phys_v[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset_v % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        V = gl.load(
            value_cache_ptr + v_off,
            mask=tile_mask_v[:, None] & (offs_d_v < HEAD_SIZE)[None, :],
            other=0.0,
        )

        # --- Q · K^T → S via MFMA (multi-warp) ---
        K_mfma = gl.convert_layout(K, layout=dotOpB)
        S = gl.zeros([BLOCK_M, TILE_SIZE], gl.float32, mfmaLayout)
        S = gl.amd.cdna4.mfma(Q_mfma, K_mfma, S)
        S = S * scale

        # Causal / sliding-window / header mask in mfmaLayout
        seq_offset_s = j * TILE_SIZE + offs_t_s
        tile_mask_s = seq_offset_s < max_seq_prefix_len
        query_abs_pos_s = context_len + query_pos_s[:, None]
        seq_mask = seq_offset_s[None, :] <= query_abs_pos_s
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos_s - seq_offset_s[None, :]) < SLIDING_WINDOW
            )
        full_mask = (
            seq_mask
            & query_mask_1_s[:, None]
            & query_mask_0_s[:, None]
            & tile_mask_s[None, :]
        )
        S = gl.where(full_mask, S, float("-inf"))

        # Online softmax update — state stays in mfmaLayout slice
        m_j = gl.maximum(M_state, gl.max(S, axis=1))
        m_j = gl.where(m_j > float("-inf"), m_j, 0.0)
        P = gl.exp(S - m_j[:, None])
        l_j = gl.sum(P, axis=1)
        alpha = gl.exp(M_state - m_j)
        L_state = L_state * alpha + l_j
        M_state = m_j

        # acc kept in mfmaLayout across iterations — no convert_layout
        acc = acc * alpha[:, None]

        # --- P · V → acc via MFMA ---
        P_dot = gl.convert_layout(P.to(V.dtype), layout=dotOpA)
        V_mfma = gl.convert_layout(V, layout=dotOpB)
        acc = gl.amd.cdna4.mfma(P_dot, V_mfma, acc)

    # ---- Epilogue: normalize + store ----
    acc = acc / L_state[:, None]

    output_offset = (
        query_offset_0_q[:, None] * output_stride_0
        + query_offset_1_q[:, None] * output_stride_1
        + offs_d_q[None, :]
    )
    acc_out = gl.convert_layout(acc, layout=qLoadLayout)
    gl.store(
        output_ptr + output_offset,
        acc_out.to(query_ptr.dtype.element_ty),
        mask=(
            dim_mask_q[None, :]
            & query_mask_0_q[:, None]
            & query_mask_1_q[:, None]
        ),
    )


def _select_tile_size(head_size: int, is_prefill: bool) -> int:
    # Keep LDS within 160 KB on gfx950 for head_size up to 512.
    if head_size >= 512:
        return 32
    if head_size >= 256:
        return 64  # Phase 2 path
    return 64 if is_prefill else 32


def gluon_unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size,
    block_table: torch.Tensor,
    use_phase2: bool = True,
):
    """Entry point. Phase-2 kernel is used for head_size==256; Phase 1
    handles the rest (including head_size==512 Gemma 4 global layers)."""
    assert causal, "GLUON_ATTN supports causal attention only"
    assert q.dtype in (torch.bfloat16, torch.float16), (
        "GLUON_ATTN supports bf16/fp16 Q only"
    )

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    is_prefill = max_seqlen_q > 1
    TILE_SIZE = _select_tile_size(head_size, is_prefill)

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    sliding_window = 1 + window_size[0] if window_size[0] >= 0 else 0

    use_v2 = use_phase2 and head_size == 256 and TILE_SIZE == 64
    kernel = (
        kernel_gluon_unified_attention_2d_v2
        if use_v2
        else kernel_gluon_unified_attention_2d_v1
    )
    num_warps = 4 if use_v2 else 1

    kernel[(total_num_q_blocks, num_kv_heads)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        query_start_len_ptr=cu_seqlens_q,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        SLIDING_WINDOW=sliding_window,
        num_seqs=num_seqs,
        num_warps=num_warps,
    )
