# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical-equivalence tests for the gfx950 native MXFP8 GEMM launchers
(MiniMax-M3): the dense ``_select_cfg`` tile selector and the grouped-MoE
``GROUP_SIZE_M`` grid swizzle.

Both kernels run ``tl.dot_scaled`` on the *same* quantized operands a torch
reference dequantizes, so outputs must match to fp32-accumulate rounding. The
grouped-GEMM cases include decode shapes (tiny ``num_tokens_post_padded`` vs a
much larger ``sorted_token_ids`` buffer) — the regime the 1-D grid swizzle must
map correctly (``num_pid_m`` is derived from ``EM = sorted_token_ids.shape[0]``,
matching the launched grid).
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_mx()),
    reason="native MXFP8 requires CDNA4 (gfx95x)",
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _assert_close(out: torch.Tensor, ref: torch.Tensor, tag: str) -> None:
    cos = _cos(out, ref)
    rel = ((out - ref).norm() / ref.norm().clamp_min(1e-6)).item()
    assert cos >= 0.999 and rel <= 6e-2, f"{tag}: cos={cos:.6f} rel_l2={rel:.4f}"


# ---------------------------------------------------------------- dense linear
# (N, K) shards covering every _select_cfg branch; K is always %128==0.
_DENSE_SHAPES = [
    (768, 384),  # short-K shared_down
    (1024, 768),
    (1536, 1024),  # qkv-class local N
    (1280, 2048),
    (1536, 2048),  # qkv-class N, K%512==0
    (1536, 6144),  # qkv-class N, deep K (K%512==0) — LDS-sensitive tile
    (2560, 6144),  # TP=4 qkv
    (768, 6144),  # small local N, deep K
]
_DENSE_M = [1, 16, 32, 64, 65, 128, 256, 257, 1024, 4096, 8192]


@pytest.mark.parametrize("N,K", _DENSE_SHAPES)
@pytest.mark.parametrize("M", _DENSE_M)
@torch.inference_mode()
def test_dense_select_cfg_equiv(M, N, K):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.1
    w_q, w_s = mxfp8_e4m3_quantize(w)

    out = _mxfp8_dot_scaled_linear(x, w_q, w_s)

    # Reference: dequantize the same operands the kernel consumes, fp32 matmul.
    x_q, x_s = mxfp8_e4m3_quantize(x)
    deq_x = dequant_mxfp8_to_bf16(x_q, x_s).to(torch.float32)
    deq_w = dequant_mxfp8_to_bf16(w_q, w_s).to(torch.float32)
    ref = (deq_x @ deq_w.T).to(out.dtype)

    assert out.shape == (M, N)
    _assert_close(out, ref, f"dense M={M} N={N} K={K}")


# ----------------------------------------------------------------- grouped MoE
def _quant_experts(w):  # w: [E, N, K] bf16 -> (q [E,N,K], s [E,N,K//32])
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    qs, ss = [], []
    for e in range(w.shape[0]):
        q, s = mxfp8_e4m3_quantize(w[e])
        qs.append(q)
        ss.append(s)
    return torch.stack(qs), torch.stack(ss)


def _ref_grouped_gemm(
    deq_a, deq_w, sorted_ids, expert_ids, num_post, M, N, block_m, a_div, mul_weight
):
    out = torch.zeros(M, N, device=deq_a.device, dtype=torch.float32)
    sids = sorted_ids.tolist()
    eids = expert_ids.tolist()
    for idx in range(int(num_post)):
        tok = sids[idx]
        if tok >= M:
            continue
        e = eids[idx // block_m]
        if e < 0:
            continue
        row = (deq_a[tok // a_div] @ deq_w[e].T).to(torch.float32)
        if mul_weight is not None:
            row = row * float(mul_weight[tok])
        out[tok] = row
    return out


# (T tokens, top_k, N, K, E). Small T => decode regime (huge sorted_ids buffer).
_MOE_CASES = [
    (1, 8, 256, 768, 32),  # decode: num_post << sorted_ids.shape[0]
    (4, 8, 512, 1024, 32),  # decode
    (16, 8, 256, 6144, 64),  # decode, deep K
    (1024, 8, 512, 768, 32),  # prefill
    (4096, 8, 256, 1024, 16),  # large prefill
]


@pytest.mark.parametrize("T,top_k,N,K,E", _MOE_CASES)
@pytest.mark.parametrize("use_mul", [False, True])
@torch.inference_mode()
def test_grouped_gemm_swizzle_equiv(T, top_k, N, K, E, use_mul):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _grouped_gemm_mxfp8,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(0)
    dev = "cuda"
    block_m = 64
    a_div = 1 if use_mul else top_k
    M = T * top_k  # num_valid_tokens

    # gemm1 indexes A by offs_token // top_k (a has T rows); gemm2 uses a_div=1
    # (A has M rows). Mirror both via a_div.
    a_rows = T if a_div == top_k else M
    a = torch.randn(a_rows, K, device=dev, dtype=torch.bfloat16) * 0.1
    w = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.1

    topk_ids = torch.randint(0, E, (T, top_k), device=dev, dtype=torch.int32)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, E)
    assert sorted_ids.shape[0] >= int(num_post)  # buffer is over-allocated

    a_q, a_s = mxfp8_e4m3_quantize(a)
    w_q, w_s = _quant_experts(w)

    mul_weight = None
    if use_mul:
        mul_weight = torch.rand(M, device=dev, dtype=torch.float32) + 0.5

    out = _grouped_gemm_mxfp8(
        a_q,
        a_s,
        w_q,
        w_s,
        sorted_ids,
        expert_ids,
        num_post,
        num_valid_tokens=M,
        top_k=top_k,
        block_m=block_m,
        out_dtype=torch.float32,
        a_div=a_div,
        mul_weight_by=mul_weight,
    )

    deq_a = dequant_mxfp8_to_bf16(a_q, a_s).to(torch.float32)
    deq_w = dequant_mxfp8_to_bf16(w_q, w_s).to(torch.float32)
    ref = _ref_grouped_gemm(
        deq_a, deq_w, sorted_ids, expert_ids, num_post, M, N, block_m, a_div, mul_weight
    )

    assert out.shape == (M, N)
    _assert_close(out, ref, f"moe T={T} N={N} K={K} E={E} mul={use_mul}")
