#!/usr/bin/env python
"""Reproduce the upstream PR #46117 'out of resource: shared memory' on gfx950.

This uses the PR's REAL _select_cfg (no monkeypatch). The bug: at TP=8 the qkv
projection has local N=1536 and K=6144 (H). For large-M prefill the selector's
``1280 < N <= 1536 and K % 512 == 0`` branch returns a 128x128x512 / num_stages=2
tile -- the 'deep-K' tile that overflows the 160 KB CDNA4 LDS. (verify_46117
removed exactly this branch.)

TOOLCHAIN-DEPENDENT: only triggers on a Triton that multi-buffers num_stages=2.
  * Triton 3.7.0 (rocm/atom-dev, rocm/pytorch-private flydsl images): OOR
        Required 272368 > 163840  (~2x of the 135168 single buffer)
  * Triton 3.6.0 (vllm/vllm-openai-rocm:nightly): single-buffers -> FITS (latent)
This is why the PR author (on 3.6.0) didn't see it but verify_46117 (3.7.0) did.

    python repro_pr46117_oor.py
"""
import torch
import vllm.model_executor.kernels.linear.mxfp8.rocm_native as rn
from vllm.platforms import current_platform

assert current_platform.is_rocm() and current_platform.supports_mx(), "needs gfx950"

# TP=8 MiniMax-M3 qkv dense shape during prefill.
M, N, K = 8192, 1536, 6144   # local N = q(1024) + KV(4*128) ; K = H = 6144

cfg = rn._select_cfg(M, N, K)
print(f"_select_cfg(M={M}, N={N}, K={K}) -> "
      f"(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) = {cfg}")

x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
w_q = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
w_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")  # E8M0 2^0

try:
    rn._mxfp8_dot_scaled_linear(x, w_q, w_scale)
    torch.cuda.synchronize()
    print("NO ERROR -- tile fit on this Triton build.")
except Exception as e:
    print(f"\nTRIGGERED: {type(e).__name__}: {e}")
