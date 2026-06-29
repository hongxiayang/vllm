#!/usr/bin/env python
"""Reproduce the MXFP8 dense GEMM 'out of resource: shared memory' on gfx950.

Run on the MI350X/MI355X (gfx950) box inside the vllm venv / nightly image:
    .venv/bin/python repro_oor.py     # or: python repro_oor.py

The shape-aware _select_cfg() in rocm_native.py is LDS-budget capped. Here we
monkeypatch it to return progressively larger tiles and run a real forward so
Triton actually compiles/launches each one, stopping at the first that throws
'out of resource: shared memory'.

NOTE: on the CDNA backend the matrix cores read operands from VGPRs, and the
Triton build may not double-buffer tiles through LDS, so a tile's true LDS draw
can be far below (A+B)*num_stages. That's why a single 256x128x256/ns=2 tile may
'fit' on a given nightly -- we escalate BLOCK_K (and num_stages) until even a
single buffer clearly exceeds the 160 KB gfx950 limit.
"""
import torch
import vllm.model_executor.kernels.linear.mxfp8.rocm_native as rn
from vllm.platforms import current_platform

assert current_platform.is_rocm() and current_platform.supports_mx(), "needs gfx950"

LDS_LIMIT_KB = 160


def lds_kb(bm, bn, bk, ns):
    """Worst-case LDS if operands+scales are staged through shared memory."""
    a = bm * bk
    b = bn * bk
    sc = (bm + bn) * (bk // 32)
    return (a + b + sc) * ns / 1024


# Real MXFP8 linear shape: large-M prefill (M>=1024 path). K=6144 is divisible
# by every BLOCK_K below (BLOCK_K must divide K -- the K-loop is unmasked).
M, K, N = 4096, 6144, 2048
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
w_q = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
w_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")  # E8M0 2^0

# Escalating tiles (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages). Each later
# entry needs strictly more LDS; we stop at the first that throws.
CANDIDATES = [
    (256, 128, 256, 8, 2),    # ~198 KB if double-buffered, ~99 KB single
    (128, 256, 512, 8, 2),    # ~198 KB single buffer already
    (128, 128, 1024, 8, 2),   # ~264 KB single buffer
    (256, 256, 1024, 8, 2),   # ~528 KB single buffer
    (256, 256, 2048, 8, 4),   # huge -- last resort
]

for tile in CANDIDATES:
    bm, bn, bk, _, ns = tile
    assert K % bk == 0, f"BLOCK_K={bk} must divide K={K}"
    rn._select_cfg = lambda Mm, Nn, Kk, _t=tile: _t
    est = lds_kb(bm, bn, bk, ns)
    print(
        f"trying {bm}x{bn}x{bk} ns={ns}  (~{est:.0f} KB if LDS-staged, "
        f"limit {LDS_LIMIT_KB} KB)...",
        flush=True,
    )
    try:
        rn._mxfp8_dot_scaled_linear(x, w_q, w_scale)
        torch.cuda.synchronize()
        print("   fit (ran clean)")
    except Exception as e:
        print(f"\nTRIGGERED on {bm}x{bn}x{bk} ns={ns}:\n   {type(e).__name__}: {e}")
        break
else:
    print(
        "\nNo tile triggered OOR -- this Triton build allocates LDS very leanly; "
        "bump BLOCK_K/BLOCK_M further."
    )
