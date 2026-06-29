#!/usr/bin/env python
"""Self-contained probe: compile a tl.dot_scaled MXFP8 matmul at a given tile and
report the LDS the Triton build allocates (or the 'out of resource' it throws).

No vLLM dependency -- isolates the bug to (tile, num_stages) x toolchain. The
default tile is the EXACT one PR #46117's dense _select_cfg emits for the TP=8
qkv shape (M>1024, local N=1536, K=6144): 128x128x512, num_stages=2.

    python probe_tile_lds.py                 # PR qkv tile
    python probe_tile_lds.py 256 128 256 2   # BM BN BK num_stages
"""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M, N, K,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_sk = tl.arange(0, BK // 32)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    as_ptrs = as_ptr + offs_m[:, None] * (K // 32) + offs_sk[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
    bs_ptrs = bs_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        asc = tl.load(as_ptrs)
        bsc = tl.load(bs_ptrs)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")
        a_ptrs += BK
        b_ptrs += BK
        as_ptrs += BK // 32
        bs_ptrs += BK // 32
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


BM, BN, BK, NS = (int(x) for x in (sys.argv[1:5] or [128, 128, 512, 2]))
M, N, K = max(BM, 256), max(BN, 256), max(BK, 6144)
K = (K // BK) * BK  # K divisible by BK
print(f"triton {triton.__version__}; tile {BM}x{BN}x{BK} ns={NS}; "
      f"shape M={M} N={N} K={K}")

dev = "cuda"
a = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
b = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
asx = torch.full((M, K // 32), 127, dtype=torch.uint8, device=dev)
bsx = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
c = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
try:
    h = _k[grid](a, asx, b, bsx, c, M, N, K, BM, BN, BK,
                 num_warps=8, num_stages=NS)
    torch.cuda.synchronize()
    smem = getattr(h, "metadata", None) and h.metadata.shared
    print(f"FIT. LDS allocated = {smem} bytes ({(smem or 0)/1024:.0f} KB)")
except Exception as e:
    print(f"OUT-OF-RESOURCE: {type(e).__name__}: {e}")
