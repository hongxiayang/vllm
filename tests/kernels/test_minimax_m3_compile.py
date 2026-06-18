# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torch.compile-compatibility guards for the MiniMax-M3 AMD/ROCm kernels.

Part of enabling ``@support_torch_compile`` on the MM3 AMD backbone. These tests
pin the Phase-0 findings (see ``docker/minimax-m3-day0/m3-torch-compile-*.md``):

  * The pure norm / activation / native-MXFP8-linear Triton kernels are captured
    by torch.compile's user-defined-Triton HOP and survive ``fullgraph=True`` —
    including the ``if M >= 1024`` tile-regime branch under ``dynamic=True`` — with
    numerics identical to eager. These run INLINE in the compiled graph.

  * The fused SwiGLU-OAI + MXFP8-quant kernel (``_swiglu_oai_quant_kernel``) is
    NOT directly Inductor-compilable on the AMD backend: Inductor's worker
    re-lowers the user Triton kernel and the ROCm path fails to legalize an
    ``f64 -> f8E4M3FN`` ``tt.fp_to_fp`` conversion. It is only safe because the
    whole MoE expert path runs behind the opaque ``vllm::moe_forward_shared``
    custom op, so Inductor never re-lowers it. This module guards that invariant
    so the MoE op boundary is not accidentally removed.

Hardware scope: ROCm-only. The native MXFP8 ``dot_scaled`` linear is gfx95x-only
(CDNA4 microscaling); gfx942 uses the BF16 emulation path.

Run:  pytest tests/kernels/test_minimax_m3_compile.py -v
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MiniMax-M3 AMD kernels require ROCm.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    _mxfp8_e4m3_quantize_torch,
)
from vllm.models.minimax_m3.amd.ops import (  # noqa: E402
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    swiglu_oai_split,
)

DEVICE = "cuda"
EPS = 1e-6


def _gcn_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # pragma: no cover - no device / non-AMD
        return ""


requires_gfx950 = pytest.mark.skipif(
    "gfx95" not in _gcn_arch(),
    reason="native MXFP8 dot_scaled is a CDNA4 (gfx95x) feature.",
)


def _as_tensor(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _compile_fullgraph_matches(fn, args, *, atol: float):
    """Compile ``fn`` with fullgraph=True; assert it runs and matches eager.

    ``fullgraph=True`` makes torch.compile raise on any graph break, so a clean
    return is itself the "no graph break" assertion.
    """
    torch._dynamo.reset()
    ref = _as_tensor(fn(*args))
    cfn = torch.compile(fn, fullgraph=True, backend="inductor")
    got = _as_tensor(cfn(*args))
    assert got.shape == ref.shape
    max_abs = (got.float() - ref.float()).abs().max().item()
    assert max_abs <= atol, f"compiled vs eager max|Δ|={max_abs} > {atol}"
    return max_abs


# --------------------------------------------------------------------------- #
# Norm / activation kernels: captured by the Triton HOP, run inline.
# --------------------------------------------------------------------------- #
# The same Triton kernel runs in eager and compiled; Inductor may reorder the
# surrounding cast/scale, costing at most ~1 ULP. These guards assert "fullgraph
# compiles (no graph break) AND numerically equivalent", not bit-exactness.
_NORM_ATOL = 2e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compile_gemma_rmsnorm_fullgraph(dtype):
    torch.manual_seed(0)
    h = 2048
    x = torch.randn(17, h, device=DEVICE, dtype=dtype) * 0.5
    w = torch.randn(h, device=DEVICE, dtype=dtype) * 0.1
    _compile_fullgraph_matches(
        lambda x, w: gemma_rmsnorm(x, w, EPS), (x, w), atol=_NORM_ATOL
    )


def test_compile_gemma_fused_add_rmsnorm_fullgraph():
    torch.manual_seed(0)
    h = 2048
    x = torch.randn(17, h, device=DEVICE, dtype=torch.bfloat16) * 0.5
    res = torch.randn(17, h, device=DEVICE, dtype=torch.bfloat16) * 0.5
    w = torch.randn(h, device=DEVICE, dtype=torch.bfloat16) * 0.1
    # returns (normed, new_residual); compare the normed output.
    _compile_fullgraph_matches(
        lambda x, r, w: gemma_fused_add_rmsnorm(x, r, w, EPS),
        (x, res, w),
        atol=_NORM_ATOL,
    )


def test_compile_swiglu_oai_split_fullgraph():
    torch.manual_seed(0)
    g = torch.randn(33, 2 * 1408, device=DEVICE, dtype=torch.bfloat16) * 0.5
    _compile_fullgraph_matches(
        lambda x: swiglu_oai_split(x, 1.702, 1.0, None), (g,), atol=_NORM_ATOL
    )


# --------------------------------------------------------------------------- #
# Native MXFP8 dense linear: compiles fullgraph at BOTH tile regimes and under
# dynamic shapes (the ``if M >= 1024`` branch becomes a guard/specialization).
# --------------------------------------------------------------------------- #
@requires_gfx950
@pytest.mark.parametrize("m", [2048, 64])  # prefill (>=1024) and decode (<1024)
@pytest.mark.parametrize("dynamic", [False, True])
@torch.inference_mode()
def test_compile_mxfp8_native_linear_fullgraph(m, dynamic):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )

    torch.manual_seed(0)
    n, k = 2048, 4096
    w_bf16 = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16, is_sf_swizzled_layout=False)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5

    torch._dynamo.reset()
    ref = _mxfp8_dot_scaled_linear(x, w_fp8, w_scale)
    cfn = torch.compile(
        _mxfp8_dot_scaled_linear, fullgraph=True, dynamic=dynamic, backend="inductor"
    )
    got = cfn(x, w_fp8, w_scale)
    assert got.shape == (m, n)
    # Same quantized inputs and kernel -> bit-identical to eager.
    assert (got.float() - ref.float()).abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
# Invariant: the MoE expert path (which contains the AMD-Inductor-hostile fused
# SwiGLU+MXFP8-quant kernel) MUST stay behind an opaque custom op. If this op or
# its fake impl disappears, the MoE kernels would be re-lowered by Inductor and
# fail to compile on ROCm -> guard it here.
# --------------------------------------------------------------------------- #
def test_moe_expert_path_is_opaque_custom_op():
    # Importing the runner registers vllm::moe_forward{,_shared}.
    import vllm.model_executor.layers.fused_moe.runner.moe_runner  # noqa: F401

    assert hasattr(torch.ops.vllm, "moe_forward_shared"), (
        "vllm::moe_forward_shared op missing — MM3 native-MXFP8 MoE would be "
        "re-lowered by Inductor and fail to compile on ROCm "
        "(f64->f8E4M3FN legalization)."
    )
    assert hasattr(torch.ops.vllm, "moe_forward")


# --------------------------------------------------------------------------- #
# The sparse-attention core is an opaque splitting op so torch.compile splits
# the FX graph there (qkv_proj/o_proj GEMMs stay compiled). Guard the op
# registration and its membership in CompilationConfig._attention_ops.
# --------------------------------------------------------------------------- #
def test_sparse_attention_is_registered_splitting_op():
    # Importing the AMD model registers vllm::minimax_m3_sparse_attention.
    import vllm.models.minimax_m3.amd.model  # noqa: F401
    from vllm.config.compilation import CompilationConfig

    assert hasattr(torch.ops.vllm, "minimax_m3_sparse_attention"), (
        "vllm::minimax_m3_sparse_attention op missing — the M3 sparse core "
        "would be traced into the Inductor region and break compile."
    )
    assert "vllm::minimax_m3_sparse_attention" in CompilationConfig._attention_ops, (
        "sparse-attention op not in _attention_ops -> FX graph won't split on it."
    )
