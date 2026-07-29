# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm KDA attention for Kimi K3.

Keeps the AMD kernel binding out of the shared layer
"""

from collections.abc import Callable

from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)


class KimiGatedDeltaNetAttentionROCm(KimiGatedDeltaNetAttention):
    @staticmethod
    def _kda_kernels() -> tuple[Callable, Callable, Callable]:
        from vllm.models.kimi_k3.amd.ops.third_party.kda import (
            chunk_kda_with_fused_gate,
            fused_recurrent_kda,
            fused_recurrent_kda_packed_decode,
        )

        return (
            chunk_kda_with_fused_gate,
            fused_recurrent_kda,
            fused_recurrent_kda_packed_decode,
        )
