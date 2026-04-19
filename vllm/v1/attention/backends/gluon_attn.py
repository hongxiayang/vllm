# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLUON_ATTN backend: Gluon-based paged attention for AMD CDNA3/CDNA4.

Wraps ``gluon_unified_attention`` for the supported common case and
falls back to vLLM's existing ``triton_unified_attention.unified_attention``
for any feature outside that subset (FP8 KV, alibi, softcap, sinks,
qq_bias, mm_prefix).
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class GluonAttentionBackend(RocmAttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 16

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Supports any head size up to 512 (Gemma4 global_head_dim).
        return 32 <= head_size <= 512

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Falls back to Triton kernel when mm_prefix is active.
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        # Falls back to Triton kernel when sinks are active.
        return True

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "GLUON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["GluonAttentionImpl"]:
        return GluonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (AttentionType.DECODER,)


class GluonAttentionImpl(RocmAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        logger.info_once(
            "Using Gluon unified attention for GluonAttentionImpl "
            "(gfx9 MFMA path, head_size=%d, sliding_window=%s)",
            head_size,
            sliding_window,
        )
        from vllm.v1.attention.ops.gluon_unified_attention import (
            gluon_unified_attention,
        )
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention as triton_unified_attention,
        )

        self._gluon_unified_attention = gluon_unified_attention
        self._triton_unified_attention = triton_unified_attention
        # Gluon kernel doesn't (yet) support quantized Q.
        self.supports_quant_query_input = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not supported"
                " for GluonAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens

        key_cache, value_cache = kv_cache.unbind(0)
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        can_use_gluon = (
            not is_quantized_kv_cache(self.kv_cache_dtype)
            and query.dtype in (torch.bfloat16, torch.float16)
            and self.alibi_slopes is None
            and (self.logits_soft_cap is None or self.logits_soft_cap == 0)
            and self.sinks is None
            and output_scale is None
        )

        if can_use_gluon:
            self._gluon_unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                block_table=block_table,
            )
            return output

        # Fallback: any unsupported feature routes to the Triton kernel.
        logger.info_once(
            "GLUON_ATTN: falling back to Triton unified_attention "
            "(quantized=%s, dtype=%s, alibi=%s, softcap=%s, sinks=%s, out_scale=%s)",
            is_quantized_kv_cache(self.kv_cache_dtype),
            query.dtype,
            self.alibi_slopes is not None,
            self.logits_soft_cap,
            self.sinks is not None,
            output_scale is not None,
        )

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,
            key.shape[1] if key is not None else self.num_kv_heads,
        )
        self._triton_unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap or 0.0,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            sinks=self.sinks,
            output_scale=output_scale,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(0)
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
