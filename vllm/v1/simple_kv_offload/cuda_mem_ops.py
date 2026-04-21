# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers."""

import ctypes
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# hipError_t / CUresult value returned when a symbol is exported but the
# underlying implementation is a stub (seen on ROCm 7.2 for
# ``hipMemcpyBatchAsync``).
_ERR_NOT_SUPPORTED_HIP = 801
_ERR_NOT_SUPPORTED_CUDA = 801  # CUDA_ERROR_NOT_SUPPORTED

# hipMemcpyKind / cudaMemcpyKind
_MEMCPY_DEFAULT = 4


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister / hipHostRegister.

    This bypasses PyTorch's CUDACachingHostAllocator which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).
    """
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


# NOTE: ``CUmemcpyAttributes`` and ``hipMemcpyAttributes`` share the same
# layout in ROCm 7.x, so a single ctypes struct definition works for both.
class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


_BATCH_MEMCPY_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint,  # CUresult / hipError_t
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
)

# Resolved lazily on first use.
_batch_memcpy_fn: Any = None
# ``hipMemcpyAsync`` / ``cudaMemcpyAsync`` fallback; resolved lazily only if
# the batch API returns NotSupported.
_memcpy_async_fn: Any = None
# Flips to False after we observe NotSupported from the batch API so
# subsequent calls skip it.
_batch_memcpy_supported: bool = True


def _resolve_batch_memcpy():
    """Resolve the platform batch-memcpy entry point (one-time).

    * CUDA: ``cuMemcpyBatchAsync`` via ``cuGetProcAddress``.
    * ROCm: ``hipMemcpyBatchAsync`` from libamdhip64 (ROCm 7.1+).

    NOTE: ROCm 7.2 ships the symbol but the implementation returns
    ``hipErrorNotSupported``; in that case ``copy_blocks`` falls back to
    per-block ``hipMemcpyAsync`` via ``_resolve_memcpy_async``.
    """
    if current_platform.is_rocm():
        lib = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
        fn = lib.hipMemcpyBatchAsync
        fn.restype = ctypes.c_uint
        fn.argtypes = [
            ctypes.c_void_p,  # dsts
            ctypes.c_void_p,  # srcs
            ctypes.c_void_p,  # sizes
            ctypes.c_size_t,  # count
            ctypes.c_void_p,  # attrs
            ctypes.c_void_p,  # attrIdxs
            ctypes.c_size_t,  # numAttrs
            ctypes.c_void_p,  # failIdx
            ctypes.c_void_p,  # stream
        ]
        return fn

    from cuda.bindings import driver as drv

    err, ptr, _ = drv.cuGetProcAddress(b"cuMemcpyBatchAsync", 12080, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGetProcAddress(cuMemcpyBatchAsync) failed: {err}")
    return _BATCH_MEMCPY_FUNC_TYPE(ptr)


def _resolve_memcpy_async():
    """Resolve per-op ``hipMemcpyAsync`` / ``cudaMemcpyAsync`` (ROCm fallback)."""
    lib_name = "libamdhip64.so" if current_platform.is_rocm() else "libcudart.so"
    sym = "hipMemcpyAsync" if current_platform.is_rocm() else "cudaMemcpyAsync"
    lib = ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
    fn = getattr(lib, sym)
    fn.restype = ctypes.c_uint
    fn.argtypes = [
        ctypes.c_void_p,  # dst
        ctypes.c_void_p,  # src
        ctypes.c_size_t,  # sizeBytes
        ctypes.c_int,  # kind (hipMemcpyKind / cudaMemcpyKind)
        ctypes.c_void_p,  # stream
    ]
    return fn


def _clear_last_error() -> None:
    """Clear the sticky last error on the current HIP/CUDA context.

    Needed because ``hipMemcpyBatchAsync`` on ROCm 7.2 returns a stub
    ``hipErrorNotSupported`` that remains sticky — subsequent torch ops
    on the device would otherwise surface the stale error.
    """
    lib_name = "libamdhip64.so" if current_platform.is_rocm() else "libcudart.so"
    sym = "hipGetLastError" if current_platform.is_rocm() else "cudaGetLastError"
    lib = ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
    fn = getattr(lib, sym)
    fn.restype = ctypes.c_uint
    fn()


class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    bpb: np.ndarray  # [num_layers] uint64 — bytes per block
    num_layers: int
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    # NOTE: cuMemcpyBatchAsync_v2() removed fail_idx field, but we use
    # cuMemcpyBatchAsync() with fail_idx for backward compatibility
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream / hipStream_t


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> BatchMemcpyParams:
    global _batch_memcpy_fn
    if _batch_memcpy_fn is None:
        _batch_memcpy_fn = _resolve_batch_memcpy()

    assert list(src_caches.keys()) == list(dst_caches.keys())
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases, bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        assert s_bpb == d.stride(0) * d.element_size()
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    # ``srcAccessOrder=3`` == CU_MEMCPY_SRC_ACCESS_ORDER_ANY /
    # hipMemcpySrcAccessOrderAny. See
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6f1ff58e3065df3eb4b573dba77ad31f  # noqa: E501
    attrs = _CUmemcpyAttributes(srcAccessOrder=3)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=stream.cuda_stream,
    )


def _copy_blocks_batch(
    n: int,
    params: BatchMemcpyParams,
    src_all: np.ndarray,
    dst_all: np.ndarray,
    sz_all: np.ndarray,
) -> int:
    """Call the batch memcpy API. Returns the driver error code."""
    total = n * params.num_layers
    return _batch_memcpy_fn(
        dst_all.ctypes.data,
        src_all.ctypes.data,
        sz_all.ctypes.data,
        total,
        ctypes.addressof(params.attrs),
        ctypes.byref(params.attrs_idx),
        1,
        ctypes.byref(params.fail_idx),
        params.stream_handle,
    )


def _copy_blocks_per_op(
    params: BatchMemcpyParams,
    src_all: np.ndarray,
    dst_all: np.ndarray,
    sz_all: np.ndarray,
) -> None:
    """Fallback: issue one ``hipMemcpyAsync`` per (block, layer) on the stream.

    Used when the batch API returns NotSupported (e.g. ROCm 7.2 stub).
    """
    global _memcpy_async_fn
    if _memcpy_async_fn is None:
        _memcpy_async_fn = _resolve_memcpy_async()
    stream = params.stream_handle
    for dst_ptr, src_ptr, sz in zip(dst_all, src_all, sz_all):
        err = _memcpy_async_fn(
            int(dst_ptr), int(src_ptr), int(sz), _MEMCPY_DEFAULT, stream
        )
        if err != 0:
            raise RuntimeError(f"per-op memcpy async failed: err={err}")


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Copy blocks via cuMemcpyBatchAsync / hipMemcpyBatchAsync.

    Falls back to per-op ``hipMemcpyAsync`` if the batch API returns
    NotSupported on the current platform (ROCm 7.2 ships the symbol but
    the implementation is a stub).
    """
    global _batch_memcpy_supported
    n = len(src_block_ids)
    if n == 0:
        return

    src_ids = np.array(src_block_ids, dtype=np.uint64)
    dst_ids = np.array(dst_block_ids, dtype=np.uint64)

    src_all = (
        params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
    ).ravel()
    dst_all = (
        params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
    ).ravel()
    sz_all = np.repeat(params.bpb, n)

    if _batch_memcpy_supported:
        err = _copy_blocks_batch(n, params, src_all, dst_all, sz_all)
        if err == 0:
            return
        if err == _ERR_NOT_SUPPORTED_HIP:
            logger.warning(
                "Batch memcpy API returned NotSupported; falling back to "
                "per-op async memcpy for the remainder of this process."
            )
            _batch_memcpy_supported = False
            # ROCm 7.2's stub leaves the error sticky on the context; clear
            # it so subsequent device work doesn't surface the stale error.
            _clear_last_error()
        else:
            raise RuntimeError(
                f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
            )

    _copy_blocks_per_op(params, src_all, dst_all, sz_all)
