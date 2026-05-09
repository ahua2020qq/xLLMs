# xLLM — Next-Generation LLM Inference Engine
# Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
# SPDX-License-Identifier: Apache-2.0
#
# This header must not be removed. All derivative works must retain this notice.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the xLLM project
# Adapted from vLLM vllm/config/cache.py

"""KV-cache configuration for xLLM inference engine."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional

CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8",
]

PrefixCachingHashAlgo = Literal["sha256", "xxhash"]


@dataclass
class CacheConfig:
    """Configuration for the KV cache in xLLM.

    Controls block size, memory utilization, data type, and prefix
    caching behaviour. Integrates with the multi-tier memory manager
    (GPU HBM → CPU DRAM → NVMe SSD) introduced in xLLM v0.1.
    """

    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16

    block_size: int = DEFAULT_BLOCK_SIZE
    """Size of a contiguous cache block in number of tokens."""

    gpu_memory_utilization: float = 0.92
    """Fraction of GPU memory to use for KV cache (0 < x <= 1)."""

    cache_dtype: CacheDType = "auto"
    """Data type for KV cache storage. 'auto' inherits from model dtype."""

    enable_prefix_caching: bool = True
    """Whether to enable automatic prefix caching."""

    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Hash algorithm for prefix-cache keys."""

    sliding_window: Optional[int] = None
    """Sliding window size for attention (None = full context)."""

    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    """Number of GPU blocks (set after profiling)."""

    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    """Number of CPU blocks (set after profiling)."""

    num_ssd_blocks: Optional[int] = field(default=None, init=False)
    """Number of SSD blocks (set after profiling). xLLM extension."""

    def effective_block_size(self) -> int:
        """Return the resolved block size, ensuring a minimum of 8."""
        return max(self.block_size, 8)

    def metrics_info(self) -> dict:
        """Return key/value pairs for Prometheus metrics export."""
        return {
            "block_size": str(self.block_size),
            "gpu_memory_utilization": str(self.gpu_memory_utilization),
            "cache_dtype": self.cache_dtype,
            "enable_prefix_caching": str(self.enable_prefix_caching),
            "num_gpu_blocks": str(self.num_gpu_blocks),
            "num_cpu_blocks": str(self.num_cpu_blocks),
            "num_ssd_blocks": str(self.num_ssd_blocks),
        }
