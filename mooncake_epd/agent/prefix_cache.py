"""
Prefix Cache - Hidden State 缓存

缓存 Vision Encoder 输出，避免重复编码相同图像。
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


@dataclass
class CachedHiddenState:
    cache_key: str
    hidden_states: torch.Tensor
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    hit_count: int = 0
    size_bytes: int = 0


def _fast_tensor_hash(tensor: torch.Tensor) -> str:
    """快速张量指纹：使用 shape + dtype + 采样值的 hash，避免全量拷贝"""
    meta = f"{tuple(tensor.shape)}_{tensor.dtype}_{tensor.device}"
    flat = tensor.detach().reshape(-1)
    n = flat.numel()
    step = max(1, n // 1024)
    sample = flat[::step]
    data = meta.encode() + sample.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.md5(data).hexdigest()


class HiddenStatePrefixCache:
    """
    Hidden State 前缀缓存

    缓存 Vision Encoder 输出，相同图像直接返回缓存结果。
    使用快速采样 hash 避免全量数据拷贝开销。
    """

    def __init__(
        self,
        max_cache_size_bytes: int = 4 * 1024**3,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0,
    ):
        self.max_cache_size_bytes = max_cache_size_bytes
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CachedHiddenState] = OrderedDict()
        self._current_size_bytes = 0
        self._total_hits = 0
        self._total_misses = 0

    def get(
        self, pixel_values: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        cache_key = _fast_tensor_hash(pixel_values)
        now = time.monotonic()

        entry = self._cache.get(cache_key)
        if entry is None:
            self._total_misses += 1
            return None

        if now - entry.created_at > self.ttl_seconds:
            self._evict(cache_key)
            self._total_misses += 1
            return None

        entry.last_accessed = now
        entry.hit_count += 1
        self._cache.move_to_end(cache_key)
        self._total_hits += 1
        return entry.hidden_states, entry.metadata

    def put(
        self,
        pixel_values: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: Dict[str, Any],
    ):
        cache_key = _fast_tensor_hash(pixel_values)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return

        size_bytes = hidden_states.nelement() * hidden_states.element_size()

        while self._should_evict(size_bytes):
            self._evict_lru()

        now = time.monotonic()
        self._cache[cache_key] = CachedHiddenState(
            cache_key=cache_key,
            hidden_states=hidden_states.clone(),
            metadata=metadata.copy(),
            created_at=now,
            last_accessed=now,
            size_bytes=size_bytes,
        )
        self._current_size_bytes += size_bytes

    def _should_evict(self, additional_bytes: int) -> bool:
        return (
            self._current_size_bytes + additional_bytes > self.max_cache_size_bytes
            or len(self._cache) >= self.max_entries
        )

    def _evict_lru(self):
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size_bytes

    def _evict(self, key: str):
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size_bytes -= entry.size_bytes

    def get_stats(self) -> Dict[str, Any]:
        total = self._total_hits + self._total_misses
        return {
            "total_entries": len(self._cache),
            "cache_size_mb": self._current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_cache_size_bytes / (1024 * 1024),
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": self._total_hits / max(total, 1),
        }

    def clear(self):
        self._cache.clear()
        self._current_size_bytes = 0
