"""
Hidden State Prefix Caching

在多模态模型中缓存 Vision Encoder 输出的 Hidden State，
实现相同图像的前缀复用，减少重复 Encoder 计算。
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
    """缓存的 Hidden State"""
    cache_key: str
    hidden_states: torch.Tensor
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    size_bytes: int = 0


class HiddenStatePrefixCache:
    """
    Hidden State 前缀缓存

    缓存 Vision Encoder 的输出，当相同图像再次输入时直接返回缓存结果，
    避免重复的 Encoder 计算。

    支持：
    1. 基于图像 hash 的精确匹配
    2. LRU 淘汰策略
    3. 可配置的缓存大小限制
    """

    def __init__(
        self,
        max_cache_size_bytes: int = 4 * 1024 * 1024 * 1024,  # 4GB
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

    def compute_cache_key(self, pixel_values: torch.Tensor) -> str:
        """计算图像 hash 作为缓存 key"""
        data = pixel_values.cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:32]

    def get(
        self, pixel_values: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        查找缓存。

        Returns:
            (hidden_states, metadata) if cache hit, None otherwise
        """
        cache_key = self.compute_cache_key(pixel_values)

        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # 检查 TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                self._evict(cache_key)
                self._total_misses += 1
                return None

            entry.last_accessed = time.time()
            entry.hit_count += 1
            self._cache.move_to_end(cache_key)
            self._total_hits += 1

            logger.debug(f"Cache HIT: key={cache_key}, hits={entry.hit_count}")
            return entry.hidden_states, entry.metadata

        self._total_misses += 1
        return None

    def put(
        self,
        pixel_values: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: Dict[str, Any],
    ):
        """写入缓存"""
        cache_key = self.compute_cache_key(pixel_values)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return

        size_bytes = hidden_states.nelement() * hidden_states.element_size()

        while self._should_evict(size_bytes):
            self._evict_lru()

        entry = CachedHiddenState(
            cache_key=cache_key,
            hidden_states=hidden_states.clone(),
            metadata=metadata.copy(),
            size_bytes=size_bytes,
        )

        self._cache[cache_key] = entry
        self._current_size_bytes += size_bytes

        logger.debug(
            f"Cache PUT: key={cache_key}, size={size_bytes / 1e6:.1f}MB"
        )

    def _should_evict(self, additional_bytes: int) -> bool:
        return (
            self._current_size_bytes + additional_bytes > self.max_cache_size_bytes
            or len(self._cache) >= self.max_entries
        )

    def _evict_lru(self):
        """淘汰最久未使用的条目"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size_bytes
            logger.debug(f"Cache EVICT (LRU): key={key}")

    def _evict(self, key: str):
        """淘汰指定条目"""
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
