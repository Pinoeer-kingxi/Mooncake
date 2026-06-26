"""
Agent State Cloning - KVCache 零拷贝克隆

实现 Agent 工作流中的 KVCache 状态克隆，支持多并行"思考"分支。
基于引用计数的共享内存管理。
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class CloneStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class BranchState:
    """一个思考分支的状态"""
    branch_id: str
    parent_id: Optional[str]
    source_cache_id: str
    status: CloneStatus = CloneStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    tokens_generated: int = 0
    score: float = 0.0


@dataclass
class _CacheEntry:
    """内部缓存条目，跟踪物理张量和引用计数"""
    kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ref_count: int = 0
    size_bytes: int = 0


class AgentStateCloner:
    """
    Agent State Cloner

    当 Agent 需要 fork 出多个并行"思考"分支时，
    通过引用计数实现 KVCache 状态的零拷贝克隆与共享。

    核心机制：
    1. 引用计数：多个分支共享同一份 KV Cache 物理内存
    2. 写时复制（CoW）：当某个分支需要修改 KV Cache 时才分配新内存
    3. 生命周期管理：引用计数为 0 时自动回收
    """

    def __init__(self, mooncake_store=None):
        self.mooncake_store = mooncake_store
        self._caches: Dict[str, _CacheEntry] = {}
        self._branch_to_cache: Dict[str, str] = {}
        self._branches: Dict[str, BranchState] = {}
        self._clone_count = 0
        self._total_clone_time_ms = 0.0
        self._fork_counter = 0

    def register_kv_cache(
        self,
        cache_id: str,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ):
        """注册一份 KV Cache 到 Store"""
        k, v = kv_cache
        size_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
        self._caches[cache_id] = _CacheEntry(
            kv_cache=kv_cache, ref_count=1, size_bytes=size_bytes,
        )
        logger.info(f"Registered KV Cache '{cache_id}', size={size_bytes / 1024 / 1024:.1f}MB")

    def clone_state(
        self,
        source_id: str,
        branch_id: str,
        parent_branch_id: Optional[str] = None,
    ) -> BranchState:
        """
        克隆 KVCache 状态（零拷贝）。

        多个分支共享同一份物理张量，仅增加引用计数。
        """
        start_time = time.perf_counter()

        entry = self._caches.get(source_id)
        if entry is None:
            raise KeyError(f"KV Cache '{source_id}' not found")

        entry.ref_count += 1
        self._branch_to_cache[branch_id] = source_id

        branch = BranchState(
            branch_id=branch_id,
            parent_id=parent_branch_id,
            source_cache_id=source_id,
        )
        self._branches[branch_id] = branch

        clone_time_ms = (time.perf_counter() - start_time) * 1000
        self._clone_count += 1
        self._total_clone_time_ms += clone_time_ms

        logger.info(
            f"Cloned '{source_id}' -> '{branch_id}' "
            f"(zero-copy, ref={entry.ref_count}, time={clone_time_ms:.4f}ms)"
        )

        return branch

    def fork_branches(
        self,
        source_id: str,
        num_branches: int,
        source_branch_id: Optional[str] = None,
    ) -> List[BranchState]:
        """从一个状态 fork 出多个并行思考分支"""
        branches = []
        self._fork_counter += 1
        fork_id = self._fork_counter

        for i in range(num_branches):
            bid = f"{source_id}_fork{fork_id}_b{i}"
            branch = self.clone_state(
                source_id=source_id,
                branch_id=bid,
                parent_branch_id=source_branch_id,
            )
            branches.append(branch)

        logger.info(
            f"Forked {num_branches} branches from '{source_id}', "
            f"avg clone time={self._total_clone_time_ms / max(self._clone_count, 1):.4f}ms"
        )

        return branches

    def get_branch_kv_cache(
        self, branch_id: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取分支的 KV Cache"""
        cache_id = self._branch_to_cache.get(branch_id)
        if cache_id is None:
            return None
        entry = self._caches.get(cache_id)
        if entry is None:
            return None
        return entry.kv_cache

    def write_copy_on_write(
        self,
        branch_id: str,
        new_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ):
        """写时复制：当分支需要修改 KV Cache 时，分配新的物理内存"""
        old_cache_id = self._branch_to_cache.get(branch_id)
        if old_cache_id is None:
            return

        new_id = f"cow_{branch_id}"
        k, v = new_kv_cache
        size_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
        self._caches[new_id] = _CacheEntry(
            kv_cache=new_kv_cache, ref_count=1, size_bytes=size_bytes,
        )
        self._branch_to_cache[branch_id] = new_id

        self._decrement_ref(old_cache_id)

        logger.info(f"CoW for branch '{branch_id}': new cache '{new_id}'")

    def release_branch(self, branch_id: str, status: CloneStatus = CloneStatus.COMPLETED):
        """释放一个分支，减少引用计数"""
        branch = self._branches.pop(branch_id, None)
        if branch is None:
            return

        branch.status = status

        cache_id = self._branch_to_cache.pop(branch_id, None)
        if cache_id is not None:
            self._decrement_ref(cache_id)

    def _decrement_ref(self, cache_id: str):
        """减少缓存引用计数，为 0 时释放"""
        entry = self._caches.get(cache_id)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        if entry.ref_count == 0:
            del self._caches[cache_id]
            logger.debug(f"Released KV Cache '{cache_id}' (ref_count=0)")

    def prune_branches(self, keep_top_k: int):
        """剪枝：保留 top-k 得分最高的分支，释放其余分支"""
        active = [
            b for b in self._branches.values()
            if b.status == CloneStatus.ACTIVE
        ]
        active.sort(key=lambda b: b.score, reverse=True)

        pruned = 0
        for branch in active[keep_top_k:]:
            self.release_branch(branch.branch_id, status=CloneStatus.PRUNED)
            pruned += 1

        logger.info(f"Pruned {pruned} branches, kept top {keep_top_k}")

    def get_stats(self) -> Dict[str, Any]:
        """获取克隆统计"""
        return {
            "total_clones": self._clone_count,
            "avg_clone_time_ms": self._total_clone_time_ms / max(self._clone_count, 1),
            "active_branches": sum(
                1 for b in self._branches.values()
                if b.status == CloneStatus.ACTIVE
            ),
            "total_kv_caches": len(self._caches),
            "total_ref_counts": sum(e.ref_count for e in self._caches.values()),
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """估算内存使用量（物理内存，不重复计算共享）"""
        total_bytes = sum(e.size_bytes for e in self._caches.values())
        unique_bytes = sum(
            e.size_bytes for e in self._caches.values() if e.ref_count >= 1
        )
        return {
            "total_bytes": total_bytes,
            "unique_physical_bytes": unique_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "unique_physical_mb": unique_bytes / (1024 * 1024),
        }
