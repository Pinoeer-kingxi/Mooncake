"""
Agent State Cloning - KVCache 零拷贝克隆

实现 Agent 工作流中的 KVCache 状态克隆，支持多并行"思考"分支。
基于 Mooncake Store 的引用计数和共享内存管理。
"""

import time
import copy
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
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
    kv_cache_ref: Optional[Tuple[torch.Tensor, torch.Tensor]]
    status: CloneStatus = CloneStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    tokens_generated: int = 0
    score: float = 0.0


class AgentStateCloner:
    """
    Agent State Cloner

    当 Agent 需要 fork 出多个并行"思考"分支时，
    通过 Mooncake Store 实现 KVCache 状态的零拷贝克隆与跨节点共享。

    核心机制：
    1. 引用计数：多个分支共享同一份 KV Cache 物理内存
    2. 写时复制（CoW）：当某个分支需要修改 KV Cache 时才分配新内存
    3. 生命周期管理：引用计数为 0 时自动回收
    """

    def __init__(self, mooncake_store=None):
        self.mooncake_store = mooncake_store
        self._ref_counts: Dict[str, int] = {}
        self._kv_store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._branches: Dict[str, BranchState] = {}
        self._clone_count = 0
        self._total_clone_time_ms = 0.0

    def register_kv_cache(
        self,
        cache_id: str,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        注册一份 KV Cache 到 Store。

        Args:
            cache_id: 唯一标识符
            kv_cache: (key_cache, value_cache)
        """
        self._kv_store[cache_id] = kv_cache
        self._ref_counts[cache_id] = 1
        logger.info(f"Registered KV Cache '{cache_id}', ref_count=1")

    def clone_state(
        self,
        source_id: str,
        branch_id: str,
        parent_branch_id: Optional[str] = None,
    ) -> BranchState:
        """
        克隆 KVCache 状态（零拷贝）。

        通过增加引用计数实现零拷贝克隆，
        多个分支共享同一份物理内存直到某个分支需要修改。

        Args:
            source_id: 源 KV Cache ID
            branch_id: 新分支 ID
            parent_branch_id: 父分支 ID

        Returns:
            BranchState: 新分支的状态
        """
        start_time = time.perf_counter()

        if source_id not in self._kv_store:
            raise KeyError(f"KV Cache '{source_id}' not found")

        kv_cache = self._kv_store[source_id]

        branch_kv_id = f"{source_id}_branch_{branch_id}"
        self._kv_store[branch_kv_id] = kv_cache  # 共享引用，零拷贝
        self._ref_counts[branch_kv_id] = 1
        self._ref_counts[source_id] = self._ref_counts.get(source_id, 0) + 1

        branch = BranchState(
            branch_id=branch_id,
            parent_id=parent_branch_id,
            kv_cache_ref=kv_cache,
        )
        self._branches[branch_id] = branch

        clone_time_ms = (time.perf_counter() - start_time) * 1000
        self._clone_count += 1
        self._total_clone_time_ms += clone_time_ms

        logger.info(
            f"Cloned '{source_id}' → branch '{branch_id}' "
            f"(zero-copy, time={clone_time_ms:.3f}ms)"
        )

        return branch

    def fork_branches(
        self,
        source_id: str,
        num_branches: int,
        source_branch_id: Optional[str] = None,
    ) -> List[BranchState]:
        """
        从一个状态 fork 出多个并行思考分支。

        Args:
            source_id: 源 KV Cache ID
            num_branches: 分支数量
            source_branch_id: 源分支 ID

        Returns:
            List[BranchState]: 所有新分支
        """
        branches = []
        for i in range(num_branches):
            bid = f"{source_id}_fork_{i}_{int(time.time() * 1000)}"
            branch = self.clone_state(
                source_id=source_id,
                branch_id=bid,
                parent_branch_id=source_branch_id,
            )
            branches.append(branch)

        logger.info(
            f"Forked {num_branches} branches from '{source_id}', "
            f"avg clone time={self._total_clone_time_ms / max(self._clone_count, 1):.3f}ms"
        )

        return branches

    def get_branch_kv_cache(self, branch_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取分支的 KV Cache"""
        branch = self._branches.get(branch_id)
        if branch is None:
            return None
        return branch.kv_cache_ref

    def write_copy_on_write(
        self,
        branch_id: str,
        new_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        写时复制：当分支需要修改 KV Cache 时，分配新的物理内存。

        Args:
            branch_id: 分支 ID
            new_kv_cache: 修改后的 KV Cache
        """
        branch = self._branches.get(branch_id)
        if branch is None:
            return

        old_ref = branch.kv_cache_ref
        branch.kv_cache_ref = new_kv_cache

        # 为新数据设置独立引用
        new_id = f"cow_{branch_id}"
        self._kv_store[new_id] = new_kv_cache
        self._ref_counts[new_id] = 1

        # 减少旧数据引用
        for cache_id, kv in self._kv_store.items():
            if kv is old_ref and cache_id != new_id:
                self._ref_counts[cache_id] = max(0, self._ref_counts[cache_id] - 1)
                if self._ref_counts[cache_id] == 0:
                    self._release(cache_id)

        logger.info(f"CoW for branch '{branch_id}': allocated new KV Cache")

    def release_branch(self, branch_id: str):
        """释放一个分支，减少引用计数"""
        branch = self._branches.pop(branch_id, None)
        if branch is None:
            return

        branch.status = CloneStatus.COMPLETED

        # 减少所有相关 KV Cache 的引用
        for cache_id in list(self._kv_store.keys()):
            if branch_id in cache_id:
                self._ref_counts[cache_id] = max(0, self._ref_counts.get(cache_id, 0) - 1)
                if self._ref_counts[cache_id] == 0:
                    self._release(cache_id)

    def prune_branches(self, keep_top_k: int, score_key: str = "score"):
        """
        剪枝：保留 top-k 得分最高的分支，释放其余分支。

        用于 Tree-of-Thought 等搜索策略。
        """
        active_branches = [
            b for b in self._branches.values()
            if b.status == CloneStatus.ACTIVE
        ]
        active_branches.sort(key=lambda b: b.score, reverse=True)

        pruned = 0
        for branch in active_branches[keep_top_k:]:
            branch.status = CloneStatus.PRUNED
            self.release_branch(branch.branch_id)
            pruned += 1

        logger.info(f"Pruned {pruned} branches, kept top {keep_top_k}")

    def _release(self, cache_id: str):
        """释放引用计数为 0 的 KV Cache"""
        if cache_id in self._kv_store:
            del self._kv_store[cache_id]
        if cache_id in self._ref_counts:
            del self._ref_counts[cache_id]
        logger.debug(f"Released KV Cache '{cache_id}' (ref_count=0)")

    def get_stats(self) -> Dict[str, Any]:
        """获取克隆统计"""
        return {
            "total_clones": self._clone_count,
            "avg_clone_time_ms": self._total_clone_time_ms / max(self._clone_count, 1),
            "active_branches": sum(
                1 for b in self._branches.values()
                if b.status == CloneStatus.ACTIVE
            ),
            "total_kv_caches": len(self._kv_store),
            "total_ref_counts": sum(self._ref_counts.values()),
        }

    def get_memory_usage(self) -> Dict[str, int]:
        """估算内存使用量"""
        total_bytes = 0
        for cache_id, (k, v) in self._kv_store.items():
            ref = self._ref_counts.get(cache_id, 0)
            # 只有 ref_count == 1 时才计入实际占用
            if ref == 1:
                total_bytes += k.nelement() * k.element_size()
                total_bytes += v.nelement() * v.element_size()
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
        }
