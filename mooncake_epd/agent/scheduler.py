"""
Agent PD 调度策略

根据 Agent 任务类型（思考型 vs 交互型）动态路由到不同的 P/D 资源。
- 思考型 Agent：高算力 Prefill 资源（长序列、复杂推理）
- 交互型 Agent：低延迟 Decode 资源（快速响应、多轮对话）
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AgentType(Enum):
    THINKING = "thinking"     # 思考型：需要高算力 Prefill
    INTERACTIVE = "interactive"  # 交互型：需要低延迟 Decode
    HYBRID = "hybrid"         # 混合型


@dataclass
class AgentRequest:
    """Agent 推理请求"""
    request_id: str
    agent_type: AgentType
    input_ids: Optional[Any] = None
    pixel_values: Optional[Any] = None
    priority: int = 0          # 优先级，越大越优先
    max_tokens: int = 256
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerLoad:
    """Worker 负载信息"""
    worker_id: str
    worker_type: str          # "prefill" or "decode"
    current_load: float = 0.0  # 0.0 ~ 1.0
    queue_size: int = 0
    gpu_utilization: float = 0.0
    avg_latency_ms: float = 0.0
    max_capacity: int = 1


class AgentPDScheduler:
    """
    Agent PD Disaggregation 调度策略

    根据 Agent 类型动态路由：
    - 思考型 Agent → 高算力 Prefill Worker（大 batch, 长 context）
    - 交互型 Agent → 低延迟 Decode Worker（小 batch, 快速响应）

    支持动态负载均衡和优先级调度。
    """

    def __init__(
        self,
        prefill_workers: Optional[List[str]] = None,
        decode_workers: Optional[List[str]] = None,
    ):
        self.prefill_workers = prefill_workers or ["prefill_0"]
        self.decode_workers = decode_workers or ["decode_0"]
        self._worker_loads: Dict[str, WorkerLoad] = {}
        self._pending_queue: deque = deque()
        self._routing_log: List[Dict[str, Any]] = []

        for wid in self.prefill_workers:
            self._worker_loads[wid] = WorkerLoad(
                worker_id=wid, worker_type="prefill"
            )
        for wid in self.decode_workers:
            self._worker_loads[wid] = WorkerLoad(
                worker_id=wid, worker_type="decode"
            )

    def route(self, request: AgentRequest) -> Dict[str, str]:
        """
        根据 Agent 类型路由到合适的 P/D Worker。

        Returns:
            {"prefill_worker": worker_id, "decode_worker": worker_id}
        """
        prefill_worker = self._select_prefill_worker(request)
        decode_worker = self._select_decode_worker(request)

        routing = {
            "prefill_worker": prefill_worker,
            "decode_worker": decode_worker,
        }

        self._routing_log.append({
            "request_id": request.request_id,
            "agent_type": request.agent_type.value,
            "routing": routing,
            "timestamp": time.time(),
        })

        logger.info(
            f"Routed {request.request_id} ({request.agent_type.value}): "
            f"P={prefill_worker}, D={decode_worker}"
        )

        return routing

    def _select_prefill_worker(self, request: AgentRequest) -> str:
        """选择 Prefill Worker"""
        if request.agent_type == AgentType.THINKING:
            # 思考型：选择算力最强的 Worker（GPU utilization 最低 = 剩余算力最多）
            return self._select_least_loaded(
                [wid for wid in self.prefill_workers],
                prefer_high_capacity=True,
            )
        else:
            # 交互型：选择延迟最低的 Worker
            return self._select_least_loaded(
                [wid for wid in self.prefill_workers],
                prefer_low_latency=True,
            )

    def _select_decode_worker(self, request: AgentRequest) -> str:
        """选择 Decode Worker"""
        if request.agent_type == AgentType.INTERACTIVE:
            # 交互型：选择延迟最低的 Decode Worker
            return self._select_least_loaded(
                [wid for wid in self.decode_workers],
                prefer_low_latency=True,
            )
        else:
            return self._select_least_loaded(
                [wid for wid in self.decode_workers],
            )

    def _select_least_loaded(
        self,
        workers: List[str],
        prefer_high_capacity: bool = False,
        prefer_low_latency: bool = False,
    ) -> str:
        """选择负载最低的 Worker"""
        if not workers:
            return "default"

        def score(wid: str) -> float:
            load = self._worker_loads.get(wid)
            if load is None:
                return 0.0
            s = load.current_load + load.queue_size * 0.1
            if prefer_low_latency:
                s += load.avg_latency_ms / 1000.0
            if prefer_high_capacity:
                s -= load.max_capacity * 0.01
            return s

        return min(workers, key=score)

    def update_load(self, worker_id: str, **kwargs):
        """更新 Worker 负载"""
        if worker_id in self._worker_loads:
            load = self._worker_loads[worker_id]
            for k, v in kwargs.items():
                if hasattr(load, k):
                    setattr(load, k, v)

    def batch_route(self, requests: List[AgentRequest]) -> List[Dict[str, str]]:
        """批量路由，按优先级排序"""
        sorted_reqs = sorted(requests, key=lambda r: -r.priority)
        return [self.route(r) for r in sorted_reqs]

    def get_stats(self) -> Dict[str, Any]:
        """获取调度统计"""
        return {
            "total_routed": len(self._routing_log),
            "routing_by_type": self._routing_distribution(),
            "worker_loads": {
                wid: {
                    "load": wl.current_load,
                    "queue": wl.queue_size,
                    "gpu_util": wl.gpu_utilization,
                    "avg_latency_ms": wl.avg_latency_ms,
                }
                for wid, wl in self._worker_loads.items()
            },
        }

    def _routing_distribution(self) -> Dict[str, int]:
        """统计各类型 Agent 的路由分布"""
        dist: Dict[str, int] = {}
        for entry in self._routing_log:
            at = entry["agent_type"]
            dist[at] = dist.get(at, 0) + 1
        return dist
