"""Pure helpers for serving-path layered KV transfer scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class LayeredTransferWorkerMeta:
    grouped_batches: int = 0
    grouped_bytes: int = 0
    grouped_descriptors: int = 0
    failed_batches: int = 0
    peer_buffer_batches: int = 0
    peer_buffer_bytes: int = 0
    fallback_batches: int = 0
    fallback_bytes: int = 0
    accumulated_group_delay_ms: float = 0.0
    received_group_batches: int = 0
    received_finished_reqs: int = 0
    layer_wait_calls: int = 0
    layer_wait_ms: float = 0.0
    receive_failures: int = 0
    backend_counts: Dict[str, int] = field(default_factory=dict)

    def aggregate(self, other: "LayeredTransferWorkerMeta") -> "LayeredTransferWorkerMeta":
        merged = LayeredTransferWorkerMeta(
            grouped_batches=self.grouped_batches + other.grouped_batches,
            grouped_bytes=self.grouped_bytes + other.grouped_bytes,
            grouped_descriptors=self.grouped_descriptors + other.grouped_descriptors,
            failed_batches=self.failed_batches + other.failed_batches,
            peer_buffer_batches=self.peer_buffer_batches + other.peer_buffer_batches,
            peer_buffer_bytes=self.peer_buffer_bytes + other.peer_buffer_bytes,
            fallback_batches=self.fallback_batches + other.fallback_batches,
            fallback_bytes=self.fallback_bytes + other.fallback_bytes,
            accumulated_group_delay_ms=self.accumulated_group_delay_ms + other.accumulated_group_delay_ms,
            received_group_batches=self.received_group_batches + other.received_group_batches,
            received_finished_reqs=self.received_finished_reqs + other.received_finished_reqs,
            layer_wait_calls=self.layer_wait_calls + other.layer_wait_calls,
            layer_wait_ms=self.layer_wait_ms + other.layer_wait_ms,
            receive_failures=self.receive_failures + other.receive_failures,
            backend_counts=dict(self.backend_counts),
        )
        for key, value in other.backend_counts.items():
            merged.backend_counts[key] = merged.backend_counts.get(key, 0) + value
        return merged

    def is_empty(self) -> bool:
        return (
            self.grouped_batches == 0
            and self.grouped_bytes == 0
            and self.grouped_descriptors == 0
            and self.failed_batches == 0
            and self.peer_buffer_batches == 0
            and self.peer_buffer_bytes == 0
            and self.fallback_batches == 0
            and self.fallback_bytes == 0
            and self.accumulated_group_delay_ms == 0.0
            and self.received_group_batches == 0
            and self.received_finished_reqs == 0
            and self.layer_wait_calls == 0
            and self.layer_wait_ms == 0.0
            and self.receive_failures == 0
            and not self.backend_counts
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grouped_batches": int(self.grouped_batches),
            "grouped_bytes": int(self.grouped_bytes),
            "grouped_descriptors": int(self.grouped_descriptors),
            "failed_batches": int(self.failed_batches),
            "peer_buffer_batches": int(self.peer_buffer_batches),
            "peer_buffer_bytes": int(self.peer_buffer_bytes),
            "fallback_batches": int(self.fallback_batches),
            "fallback_bytes": int(self.fallback_bytes),
            "accumulated_group_delay_ms": float(self.accumulated_group_delay_ms),
            "received_group_batches": int(self.received_group_batches),
            "received_finished_reqs": int(self.received_finished_reqs),
            "layer_wait_calls": int(self.layer_wait_calls),
            "layer_wait_ms": float(self.layer_wait_ms),
            "receive_failures": int(self.receive_failures),
            "backend_counts": {str(k): int(v) for k, v in self.backend_counts.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "LayeredTransferWorkerMeta":
        payload = dict(payload or {})
        return cls(
            grouped_batches=int(payload.get("grouped_batches", 0) or 0),
            grouped_bytes=int(payload.get("grouped_bytes", 0) or 0),
            grouped_descriptors=int(payload.get("grouped_descriptors", 0) or 0),
            failed_batches=int(payload.get("failed_batches", 0) or 0),
            peer_buffer_batches=int(payload.get("peer_buffer_batches", 0) or 0),
            peer_buffer_bytes=int(payload.get("peer_buffer_bytes", 0) or 0),
            fallback_batches=int(payload.get("fallback_batches", 0) or 0),
            fallback_bytes=int(payload.get("fallback_bytes", 0) or 0),
            accumulated_group_delay_ms=float(payload.get("accumulated_group_delay_ms", 0.0) or 0.0),
            received_group_batches=int(payload.get("received_group_batches", 0) or 0),
            received_finished_reqs=int(payload.get("received_finished_reqs", 0) or 0),
            layer_wait_calls=int(payload.get("layer_wait_calls", 0) or 0),
            layer_wait_ms=float(payload.get("layer_wait_ms", 0.0) or 0.0),
            receive_failures=int(payload.get("receive_failures", 0) or 0),
            backend_counts={
                str(key): int(value)
                for key, value in dict(payload.get("backend_counts") or {}).items()
            },
        )


def infer_group_count(total_regions: int, layers_per_group: int) -> int:
    total_regions = max(1, int(total_regions))
    layers_per_group = max(1, int(layers_per_group))
    return max(1, math.ceil(total_regions / layers_per_group))


def infer_descriptors_per_group(
    total_descriptors: int,
    *,
    total_regions: int,
    layers_per_group: int,
) -> int:
    total_descriptors = max(1, int(total_descriptors))
    groups = infer_group_count(total_regions, layers_per_group)
    return max(1, math.ceil(total_descriptors / groups))


def chunk_transfer_descriptors(
    src_ptrs: Sequence[int],
    dst_ptrs: Sequence[int],
    lengths: Sequence[int],
    *,
    descriptors_per_group: int,
    max_group_bytes: int = 0,
) -> List[Tuple[List[int], List[int], List[int]]]:
    if not (len(src_ptrs) == len(dst_ptrs) == len(lengths)):
        raise ValueError("src_ptrs, dst_ptrs and lengths must have identical lengths")
    if not src_ptrs:
        return []
    descriptors_per_group = max(1, int(descriptors_per_group))
    max_group_bytes = max(0, int(max_group_bytes))

    groups: List[Tuple[List[int], List[int], List[int]]] = []
    cur_src: List[int] = []
    cur_dst: List[int] = []
    cur_len: List[int] = []
    cur_bytes = 0

    for src, dst, size in zip(src_ptrs, dst_ptrs, lengths):
        next_would_overflow = (
            bool(cur_src)
            and (
                len(cur_src) >= descriptors_per_group
                or (max_group_bytes > 0 and cur_bytes + int(size) > max_group_bytes)
            )
        )
        if next_would_overflow:
            groups.append((cur_src, cur_dst, cur_len))
            cur_src, cur_dst, cur_len = [], [], []
            cur_bytes = 0
        cur_src.append(int(src))
        cur_dst.append(int(dst))
        cur_len.append(int(size))
        cur_bytes += int(size)

    if cur_src:
        groups.append((cur_src, cur_dst, cur_len))
    return groups
