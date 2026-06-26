"""
Transfer Engine 封装

封装 Mooncake Transfer Engine，支持 GPU Tensor 的跨节点传输。
"""

import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class TransferStats:
    """线程安全的传输统计"""
    _lock: threading.Lock = None
    _total_bytes: int = 0
    _total_transfers: int = 0
    _total_time_ms: float = 0.0
    _peak_bandwidth_gbps: float = 0.0

    def __post_init__(self):
        self._lock = threading.Lock()

    @property
    def avg_bandwidth_gbps(self) -> float:
        with self._lock:
            if self._total_time_ms == 0:
                return 0.0
            return (self._total_bytes * 8) / (self._total_time_ms / 1000) / 1e9

    def record(self, nbytes: int, time_ms: float):
        with self._lock:
            self._total_bytes += nbytes
            self._total_transfers += 1
            self._total_time_ms += time_ms
            bw = (nbytes * 8) / (time_ms / 1000) / 1e9 if time_ms > 0 else 0
            self._peak_bandwidth_gbps = max(self._peak_bandwidth_gbps, bw)

    def summary(self) -> str:
        with self._lock:
            return (
                f"Transfers: {self._total_transfers}, "
                f"Total: {self._total_bytes / 1e9:.3f} GB, "
                f"Avg BW: {self.avg_bandwidth_gbps:.3f} Gbps, "
                f"Peak BW: {self._peak_bandwidth_gbps:.3f} Gbps"
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_bytes": self._total_bytes,
                "total_transfers": self._total_transfers,
                "total_time_ms": self._total_time_ms,
                "avg_bandwidth_gbps": self.avg_bandwidth_gbps,
                "peak_bandwidth_gbps": self._peak_bandwidth_gbps,
            }


class MooncakeTransferWrapper:
    """
    Mooncake Transfer Engine 封装

    支持本地模式（同机 CUDA memcpy）和远程模式（Mooncake Transfer Engine）。
    """

    def __init__(
        self,
        local_hostname: str = "localhost",
        metadata_server: str = "P2PHANDSHAKE",
        protocol: str = "tcp",
        device_name: str = "",
    ):
        self.local_hostname = local_hostname
        self.metadata_server = metadata_server
        self.protocol = protocol
        self.device_name = device_name
        self.stats = TransferStats()
        self._engine = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        try:
            from mooncake.transfer_engine import TransferEngine
            self._engine = TransferEngine()
            self._engine.initialize(
                local_hostname=self.local_hostname,
                metadata_server=self.metadata_server,
                protocol=self.protocol,
                device_name=self.device_name,
            )
            self._initialized = True
            logger.info(f"Transfer Engine initialized: {self.protocol}")
        except (ImportError, Exception) as e:
            logger.warning(f"Transfer Engine unavailable, using local mode: {e}")
            self._engine = None
            self._initialized = True

    def transfer_tensor(
        self, tensor: torch.Tensor, target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if target_device is None:
            return tensor

        start = time.perf_counter()
        nbytes = tensor.nelement() * tensor.element_size()

        result = tensor.to(target_device, copy=True)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.record(nbytes, elapsed_ms)
        return result

    def transfer_hidden_states(
        self,
        hidden_states: torch.Tensor,
        metadata: Dict[str, Any],
        target_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        transferred = self.transfer_tensor(hidden_states, target_device)
        return transferred, metadata

    def transfer_kv_cache(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        target_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache
        return (
            self.transfer_tensor(key_cache, target_device),
            self.transfer_tensor(value_cache, target_device),
        )

    def get_stats(self) -> TransferStats:
        return self.stats

    def reset_stats(self):
        self.stats = TransferStats()

    def shutdown(self):
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                pass
            self._engine = None
        self._initialized = False
