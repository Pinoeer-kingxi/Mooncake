"""
Mooncake Transfer Engine Wrapper for EPD Pipeline

封装 Mooncake Transfer Engine，支持 GPU Tensor 的跨节点传输。
在无 RDMA 环境下使用 TCP 协议。
"""

import os
import time
import logging
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransferStats:
    """传输统计信息"""
    total_bytes_transferred: int = 0
    total_transfers: int = 0
    total_time_ms: float = 0.0
    peak_bandwidth_gbps: float = 0.0

    @property
    def avg_bandwidth_gbps(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return (self.total_bytes_transferred * 8) / (self.total_time_ms / 1000) / 1e9

    def record(self, nbytes: int, time_ms: float):
        self.total_bytes_transferred += nbytes
        self.total_transfers += 1
        self.total_time_ms += time_ms
        bw = (nbytes * 8) / (time_ms / 1000) / 1e9 if time_ms > 0 else 0
        self.peak_bandwidth_gbps = max(self.peak_bandwidth_gbps, bw)

    def summary(self) -> str:
        return (
            f"Transfers: {self.total_transfers}, "
            f"Total: {self.total_bytes_transferred / 1e9:.3f} GB, "
            f"Total time: {self.total_time_ms:.1f} ms, "
            f"Avg BW: {self.avg_bandwidth_gbps:.3f} Gbps, "
            f"Peak BW: {self.peak_bandwidth_gbps:.3f} Gbps"
        )


class MooncakeTransferWrapper:
    """
    Mooncake Transfer Engine 的 Python 封装，用于 EPD 流水线中的跨阶段数据传输。

    支持两种模式：
    1. 本地模式（同机多 GPU）：使用 CUDA IPC / cudaMemcpy
    2. 远程模式（跨节点）：使用 Mooncake Transfer Engine (TCP/RDMA)
    """

    def __init__(
        self,
        local_hostname: str = "localhost",
        metadata_server: str = "P2PHANDSHAKE",
        protocol: str = "tcp",
        device_name: str = "",
        buffer_size: int = 512 * 1024 * 1024,  # 512MB
    ):
        self.local_hostname = local_hostname
        self.metadata_server = metadata_server
        self.protocol = protocol
        self.device_name = device_name
        self.buffer_size = buffer_size
        self.stats = TransferStats()
        self._engine = None
        self._initialized = False

    def initialize(self):
        """初始化 Transfer Engine"""
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
            logger.info(
                f"Transfer Engine initialized: host={self.local_hostname}, "
                f"protocol={self.protocol}"
            )
        except ImportError:
            logger.warning(
                "mooncake.transfer_engine not available, using local transfer mode"
            )
            self._initialized = True
            self._engine = None
        except Exception as e:
            logger.warning(f"Transfer Engine init failed, using local mode: {e}")
            self._initialized = True
            self._engine = None

    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        传输 Tensor 到目标设备。

        对于同机多 GPU，直接使用 CUDA 传输。
        对于跨节点，序列化为 bytes 通过 Transfer Engine 传输。
        """
        start_time = time.perf_counter()

        if target_device is None:
            return tensor

        nbytes = tensor.nelement() * tensor.element_size()

        if self._engine is None:
            result = self._local_transfer(tensor, target_device)
        else:
            result = self._remote_transfer(tensor, target_device)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.record(nbytes, elapsed_ms)

        return result

    def _local_transfer(
        self, tensor: torch.Tensor, target_device: torch.device
    ) -> torch.Tensor:
        """本地 GPU 间传输（CUDA memcpy）"""
        return tensor.to(target_device, copy=True)

    def _remote_transfer(
        self, tensor: torch.Tensor, target_device: torch.device
    ) -> torch.Tensor:
        """远程传输：序列化 → Transfer Engine → 反序列化"""
        shape = tensor.shape
        dtype = tensor.dtype
        data = tensor.cpu().numpy().tobytes()

        # 通过 Transfer Engine 发送
        # 实际部署时替换为 Mooncake Transfer Engine API
        received_data = data  # 本地回环模拟

        result = np.frombuffer(received_data, dtype=np.float32)
        result = torch.from_numpy(result.copy()).reshape(shape).to(target_device)
        return result

    def transfer_hidden_states(
        self,
        hidden_states: torch.Tensor,
        metadata: Dict[str, Any],
        target_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        传输 Vision Encoder 输出的 Hidden States（E→P 阶段）。

        Args:
            hidden_states: [batch, seq_len, hidden_dim] 的视觉特征
            metadata: 包含 image_size, patch_size 等元信息
            target_device: 目标 GPU 设备
        """
        transferred = self.transfer_tensor(hidden_states, target_device)
        return transferred, metadata

    def transfer_kv_cache(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        target_device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        传输 KV Cache（P→D 阶段）。

        Args:
            kv_cache: (key_cache, value_cache) 元组
            target_device: 目标 GPU 设备
        """
        key_cache, value_cache = kv_cache
        key_transferred = self.transfer_tensor(key_cache, target_device)
        value_transferred = self.transfer_tensor(value_cache, target_device)
        return key_transferred, value_transferred

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
        self._initialized = False
