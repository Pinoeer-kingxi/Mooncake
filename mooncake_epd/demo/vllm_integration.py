"""
vLLM + MooncakeConnector 集成 Demo

使用 vLLM 的 MooncakeConnector 实现 Qwen3-VL 的 EPD 分离推理。
支持 V1 后端的 disaggregated prefill-decode 模式。
"""

import os
import sys
import json
import time
import subprocess
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PREFILL_PORT = 8100
DECODE_PORT = 8200
PROXY_PORT = 8000


@dataclass
class VLLMDisaggConfig:
    """vLLM Disaggregated Serving 配置"""
    model: str = DEFAULT_MODEL
    prefill_port: int = PREFILL_PORT
    decode_port: int = DECODE_PORT
    proxy_port: int = PROXY_PORT
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    kv_buffer_size: int = 2_000_000_000  # 2GB
    protocol: str = "tcp"  # "tcp" or "rdma"
    device_name: str = ""
    metadata_server: str = "127.0.0.1:2379"
    metadata_backend: str = "etcd"
    prefill_ip: str = "127.0.0.1"
    decode_ip: str = "127.0.0.1"

    def to_mooncake_json(self) -> Dict[str, Any]:
        """生成 mooncake.json 配置"""
        return {
            "prefill_url": f"{self.prefill_ip}:{self.prefill_port + 1000}",
            "decode_url": f"{self.decode_ip}:{self.decode_port + 1000}",
            "metadata_server": self.metadata_server,
            "metadata_backend": self.metadata_backend,
            "protocol": self.protocol,
            "device_name": self.device_name,
        }


def generate_configs(
    output_dir: str,
    config: Optional[VLLMDisaggConfig] = None,
) -> Dict[str, str]:
    """生成所有配置文件"""
    if config is None:
        config = VLLMDisaggConfig()

    os.makedirs(output_dir, exist_ok=True)
    files = {}

    # mooncake.json
    mooncake_config = config.to_mooncake_json()
    mooncake_path = os.path.join(output_dir, "mooncake.json")
    with open(mooncake_path, "w") as f:
        json.dump(mooncake_config, f, indent=2)
    files["mooncake_json"] = mooncake_path

    # prefill command
    prefill_script = f"""#!/bin/bash
# Prefill Node - VLM Vision Encoder + Prefill
export MOONCAKE_CONFIG_PATH={mooncake_path}

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \\
    --model {config.model} \\
    --port {config.prefill_port} \\
    --tensor-parallel-size {config.tensor_parallel_size} \\
    --max-model-len {config.max_model_len} \\
    --gpu-memory-utilization {config.gpu_memory_utilization} \\
    --kv-transfer-config '{{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}}'
"""
    prefill_path = os.path.join(output_dir, "start_prefill.sh")
    with open(prefill_path, "w") as f:
        f.write(prefill_script)
    os.chmod(prefill_path, 0o755)
    files["prefill"] = prefill_path

    # decode command
    decode_script = f"""#!/bin/bash
# Decode Node
export MOONCAKE_CONFIG_PATH={mooncake_path}

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \\
    --model {config.model} \\
    --port {config.decode_port} \\
    --tensor-parallel-size {config.tensor_parallel_size} \\
    --max-model-len {config.max_model_len} \\
    --gpu-memory-utilization {config.gpu_memory_utilization} \\
    --kv-transfer-config '{{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}}'
"""
    decode_path = os.path.join(output_dir, "start_decode.sh")
    with open(decode_path, "w") as f:
        f.write(decode_script)
    os.chmod(decode_path, 0o755)
    files["decode"] = decode_path

    # proxy server
    proxy_script = f"""#!/bin/bash
# Proxy Server - routes requests to prefill and decode
python3 {os.path.join(os.path.dirname(__file__), "..", "mooncake-wheel", "mooncake", "vllm_v1_proxy_server.py")} \\
    --prefiller-host {config.prefill_ip} --prefiller-port {config.prefill_port} \\
    --decoder-host {config.decode_ip} --decoder-port {config.decode_port} \\
    --port {config.proxy_port}
"""
    proxy_path = os.path.join(output_dir, "start_proxy.sh")
    with open(proxy_path, "w") as f:
        f.write(proxy_script)
    os.chmod(proxy_path, 0o755)
    files["proxy"] = proxy_path

    return files


def generate_test_request() -> Dict[str, Any]:
    """生成测试请求"""
    return {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "config")

    # 生成配置
    config = VLLMDisaggConfig(
        protocol="tcp",
        tensor_parallel_size=1,
    )
    files = generate_configs(output_dir, config)

    print("Generated configuration files:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    print(f"\nUsage:")
    print(f"  1. Start etcd: etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379")
    print(f"  2. Start Prefill: bash {files['prefill']}")
    print(f"  3. Start Decode:  bash {files['decode']}")
    print(f"  4. Start Proxy:   bash {files['proxy']}")
    print(f"  5. Test: curl http://127.0.0.1:{PROXY_PORT}/v1/chat/completions ...")

    # 生成测试请求
    test_req = generate_test_request()
    test_path = os.path.join(output_dir, "test_request.json")
    with open(test_path, "w") as f:
        json.dump(test_req, f, indent=2, ensure_ascii=False)
    print(f"\nTest request: {test_path}")
    print(f"  curl -X POST http://127.0.0.1:{PROXY_PORT}/v1/chat/completions \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d @{test_path}")


if __name__ == "__main__":
    main()
