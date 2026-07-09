"""Generate runnable vLLM + MooncakeConnector configs for this repo.

Targets the local real-model environment:
- model: /home/songbinbin/Proj/Proj_LWX/Qwen3-VL-8B-Instruct
- prefill GPU: 3
- decode GPU: 4

The generated commands opt into the repo-local external MooncakeConnector
module so layered transfer scheduling and serving control-plane metadata are
available on the real vLLM serving path.
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_ROOT = REPO_ROOT.parent / "venv_mooncake"
MODEL_PATH = "/home/songbinbin/Proj/Proj_LWX/Qwen3-VL-8B-Instruct"
CONNECTOR_MODULE_PATH = "mooncake_epd.core.control.vllm_mooncake_connector"


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _pick_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    if preferred > 0 and not _port_in_use(preferred, host):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass
class VLLMDisaggConfig:
    model: str = MODEL_PATH
    prefill_port: int = 8100
    decode_port: int = 8200
    proxy_port: int = 8000
    metadata_port: int = 8090
    master_port: int = 50061
    master_metrics_port: int = 59003
    prefill_bootstrap_port: int = 0
    decode_bootstrap_port: int = 0
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.65
    protocol: str = "tcp"
    prefill_gpu: int = 3
    decode_gpu: int = 4
    prefill_gpus: Tuple[int, ...] = ()
    decode_gpus: Tuple[int, ...] = ()
    prefill_ports: Tuple[int, ...] = ()
    decode_ports: Tuple[int, ...] = ()
    prefill_bootstrap_ports: Tuple[int, ...] = ()
    decode_bootstrap_ports: Tuple[int, ...] = ()
    local_hostname: str = "127.0.0.1"
    global_segment_size: int = 1073741824
    local_buffer_size: int = 268435456
    layers_per_group: int = 4
    group_delay_ms: float = 0.0
    max_group_bytes: int = 16 * 1024 * 1024
    max_transfer_descriptors: int = 32
    max_transfer_bytes: int = 16 * 1024 * 1024
    allow_transfer_fallback: bool = False
    transfer_retry_attempts: int = 6
    transfer_retry_backoff_ms: float = 250.0
    proxy_warn_rho: float = 0.85
    proxy_critical_rho: float = 0.95
    proxy_max_backpressure_delay_ms: float = 150.0
    owner_shards: int = 1
    kv_directory_rpc_url: Optional[str] = None
    workflow_registry_wal_path: Optional[str] = None
    connector_metrics_dir: Optional[str] = None
    mm_prefetch_mode: str = "asset_bytes"
    prefill_supports_feature_handles: bool = False
    encoder_service_url: Optional[str] = None
    prefill_direct_buffer_service_url: Optional[str] = None
    enable_prefill_direct_feature_buffer_routes: bool = False
    direct_feature_buffer_root_routes: bool = True
    release_direct_feature_buffers_after_prefill: bool = True
    strict_no_fallback: bool = False

    @property
    def metadata_server(self) -> str:
        return f"http://{self.local_hostname}:{self.metadata_port}/metadata"

    @property
    def master_server(self) -> str:
        return f"{self.local_hostname}:{self.master_port}"

    def to_mooncake_json(self) -> Dict[str, object]:
        return {
            "local_hostname": self.local_hostname,
            "metadata_server": self.metadata_server,
            "global_segment_size": self.global_segment_size,
            "local_buffer_size": self.local_buffer_size,
            "protocol": self.protocol,
            "device_name": "",
            "master_server_address": self.master_server,
        }

    def kv_transfer_config(self, role: str, engine_id: str) -> Dict[str, object]:
        extra_config: Dict[str, object] = {
            "mooncake_protocol": self.protocol,
            "num_workers": 4,
            "layered_kv_transfer": True,
            "layers_per_group": self.layers_per_group,
            "group_delay_ms": self.group_delay_ms,
            "max_group_bytes": self.max_group_bytes,
            "max_transfer_descriptors": self.max_transfer_descriptors,
            "max_transfer_bytes": self.max_transfer_bytes,
            "allow_transfer_fallback": self.allow_transfer_fallback,
            "transfer_retry_attempts": self.transfer_retry_attempts,
            "transfer_retry_backoff_ms": self.transfer_retry_backoff_ms,
            "transport_backend": "mooncake_engine_direct",
        }
        if self.connector_metrics_dir:
            extra_config["connector_metrics_dir"] = self.connector_metrics_dir
        return {
            "kv_connector": "MooncakeConnector",
            "kv_role": role,
            "engine_id": engine_id,
            "kv_connector_module_path": CONNECTOR_MODULE_PATH,
            "kv_connector_extra_config": extra_config,
        }


def _expand_ints(primary: int, values: Tuple[int, ...], count: int, *, fill: int = 0) -> list[int]:
    if values:
        out = [int(v) for v in values]
    else:
        out = [int(primary)]
    while len(out) < count:
        out.append(int(fill))
    return out[:count]


def validate_environment(config: Optional[VLLMDisaggConfig] = None) -> Dict[str, object]:
    config = config or VLLMDisaggConfig()
    checks = {
        "model_exists": Path(config.model).exists(),
        "venv_exists": VENV_ROOT.exists(),
        "vllm_bin": str(VENV_ROOT / "bin" / "vllm"),
        "mooncake_master_bin": str(VENV_ROOT / "bin" / "mooncake_master"),
        "python_bin": str(VENV_ROOT / "bin" / "python"),
        "proxy_script": str(REPO_ROOT / "scripts" / "vllm_disagg_proxy.py"),
        "connector_module": CONNECTOR_MODULE_PATH,
    }
    checks["vllm_bin_exists"] = Path(checks["vllm_bin"]).exists()
    checks["mooncake_master_exists"] = Path(checks["mooncake_master_bin"]).exists()
    checks["python_bin_exists"] = Path(checks["python_bin"]).exists()
    checks["proxy_script_exists"] = Path(checks["proxy_script"]).exists()
    return checks


def _common_env_block(
    config: VLLMDisaggConfig,
    mooncake_json: Path,
    *,
    bootstrap_port: Optional[int] = None,
) -> str:
    parent_path = str(REPO_ROOT.parent)
    lines = [
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY",
        "export NO_PROXY=127.0.0.1,localhost",
        f"export PYTHONPATH={parent_path}:${{PYTHONPATH:-}}",
        f"source {VENV_ROOT}/bin/activate",
        f"export MOONCAKE_CONFIG_PATH={mooncake_json}",
        f"export MOONCAKE_MASTER={config.master_server}",
        f"export MOONCAKE_TE_META_DATA_SERVER={config.metadata_server}",
        f"export MOONCAKE_PROTOCOL={config.protocol}",
        f"export MOONCAKE_LOCAL_HOSTNAME={config.local_hostname}",
        f"export VLLM_HOST_IP={config.local_hostname}",
        f"export MOONCAKE_GLOBAL_SEGMENT_SIZE={config.global_segment_size}",
        f"export MOONCAKE_LOCAL_BUFFER_SIZE={config.local_buffer_size}",
        "export OPENAI_API_KEY=sk-local",
        "export MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE=${MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE:-1}",
        "export MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE_MAX_ENTRIES=${MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE_MAX_ENTRIES:-64}",
        "export MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE_MAX_BYTES=${MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE_MAX_BYTES:-2147483648}",
    ]
    if config.strict_no_fallback:
        lines.extend(
            [
                "export MOONCAKE_EPD_STRICT=1",
                "export MOONCAKE_EPD_VLLM_FEATURE_HANDLE_STRICT=1",
                "export MOONCAKE_EPD_ALLOW_TRANSFER_FALLBACK=0",
            ]
        )
    if bootstrap_port is not None:
        lines.append(f"export VLLM_MOONCAKE_BOOTSTRAP_PORT={bootstrap_port}")
    if config.connector_metrics_dir:
        lines.append(
            f"export MOONCAKE_EPD_CONNECTOR_METRICS_DIR={config.connector_metrics_dir}"
        )
    return "\n".join(lines)


def _json_flag(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_configs(output_dir: str, config: Optional[VLLMDisaggConfig] = None) -> Dict[str, object]:
    config = config or VLLMDisaggConfig()
    prefill_gpus = list(config.prefill_gpus or (config.prefill_gpu,))
    decode_gpus = list(config.decode_gpus or (config.decode_gpu,))
    if not prefill_gpus:
        raise ValueError("at least one prefill GPU is required")
    if not decode_gpus:
        raise ValueError("at least one decode GPU is required")
    prefill_ports = _expand_ints(config.prefill_port, config.prefill_ports, len(prefill_gpus))
    decode_ports = _expand_ints(config.decode_port, config.decode_ports, len(decode_gpus))
    prefill_bootstrap_ports = _expand_ints(
        config.prefill_bootstrap_port,
        config.prefill_bootstrap_ports,
        len(prefill_gpus),
    )
    decode_bootstrap_ports = _expand_ints(
        config.decode_bootstrap_port,
        config.decode_bootstrap_ports,
        len(decode_gpus),
    )
    prefill_ports = [_pick_free_port(port, config.local_hostname) for port in prefill_ports]
    decode_ports = [_pick_free_port(port, config.local_hostname) for port in decode_ports]
    prefill_bootstrap_ports = [
        _pick_free_port(port, config.local_hostname) for port in prefill_bootstrap_ports
    ]
    decode_bootstrap_ports = [
        _pick_free_port(port, config.local_hostname) for port in decode_bootstrap_ports
    ]
    used_bootstrap = set()
    for idx, port in enumerate(prefill_bootstrap_ports):
        while port in used_bootstrap:
            port = _pick_free_port(0, config.local_hostname)
        prefill_bootstrap_ports[idx] = port
        used_bootstrap.add(port)
    for idx, port in enumerate(decode_bootstrap_ports):
        while port in used_bootstrap:
            port = _pick_free_port(0, config.local_hostname)
        decode_bootstrap_ports[idx] = port
        used_bootstrap.add(port)
    config.prefill_port = prefill_ports[0]
    config.decode_port = decode_ports[0]
    config.prefill_bootstrap_port = prefill_bootstrap_ports[0]
    config.decode_bootstrap_port = decode_bootstrap_ports[0]
    config.proxy_port = _pick_free_port(config.proxy_port, config.local_hostname)
    config.metadata_port = _pick_free_port(config.metadata_port, config.local_hostname)
    config.master_port = _pick_free_port(config.master_port, config.local_hostname)
    config.master_metrics_port = _pick_free_port(config.master_metrics_port, config.local_hostname)
    if (
        config.enable_prefill_direct_feature_buffer_routes
        and not config.prefill_direct_buffer_service_url
        and len(prefill_ports) == 1
    ):
        config.prefill_direct_buffer_service_url = f"http://{config.local_hostname}:{prefill_ports[0]}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, object] = {}
    config.workflow_registry_wal_path = (
        config.workflow_registry_wal_path
        or str(out_dir / "proxy_workflow_registry.jsonl")
    )
    config.connector_metrics_dir = (
        config.connector_metrics_dir
        or str(out_dir / "connector_metrics")
    )
    mooncake_path = out_dir / "mooncake.json"
    mooncake_path.write_text(json.dumps(config.to_mooncake_json(), indent=2), encoding="utf-8")
    files["mooncake_json"] = str(mooncake_path)

    metadata_script = out_dir / "start_metadata.sh"
    metadata_script.write_text(
        "#!/bin/bash\n"
        + _common_env_block(config, mooncake_path)
        + "\n"
        + f"python -m mooncake.http_metadata_server --host {config.local_hostname} --port {config.metadata_port}\n",
        encoding="utf-8",
    )
    metadata_script.chmod(0o755)
    files["metadata"] = str(metadata_script)

    master_script = out_dir / "start_master.sh"
    master_script.write_text(
        "#!/bin/bash\n"
        + _common_env_block(config, mooncake_path)
        + "\n"
        + f"mooncake_master --rpc_port={config.master_port} --metrics_port={config.master_metrics_port}\n",
        encoding="utf-8",
    )
    master_script.chmod(0o755)
    files["master"] = str(master_script)

    prefill_scripts: list[str] = []
    for idx, (gpu, port, bootstrap_port) in enumerate(
        zip(prefill_gpus, prefill_ports, prefill_bootstrap_ports)
    ):
        engine_id = f"epd-prefill-{idx}" if len(prefill_gpus) > 1 else "epd-prefill"
        prefill_kv_cfg = _json_flag(config.kv_transfer_config("kv_producer", engine_id))
        prefill_script = out_dir / ("start_prefill.sh" if idx == 0 else f"start_prefill_{idx}.sh")
        prefill_script.write_text(
            "#!/bin/bash\n"
            + _common_env_block(
                config,
                mooncake_path,
                bootstrap_port=bootstrap_port,
            )
            + "\n"
            + f"export MOONCAKE_EPD_ENGINE_ID={engine_id}\n"
            + "export MOONCAKE_EPD_KV_ROLE=kv_producer\n"
            + (
                "export MOONCAKE_EPD_ENABLE_DIRECT_FEATURE_BUFFER=1\n"
                f"export MOONCAKE_EPD_DIRECT_BUFFER_WORKER_ID=prefill-{idx}\n"
                f"export MOONCAKE_EPD_FEATURE_HANDLE_WORKER_ID=prefill-{idx}\n"
                "export MOONCAKE_EPD_DIRECT_BUFFER_DEVICE=cuda\n"
                f"export MOONCAKE_EPD_DIRECT_LOCAL_HOSTNAME={config.local_hostname}:{18000 + idx}\n"
                "export MOONCAKE_EPD_DIRECT_TARGET_MODE=managed_buffer\n"
                "export MOONCAKE_EPD_DIRECT_REGISTER_MEMORY=0\n"
                f"export MOONCAKE_EPD_DIRECT_BUFFER_ROOT_ROUTES={1 if config.direct_feature_buffer_root_routes else 0}\n"
                if config.enable_prefill_direct_feature_buffer_routes
                else ""
            )
            + f"CUDA_VISIBLE_DEVICES={gpu} vllm serve {config.model} "
            + f"--port {port} "
            + f"--tensor-parallel-size {config.tensor_parallel_size} "
            + f"--max-model-len {config.max_model_len} "
            + f"--gpu-memory-utilization {config.gpu_memory_utilization} "
            + "--kv-transfer-config "
            + f"'{prefill_kv_cfg}'\n",
            encoding="utf-8",
        )
        prefill_script.chmod(0o755)
        prefill_scripts.append(str(prefill_script))
    files["prefill"] = prefill_scripts[0]
    files["prefill_scripts"] = prefill_scripts
    files["prefill_ports"] = prefill_ports
    files["prefill_gpus"] = prefill_gpus

    decode_scripts: list[str] = []
    for idx, (gpu, port, bootstrap_port) in enumerate(
        zip(decode_gpus, decode_ports, decode_bootstrap_ports)
    ):
        engine_id = f"epd-decode-{idx}" if len(decode_gpus) > 1 else "epd-decode"
        decode_kv_cfg = _json_flag(config.kv_transfer_config("kv_consumer", engine_id))
        decode_script = out_dir / ("start_decode.sh" if idx == 0 else f"start_decode_{idx}.sh")
        decode_script.write_text(
            "#!/bin/bash\n"
            + _common_env_block(
                config,
                mooncake_path,
                bootstrap_port=bootstrap_port,
            )
            + "\n"
            + f"export MOONCAKE_EPD_ENGINE_ID={engine_id}\n"
            + "export MOONCAKE_EPD_KV_ROLE=kv_consumer\n"
            + f"CUDA_VISIBLE_DEVICES={gpu} vllm serve {config.model} "
            + f"--port {port} "
            + f"--tensor-parallel-size {config.tensor_parallel_size} "
            + f"--max-model-len {config.max_model_len} "
            + f"--gpu-memory-utilization {config.gpu_memory_utilization} "
            + "--kv-transfer-config "
            + f"'{decode_kv_cfg}'\n",
            encoding="utf-8",
        )
        decode_script.chmod(0o755)
        decode_scripts.append(str(decode_script))
    files["decode"] = decode_scripts[0]
    files["decode_scripts"] = decode_scripts
    files["decode_ports"] = decode_ports
    files["decode_gpus"] = decode_gpus

    proxy_script = out_dir / "start_proxy.sh"
    prefill_hosts_flag = " ".join([config.local_hostname for _ in prefill_ports])
    prefill_ports_flag = " ".join(str(port) for port in prefill_ports)
    decode_hosts_flag = " ".join([config.local_hostname for _ in decode_ports])
    decode_ports_flag = " ".join(str(port) for port in decode_ports)
    high_prefill_ids = " ".join(["prefill-0"]) if prefill_ports else ""
    standard_prefill_ids = " ".join(f"prefill-{idx}" for idx in range(1, len(prefill_ports)))
    low_decode_ids = " ".join(["decode-0"]) if decode_ports else ""
    standard_decode_ids = " ".join(f"decode-{idx}" for idx in range(1, len(decode_ports)))
    proxy_script.write_text(
        "#!/bin/bash\n"
        + _common_env_block(config, mooncake_path)
        + "\n"
        + f"python {REPO_ROOT / 'scripts' / 'vllm_disagg_proxy.py'} "
        + f"--prefiller-hosts {prefill_hosts_flag} --prefiller-ports {prefill_ports_flag} "
        + f"--decoder-hosts {decode_hosts_flag} --decoder-ports {decode_ports_flag} "
        + f"--layers-per-group {config.layers_per_group} "
        + f"--group-delay-ms {config.group_delay_ms} "
        + f"--max-group-bytes {config.max_group_bytes} "
        + f"--warn-rho {config.proxy_warn_rho} "
        + f"--critical-rho {config.proxy_critical_rho} "
        + f"--max-backpressure-delay-ms {config.proxy_max_backpressure_delay_ms} "
        + f"--transport-backend mooncake_engine_direct "
        + f"--mm-prefetch-mode {config.mm_prefetch_mode} "
        + ("--prefill-supports-feature-handles " if config.prefill_supports_feature_handles else "")
        + f"--owner-shards {config.owner_shards} "
        + (
            f"--kv-directory-rpc-url {config.kv_directory_rpc_url} "
            if config.kv_directory_rpc_url
            else ""
        )
        + f"--connector-metrics-dir {config.connector_metrics_dir} "
        + f"--workflow-registry-wal {config.workflow_registry_wal_path} "
        + (f"--high-prefill-worker-ids {high_prefill_ids} " if high_prefill_ids else "")
        + (f"--standard-prefill-worker-ids {standard_prefill_ids} " if standard_prefill_ids else "")
        + (f"--low-latency-decode-worker-ids {low_decode_ids} " if low_decode_ids else "")
        + (f"--standard-decode-worker-ids {standard_decode_ids} " if standard_decode_ids else "")
        + (f"--encoder-service-url {config.encoder_service_url} " if config.encoder_service_url else "")
        + (
            f"--prefill-direct-buffer-service-url {config.prefill_direct_buffer_service_url} "
            if config.prefill_direct_buffer_service_url
            else ""
        )
        + (
            "--release-direct-feature-buffers-after-prefill "
            if config.release_direct_feature_buffers_after_prefill
            else "--no-release-direct-feature-buffers-after-prefill "
        )
        + "--enable-agent-state-clone "
        + ("--strict-no-fallback " if config.strict_no_fallback else "--no-strict-no-fallback ")
        + f"--port {config.proxy_port}\n",
        encoding="utf-8",
    )
    proxy_script.chmod(0o755)
    files["proxy"] = str(proxy_script)
    files["proxy_workflow_registry"] = config.workflow_registry_wal_path
    files["connector_metrics_dir"] = config.connector_metrics_dir

    test_req = out_dir / "test_request.json"
    test_req.write_text(
        json.dumps(
            {
                "model": config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Please introduce Mooncake PD disaggregation briefly."}],
                    }
                ],
                "max_tokens": 64,
                "temperature": 0.0,
                "metadata": {"workflow_id": "demo-text-workflow"},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files["test_request"] = str(test_req)
    return files


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = VLLMDisaggConfig()
    checks = validate_environment(config)
    files = generate_configs(str(REPO_ROOT / "config"), config)

    print("Environment checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    print("\nGenerated files:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    print("\nStartup order:")
    print(f"  1. bash {files['metadata']}")
    print(f"  2. bash {files['master']}")
    print(f"  3. bash {files['prefill']}")
    print(f"  4. bash {files['decode']}")
    print(f"  5. bash {files['proxy']}")
    print(
        "  6. python "
        f"{REPO_ROOT / 'scripts' / 'check_vllm_disagg.py'} "
        f"--prefill-url http://{config.local_hostname}:{config.prefill_port} "
        f"--decode-url http://{config.local_hostname}:{config.decode_port} "
        f"--proxy-url http://{config.local_hostname}:{config.proxy_port} "
        f"--request {files['test_request']}"
    )


if __name__ == "__main__":
    main()
