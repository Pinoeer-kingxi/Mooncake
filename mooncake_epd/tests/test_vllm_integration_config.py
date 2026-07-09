from __future__ import annotations

from mooncake_epd.demo.vllm_integration import CONNECTOR_MODULE_PATH, VLLMDisaggConfig, generate_configs


def test_generate_configs_enables_repo_local_connector(tmp_path):
    config = VLLMDisaggConfig(local_hostname="127.0.0.1", layers_per_group=6, group_delay_ms=1.25)
    files = generate_configs(str(tmp_path), config)

    prefill_script = (tmp_path / "start_prefill.sh").read_text(encoding="utf-8")
    decode_script = (tmp_path / "start_decode.sh").read_text(encoding="utf-8")
    proxy_script = (tmp_path / "start_proxy.sh").read_text(encoding="utf-8")

    assert f'"kv_connector_module_path":"{CONNECTOR_MODULE_PATH}"' in prefill_script
    assert f'"kv_connector_module_path":"{CONNECTOR_MODULE_PATH}"' in decode_script
    assert '"connector_metrics_dir":"' in prefill_script
    assert '"connector_metrics_dir":"' in decode_script
    assert 'export PYTHONPATH=' in prefill_script
    assert 'export MOONCAKE_EPD_CONNECTOR_METRICS_DIR=' in prefill_script
    assert 'export MOONCAKE_EPD_CONNECTOR_METRICS_DIR=' in decode_script
    assert 'export MOONCAKE_EPD_CONNECTOR_METRICS_DIR=' in proxy_script
    assert 'export VLLM_MOONCAKE_BOOTSTRAP_PORT=' in prefill_script
    assert 'export VLLM_MOONCAKE_BOOTSTRAP_PORT=' in decode_script
    assert 'export VLLM_MOONCAKE_BOOTSTRAP_PORT=' not in proxy_script
    assert prefill_script.split('export VLLM_MOONCAKE_BOOTSTRAP_PORT=', 1)[1].splitlines()[0] != decode_script.split('export VLLM_MOONCAKE_BOOTSTRAP_PORT=', 1)[1].splitlines()[0]
    assert '--layers-per-group 6' in proxy_script
    assert '--group-delay-ms 1.25' in proxy_script
    assert '--workflow-registry-wal ' in proxy_script
    assert '--connector-metrics-dir ' in proxy_script
    assert '--enable-agent-state-clone' in proxy_script
    assert "proxy_workflow_registry" in files
    assert "connector_metrics_dir" in files
    assert files["prefill"].endswith("start_prefill.sh")


def test_generate_configs_can_enable_feature_handle_proxy_mode(tmp_path):
    config = VLLMDisaggConfig(
        local_hostname="127.0.0.1",
        mm_prefetch_mode="feature_handle",
        prefill_supports_feature_handles=True,
    )
    generate_configs(str(tmp_path), config)
    proxy_script = (tmp_path / "start_proxy.sh").read_text(encoding="utf-8")

    assert "--mm-prefetch-mode feature_handle" in proxy_script
    assert "--prefill-supports-feature-handles" in proxy_script


def test_generate_configs_enables_prefill_direct_feature_buffer_routes(tmp_path):
    config = VLLMDisaggConfig(
        local_hostname="127.0.0.1",
        mm_prefetch_mode="feature_handle",
        prefill_supports_feature_handles=True,
        enable_prefill_direct_feature_buffer_routes=True,
        encoder_service_url="http://127.0.0.1:8330",
    )
    generate_configs(str(tmp_path), config)
    prefill_script = (tmp_path / "start_prefill.sh").read_text(encoding="utf-8")
    proxy_script = (tmp_path / "start_proxy.sh").read_text(encoding="utf-8")

    assert "export MOONCAKE_EPD_ENABLE_DIRECT_FEATURE_BUFFER=1" in prefill_script
    assert "export MOONCAKE_EPD_DIRECT_BUFFER_WORKER_ID=prefill-0" in prefill_script
    assert "export MOONCAKE_EPD_FEATURE_HANDLE_WORKER_ID=prefill-0" in prefill_script
    assert "export MOONCAKE_EPD_DIRECT_BUFFER_DEVICE=cuda" in prefill_script
    assert "--encoder-service-url http://127.0.0.1:8330" in proxy_script
    assert f"--prefill-direct-buffer-service-url {config.prefill_direct_buffer_service_url}" in proxy_script
    assert "--release-direct-feature-buffers-after-prefill" in proxy_script


def test_generate_configs_can_enforce_strict_no_fallback(tmp_path):
    config = VLLMDisaggConfig(
        local_hostname="127.0.0.1",
        mm_prefetch_mode="feature_handle",
        prefill_supports_feature_handles=True,
        enable_prefill_direct_feature_buffer_routes=True,
        strict_no_fallback=True,
    )
    generate_configs(str(tmp_path), config)
    prefill_script = (tmp_path / "start_prefill.sh").read_text(encoding="utf-8")
    decode_script = (tmp_path / "start_decode.sh").read_text(encoding="utf-8")
    proxy_script = (tmp_path / "start_proxy.sh").read_text(encoding="utf-8")

    for script in (prefill_script, decode_script, proxy_script):
        assert "export MOONCAKE_EPD_STRICT=1" in script
        assert "export MOONCAKE_EPD_VLLM_FEATURE_HANDLE_STRICT=1" in script
        assert "export MOONCAKE_EPD_ALLOW_TRANSFER_FALLBACK=0" in script
    assert "--strict-no-fallback" in proxy_script
