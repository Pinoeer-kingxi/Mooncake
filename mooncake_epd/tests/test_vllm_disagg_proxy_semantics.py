from __future__ import annotations

import json

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from mooncake_epd.core.control import ServingControlPlane, ServingControlPlaneConfig
from mooncake_epd.core.control.connector_metrics import ConnectorMetricsSink
from mooncake_epd.core.control.vllm_transfer_primitives import LayeredTransferWorkerMeta
from mooncake_epd.core.state import WorkflowStateRegistry
from mooncake_epd.scripts.vllm_disagg_proxy import ProxyConfig, create_app



def _build_prefill_app(record: dict) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions/render")
    async def render_chat(request: Request):
        body = await request.json()
        record["prefill_render_body"] = body
        return JSONResponse(
            {
                "request_id": "rendered-prefill-0",
                "token_ids": [1, 2, 3, 4],
                "sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 16,
                    "min_tokens": 0,
                },
                "model": "fake-model",
                "stream": bool(body.get("stream")),
                "priority": 0,
            }
        )

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()
        record["prefill_generate_body"] = body
        sampling_params = dict(body.get("sampling_params") or {})
        extra_args = dict(sampling_params.get("extra_args") or {})
        kv = dict(extra_args.get("kv_transfer_params") or {})
        kv.update(
            {
                "transfer_id": kv.get("transfer_id") or request.headers.get("X-Request-Id"),
                "remote_engine_id": "prefill-engine-0",
                "remote_bootstrap_addr": "http://prefill-bootstrap:8998",
                "remote_block_ids": [[11, 12, 13]],
            }
        )
        return JSONResponse(
            {
                "request_id": "prefill-response",
                "choices": [{"index": 0, "finish_reason": "length", "token_ids": []}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                },
                "kv_transfer_params": kv,
            }
        )

    return app


def _build_decode_app(record: dict) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        record["decode_body"] = body
        if body.get("stream"):
            async def _gen():
                yield (
                    b'data: {"id":"chunk-0","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                )
                yield (
                    b'data: {"id":"chunk-1","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
                )
                yield (
                    b'data: {"id":"chunk-2","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}],'
                    b'"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n\n'
                )
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "decode-response",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "helloworld"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            }
        )

    return app


def _build_empty_decode_app(record: dict) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        record["decode_body"] = body

        async def _gen():
            if False:
                yield b""

        return StreamingResponse(_gen(), media_type="text/event-stream")

    return app


def _client_override(app: FastAPI, worker_id: str, host: str, port: int) -> dict:
    transport = httpx.ASGITransport(app=app)
    return {
        "client": httpx.AsyncClient(base_url=f"http://{host}:{port}", transport=transport, timeout=None),
        "host": host,
        "port": port,
        "id": 0,
        "worker_id": worker_id,
    }



def test_proxy_propagates_control_plane_metadata_and_metrics(tmp_path):
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    sink = ConnectorMetricsSink(
        tmp_path,
        engine_id="prefill-engine-0",
        role="producer",
        hostname="prefill.local",
        rpc_port=8998,
        tp_rank=0,
    )
    sink.record(
        LayeredTransferWorkerMeta(
            grouped_batches=1,
            grouped_bytes=96,
            grouped_descriptors=3,
            peer_buffer_batches=1,
            peer_buffer_bytes=96,
            backend_counts={"peer_buffer_direct": 1},
        ),
        path_totals={
            "EPD": LayeredTransferWorkerMeta(
                grouped_batches=1,
                grouped_bytes=96,
                grouped_descriptors=3,
            )
        },
    )
    cp = ServingControlPlane(
        ServingControlPlaneConfig(
            node_id="proxy-it",
            layers_per_group=6,
            group_delay_ms=2.5,
            connector_metrics_dir=str(tmp_path),
            enable_agent_state_clone=True,
        )
    )
    proxy_app = create_app(
        ProxyConfig(
            layers_per_group=6,
            group_delay_ms=2.5,
            transport_backend="mooncake_engine_direct",
            connector_metrics_dir=str(tmp_path),
        ),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    {"type": "text", "text": "What is shown here?"},
                ],
            }
        ],
        "max_tokens": 16,
        "stream": True,
        "metadata": {"workflow_id": "wf-proxy-mm"},
    }

    with TestClient(proxy_app) as client:
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.headers["x-epd-routing-path"] == "EPD"
        assert response.headers["x-epd-admission"] in {"ADMIT", "BACKPRESSURE"}
        assert b"data:" in response.content

        prefill_kv = record["prefill_generate_body"]["sampling_params"]["extra_args"]["kv_transfer_params"]
        decode_kv = record["decode_body"]["kv_transfer_params"]
        assert record["prefill_render_body"]["messages"] == body["messages"]
        assert record["prefill_generate_body"]["sampling_params"]["max_tokens"] == 0
        assert record["prefill_generate_body"]["sampling_params"]["min_tokens"] == 0
        assert prefill_kv["transfer_id"] == decode_kv["transfer_id"]
        assert prefill_kv["layered_kv_transfer"] is True
        assert prefill_kv["layers_per_group"] == 6
        assert prefill_kv["mm_prefetch_policy"] == "event_driven"
        assert decode_kv["do_remote_prefill"] is True
        assert decode_kv["do_remote_decode"] is False
        assert decode_kv["handoff_id"]
        assert decode_kv["workflow_id"] == "wf-proxy-mm"
        assert decode_kv["transport_backend"] == "mooncake_engine_direct"
        assert decode_kv["a2a_source_node"] == "prefill-0"
        assert decode_kv["a2a_target_node"] == "decode-0"

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        payload = metrics.json()
        assert payload["metrics"]["handoff_prepared"] == 1
        assert payload["metrics"]["handoff_committed"] == 1
        assert payload["metrics"]["requests_multimodal"] == 1
        assert payload["metrics"]["peer_buffer_batches"] == 1
        assert payload["metrics"]["path_stats"]["EPD"]["requests_total"] == 1
        assert payload["metrics"]["path_stats"]["EPD"]["handoff_committed"] == 1
        assert payload["metrics"]["path_stats"]["EPD"]["stage_dispatches"] == {
            "prefill": 1,
            "decode": 1,
        }
        assert payload["metrics"]["connector_path_stats"]["EPD"]["grouped_bytes"] == 96
        assert payload["metrics"]["remote_transfer_backend_counts"] == {"peer_buffer_direct": 1}

        fork = client.post(
            "/mooncake_epd/agent/fork",
            json={
                "workflow_id": "wf-proxy-mm",
                "parent_request_id": response.headers["x-request-id"],
                "branch_count": 2,
                "target_node_id": "decode-0",
            },
        )
        assert fork.status_code == 200
        fork_payload = fork.json()
        assert fork_payload["zero_copy_branches"] == 2
        assert fork_payload["copied_bytes"] == 0
        assert fork_payload["kv_block_ids"] == [
            "prefill-engine-0:11",
            "prefill-engine-0:12",
            "prefill-engine-0:13",
        ]

        metrics_after_fork = client.get("/metrics").json()
        assert metrics_after_fork["metrics"]["agent_state_clone_requests"] == 1
        assert metrics_after_fork["metrics"]["agent_state_clone_branches"] == 2



def test_proxy_rejects_when_decode_stage_is_exhausted():
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-it-2"))
    proxy_app = create_app(
        ProxyConfig(),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )

    with TestClient(proxy_app) as client:
        cp.update_worker_load(
            "decode",
            "decode-0",
            current_load=64,
            max_capacity=64,
            queue_size=64,
            queue_capacity=64,
            service_rate=10.0,
            arrival_rate=20.0,
        )
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        )
        assert response.status_code == 503
        assert "decode" in response.text or "rejected" in response.text


def test_proxy_rolls_back_handoff_when_decode_stream_never_yields():
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_empty_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-empty-stream"))
    proxy_app = create_app(
        ProxyConfig(),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
                "stream": True,
                "metadata": {"workflow_id": "wf-empty-stream"},
            },
        )
        assert response.status_code == 200
        assert response.content == b""

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        payload = metrics.json()
        assert payload["metrics"]["handoff_prepared"] == 1
        assert payload["metrics"]["handoff_committed"] == 0
        assert payload["metrics"]["handoff_rolled_back"] == 1
        assert payload["metrics"]["path_stats"]["PD"]["requests_total"] == 1
        assert payload["metrics"]["path_stats"]["PD"]["handoff_rolled_back"] == 1


def test_proxy_updates_workflow_registry_across_request_lifecycle(tmp_path):
    record: dict = {}
    registry = WorkflowStateRegistry(str(tmp_path / "proxy-registry.jsonl"))
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(
        ServingControlPlaneConfig(
            node_id="proxy-registry-it",
            workflow_registry_wal_path=str(tmp_path / "serving-registry-shadow.jsonl"),
        ),
        workflow_registry=registry,
    )
    proxy_app = create_app(
        ProxyConfig(
            node_id="proxy-registry-it",
            workflow_registry_wal_path=str(tmp_path / "serving-registry-shadow.jsonl"),
        ),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello registry"}],
                    }
                ],
                "stream": True,
                "metadata": {"workflow_id": "wf-proxy-registry"},
            },
        )
        assert response.status_code == 200
        request_id = response.headers["x-request-id"]

        registry_record = registry.get_record(request_id)
        assert registry_record is not None
        assert registry_record.workflow_id == "wf-proxy-registry"
        assert registry_record.status == "RELEASED"
        assert registry_record.agent_id == "decode-0"
        assert registry_record.released_at is not None

        metrics = client.get("/metrics").json()
        reg_snapshot = metrics["workflow_registry"]
        assert reg_snapshot["enabled"] is True
        assert request_id not in reg_snapshot["active_state_ids"]
        assert reg_snapshot["status_counts"]["RELEASED"] >= 1


def test_proxy_uses_prompt_only_prefill_and_keeps_decode_payload_unpatched():
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-prompt-only"))
    proxy_app = create_app(
        ProxyConfig(),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is the answer?"}],
                    }
                ],
                "max_tokens": 18,
                "stream": True,
                "metadata": {"workflow_id": "wf-prompt-only"},
            },
        )
        assert response.status_code == 200
        assert record["prefill_generate_body"]["request_id"] == response.headers["x-request-id"]
        assert record["prefill_generate_body"]["sampling_params"]["max_tokens"] == 0
        assert record["prefill_generate_body"]["sampling_params"]["min_tokens"] == 0
        assert "continue_final_message" not in record["decode_body"]
        assert "add_generation_prompt" not in record["decode_body"]
        assert record["decode_body"]["max_tokens"] == 18
        assert record["decode_body"]["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the answer?"}],
            }
        ]
        text = response.text
        chunks = []
        for line in text.splitlines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            packet = json.loads(line[6:])
            choices = list(packet.get("choices") or [])
            if not choices:
                continue
            delta = dict(choices[0].get("delta") or {})
            content = delta.get("content")
            if content:
                chunks.append(str(content))
        assert "".join(chunks) == "helloworld"
        usage_line = next(
            line for line in text.splitlines() if '"usage"' in line and '"completion_tokens"' in line
        )
        assert '"prompt_tokens": 12' in usage_line
        assert '"completion_tokens": 2' in usage_line
        assert '"total_tokens": 14' in usage_line


def test_prompt_only_prefill_does_not_short_circuit_when_user_budget_is_one():
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-budget-one"))
    proxy_app = create_app(
        ProxyConfig(),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Say hi"}],
                    }
                ],
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert record["decode_body"]["max_tokens"] == 1
        payload = response.json()
        assert payload["id"] == "decode-response"


def test_proxy_mm_store_prefetches_data_url_on_serving_hot_path(tmp_path):
    import base64

    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-mm-prefetch"))
    proxy_app = create_app(
        ProxyConfig(mm_prefetch_wait_ms=500.0),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )
    data_url = "data:image/png;base64," + base64.b64encode(b"not-a-real-png-but-real-bytes").decode("ascii")

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                "metadata": {"workflow_id": "wf-mm-prefetch"},
            },
        )
        assert response.status_code == 200
        metrics = client.get("/metrics").json()
        assert metrics["metrics"]["mm_prefetch_attempted"] == 1
        assert metrics["metrics"]["mm_prefetch_completed"] == 1
        assert metrics["metrics"]["mm_prefetch_failed"] == 0
        assert metrics["metrics"]["path_stats"]["EPD"]["mm_prefetch_completed"] == 1
        assert metrics["metrics"]["path_stats"]["EPD"]["mm_prefetch_wait_ms_count"] == 1
        assert metrics["metrics"]["path_stats"]["EPD"]["mm_prefetch_wait_ms_avg"] >= 0.0
        assert metrics["mm_store"]["completed"] >= 1
        rendered_url = record["prefill_render_body"]["messages"][0]["content"][0]["image_url"]["url"]
        decode_url = record["decode_body"]["messages"][0]["content"][0]["image_url"]["url"]
        assert rendered_url.startswith("data:image/png;base64,")
        assert decode_url == rendered_url


def _feature_handle_payload_for_item(item: dict, *, feature_id: str = "feature-hidden-0") -> dict:
    from mooncake_epd.core.control import ServingControlPlane
    from mooncake_epd.core.state import FeatureBundle, FeatureHandle

    bundle = FeatureBundle(
        image_hash=feature_id,
        last_hidden=__import__("torch").randn(2, 4),
        intermediates=[],
        metadata={"model_fingerprint": "model-x", "processor_fingerprint": "processor-x"},
    )
    source_mm_hash = ServingControlPlane._stable_mm_hash(item)
    handle = FeatureHandle(
        handle_id="handle-0",
        feature_id=feature_id,
        store_id="external-encoder-store",
        uri=f"mmstore://external-encoder-store/{feature_id}",
        descriptor=bundle.descriptor(checksum=False),
        metadata={"source_mm_hash": source_mm_hash},
    )
    return handle.as_control_payload()


def test_proxy_feature_handle_mode_fails_fast_without_prefill_support():
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-feature-handle-unsupported"))
    proxy_app = create_app(
        ProxyConfig(mm_prefetch_mode="feature_handle", prefill_supports_feature_handles=False),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )
    image_item = {"type": "image_url", "image_url": {"url": "https://example.com/hidden.png"}}

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": [image_item, {"type": "text", "text": "describe"}]}
                ],
                "metadata": {
                    "workflow_id": "wf-feature-handle-unsupported",
                    "mooncake_epd_feature_handles": [_feature_handle_payload_for_item(image_item)],
                },
            },
        )
        assert response.status_code == 501
        assert "external multimodal hidden-state handles" in response.json()["detail"]
        metrics = client.get("/metrics").json()
        assert metrics["metrics"]["path_stats"]["EPD"]["requests_active"] == 0
        assert "prefill_generate_body" not in record


def test_proxy_feature_handle_mode_forwards_handles_when_prefill_supports_them():
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    cp = ServingControlPlane(ServingControlPlaneConfig(node_id="proxy-feature-handle"))
    proxy_app = create_app(
        ProxyConfig(mm_prefetch_mode="feature_handle", prefill_supports_feature_handles=True),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
        control_plane=cp,
    )
    image_url = "https://example.com/hidden.png"
    image_item = {"type": "image_url", "image_url": {"url": image_url}}
    feature_handle = _feature_handle_payload_for_item(image_item)

    with TestClient(proxy_app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": [image_item, {"type": "text", "text": "describe"}]}
                ],
                "metadata": {
                    "workflow_id": "wf-feature-handle",
                    "mooncake_epd_feature_handles": [feature_handle],
                },
            },
        )
        assert response.status_code == 200
        prefill_kv = record["prefill_generate_body"]["sampling_params"]["extra_args"]["kv_transfer_params"]
        assert prefill_kv["mm_prefetch_policy"] == "feature_handle"
        assert prefill_kv["mm_feature_handles"][0]["handle_id"] == "handle-0"
        assert prefill_kv["mm_feature_handle_target_worker"] == "prefill-0"
        assert record["prefill_render_body"]["messages"][0]["content"][0]["image_url"]["url"] == image_url
        metrics = client.get("/metrics").json()
        assert metrics["metrics"]["path_stats"]["EPD"]["requests_active"] == 0
        assert metrics["metrics"]["path_stats"]["EPD"]["requests_total"] == 1
