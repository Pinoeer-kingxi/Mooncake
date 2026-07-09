"""Production-oriented vLLM disaggregated proxy with EPD control-plane hooks.

Compared with the upstream Mooncake proxy, this variant adds:

- stable ``transfer_id`` propagation on the first prefill leg;
- admission / backpressure before prefill and decode dispatch;
- A2A-style 2PC handoff bookkeeping around the P->D transition;
- request-level metadata injection for layered KV transfer, MM prefetch, and
  transport backend hints;
- health / metrics endpoints for operational inspection.

The downstream data plane is still real vLLM + Mooncake. This proxy only owns
routing, metadata propagation, and serving-time control semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from mooncake_epd.agent.coordination.scheduler import AdmissionAction  # noqa: E402
from mooncake_epd.core.control import ServingControlPlane, ServingControlPlaneConfig  # noqa: E402
from mooncake_epd.core.strict_mode import strict_no_fallback_enabled  # noqa: E402
from mooncake_epd.core.state import FeatureBundle, FeatureHandle, MMStore  # noqa: E402
from mooncake_epd.core.transfer import TransferEngine  # noqa: E402


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    prefiller_instances: List[tuple[str, int]] = field(default_factory=lambda: [("127.0.0.1", 8100)])
    decoder_instances: List[tuple[str, int]] = field(default_factory=lambda: [("127.0.0.1", 8200)])
    layers_per_group: int = 4
    group_delay_ms: float = 0.0
    max_group_bytes: int = 0
    warn_rho: float = 0.85
    critical_rho: float = 0.95
    max_backpressure_delay_ms: float = 150.0
    transport_backend: str = "mooncake_engine_direct"
    node_id: str = "proxy"
    owner_shards: int = 1
    kv_directory_rpc_url: Optional[str] = None
    connector_metrics_dir: Optional[str] = None
    workflow_registry_wal_path: Optional[str] = None
    enable_mm_prefetch: bool = True
    mm_prefetch_mode: str = "asset_bytes"
    prefill_supports_feature_handles: bool = False
    mm_prefetch_wait_ms: float = 100.0
    mm_prefetch_max_asset_bytes: int = 16 * 1024 * 1024
    mm_prefetch_queue_size: int = 256
    encoder_service_url: Optional[str] = None
    encoder_service_timeout_s: float = 120.0
    prefill_direct_buffer_service_url: Optional[str] = None
    prefill_direct_buffer_timeout_s: float = 30.0
    release_direct_feature_buffers_after_prefill: bool = True
    strict_no_fallback: bool = field(default_factory=strict_no_fallback_enabled)
    enable_agent_state_clone: bool = True
    high_prefill_worker_ids: List[str] = field(default_factory=list)
    low_latency_decode_worker_ids: List[str] = field(default_factory=list)
    standard_prefill_worker_ids: List[str] = field(default_factory=list)
    standard_decode_worker_ids: List[str] = field(default_factory=list)


@dataclass
class _DispatchContext:
    request_id: str
    request_body: Dict[str, Any]
    control_ctx: Any


@dataclass
class _PrefillContinuation:
    text: str = ""
    completion_tokens: int = 0
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.completion_tokens > 0 and bool(self.text)


def parse_args() -> ProxyConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--prefiller-hosts", "--prefiller-host", type=str, nargs="+", default=["127.0.0.1"])
    parser.add_argument("--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100])
    parser.add_argument("--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["127.0.0.1"])
    parser.add_argument("--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200])
    parser.add_argument("--layers-per-group", type=int, default=4)
    parser.add_argument("--group-delay-ms", type=float, default=0.0)
    parser.add_argument("--max-group-bytes", type=int, default=0)
    parser.add_argument("--warn-rho", type=float, default=0.85)
    parser.add_argument("--critical-rho", type=float, default=0.95)
    parser.add_argument("--max-backpressure-delay-ms", type=float, default=150.0)
    parser.add_argument("--transport-backend", type=str, default="mooncake_engine_direct")
    parser.add_argument("--node-id", type=str, default="proxy")
    parser.add_argument("--owner-shards", type=int, default=1)
    parser.add_argument("--kv-directory-rpc-url", type=str, default=None)
    parser.add_argument("--connector-metrics-dir", type=str, default=None)
    parser.add_argument("--workflow-registry-wal", type=str, default=None)
    parser.add_argument("--enable-mm-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mm-prefetch-mode", choices=["asset_bytes", "feature_handle"], default="asset_bytes")
    parser.add_argument("--prefill-supports-feature-handles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mm-prefetch-wait-ms", type=float, default=100.0)
    parser.add_argument("--mm-prefetch-max-asset-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--mm-prefetch-queue-size", type=int, default=256)
    parser.add_argument("--encoder-service-url", type=str, default=os.getenv("MOONCAKE_EPD_ENCODER_SERVICE_URL"))
    parser.add_argument("--encoder-service-timeout-s", type=float, default=float(os.getenv("MOONCAKE_EPD_ENCODER_SERVICE_TIMEOUT_S", "120")))
    parser.add_argument("--prefill-direct-buffer-service-url", type=str, default=os.getenv("MOONCAKE_EPD_PREFILL_DIRECT_BUFFER_SERVICE_URL"))
    parser.add_argument("--prefill-direct-buffer-timeout-s", type=float, default=float(os.getenv("MOONCAKE_EPD_PREFILL_DIRECT_BUFFER_TIMEOUT_S", "30")))
    parser.add_argument("--release-direct-feature-buffers-after-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-no-fallback", action=argparse.BooleanOptionalAction, default=strict_no_fallback_enabled())
    parser.add_argument("--enable-agent-state-clone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high-prefill-worker-ids", nargs="*", default=None)
    parser.add_argument("--low-latency-decode-worker-ids", nargs="*", default=None)
    parser.add_argument("--standard-prefill-worker-ids", nargs="*", default=None)
    parser.add_argument("--standard-decode-worker-ids", nargs="*", default=None)
    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    return ProxyConfig(
        host=args.host,
        port=args.port,
        prefiller_instances=list(zip(args.prefiller_hosts, args.prefiller_ports)),
        decoder_instances=list(zip(args.decoder_hosts, args.decoder_ports)),
        layers_per_group=args.layers_per_group,
        group_delay_ms=args.group_delay_ms,
        max_group_bytes=args.max_group_bytes,
        warn_rho=args.warn_rho,
        critical_rho=args.critical_rho,
        max_backpressure_delay_ms=args.max_backpressure_delay_ms,
        transport_backend=args.transport_backend,
        node_id=args.node_id,
        owner_shards=args.owner_shards,
        kv_directory_rpc_url=args.kv_directory_rpc_url,
        connector_metrics_dir=args.connector_metrics_dir,
        workflow_registry_wal_path=args.workflow_registry_wal,
        enable_mm_prefetch=bool(args.enable_mm_prefetch),
        mm_prefetch_mode=str(args.mm_prefetch_mode),
        prefill_supports_feature_handles=bool(args.prefill_supports_feature_handles),
        mm_prefetch_wait_ms=args.mm_prefetch_wait_ms,
        mm_prefetch_max_asset_bytes=args.mm_prefetch_max_asset_bytes,
        mm_prefetch_queue_size=args.mm_prefetch_queue_size,
        encoder_service_url=args.encoder_service_url,
        encoder_service_timeout_s=args.encoder_service_timeout_s,
        prefill_direct_buffer_service_url=args.prefill_direct_buffer_service_url,
        prefill_direct_buffer_timeout_s=args.prefill_direct_buffer_timeout_s,
        release_direct_feature_buffers_after_prefill=bool(args.release_direct_feature_buffers_after_prefill),
        strict_no_fallback=bool(args.strict_no_fallback),
        enable_agent_state_clone=bool(args.enable_agent_state_clone),
        high_prefill_worker_ids=list(args.high_prefill_worker_ids or []),
        low_latency_decode_worker_ids=list(args.low_latency_decode_worker_ids or []),
        standard_prefill_worker_ids=list(args.standard_prefill_worker_ids or []),
        standard_decode_worker_ids=list(args.standard_decode_worker_ids or []),
    )


def _make_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=None,
        base_url=base_url,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
        trust_env=False,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config: ProxyConfig = app.state.proxy_config
    control_plane: ServingControlPlane = app.state.control_plane
    prefill_overrides = getattr(app.state, "prefill_client_overrides", None)
    decode_overrides = getattr(app.state, "decode_client_overrides", None)

    if prefill_overrides is None:
        app.state.prefill_clients = [
            {
                "client": _make_client(f"http://{host}:{port}"),
                "host": host,
                "port": port,
                "id": idx,
                "worker_id": f"prefill-{idx}",
            }
            for idx, (host, port) in enumerate(config.prefiller_instances)
        ]
    else:
        app.state.prefill_clients = list(prefill_overrides)

    if decode_overrides is None:
        app.state.decode_clients = [
            {
                "client": _make_client(f"http://{host}:{port}"),
                "host": host,
                "port": port,
                "id": idx,
                "worker_id": f"decode-{idx}",
            }
            for idx, (host, port) in enumerate(config.decoder_instances)
        ]
    else:
        app.state.decode_clients = list(decode_overrides)

    control_plane.register_stage_workers(
        "prefill", [client["worker_id"] for client in app.state.prefill_clients]
    )
    control_plane.register_stage_workers(
        "decode", [client["worker_id"] for client in app.state.decode_clients]
    )
    app.state.mm_fetch_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        follow_redirects=True,
        trust_env=False,
    )
    app.state.encoder_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.encoder_service_timeout_s, connect=10.0),
        follow_redirects=True,
        trust_env=False,
        base_url=(config.encoder_service_url.rstrip("/") if config.encoder_service_url else ""),
    )
    app.state.prefill_direct_buffer_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.prefill_direct_buffer_timeout_s, connect=5.0),
        follow_redirects=True,
        trust_env=False,
        base_url=(
            config.prefill_direct_buffer_service_url.rstrip("/")
            if config.prefill_direct_buffer_service_url
            else ""
        ),
    )

    try:
        yield
    finally:
        mm_fetch_client = getattr(app.state, "mm_fetch_client", None)
        if mm_fetch_client is not None:
            await mm_fetch_client.aclose()
        encoder_client = getattr(app.state, "encoder_client", None)
        if encoder_client is not None:
            await encoder_client.aclose()
        prefill_direct_client = getattr(app.state, "prefill_direct_buffer_client", None)
        if prefill_direct_client is not None:
            await prefill_direct_client.aclose()
        mm_store = getattr(app.state, "mm_store", None)
        if mm_store is not None:
            mm_store.stop()
        for client_info in list(app.state.prefill_clients) + list(app.state.decode_clients):
            client = client_info.get("client")
            if client is not None:
                await client.aclose()


def create_app(
    config: Optional[ProxyConfig] = None,
    *,
    prefill_clients: Optional[Sequence[Dict[str, Any]]] = None,
    decode_clients: Optional[Sequence[Dict[str, Any]]] = None,
    control_plane: Optional[ServingControlPlane] = None,
) -> FastAPI:
    config = config or ProxyConfig()
    cp = control_plane or ServingControlPlane(
        ServingControlPlaneConfig(
            node_id=config.node_id,
            layers_per_group=config.layers_per_group,
            group_delay_ms=config.group_delay_ms,
            max_group_bytes=config.max_group_bytes,
            warn_rho=config.warn_rho,
            critical_rho=config.critical_rho,
            max_backpressure_delay_ms=config.max_backpressure_delay_ms,
            transport_backend=config.transport_backend,
            owner_shards=config.owner_shards,
            kv_directory_rpc_url=config.kv_directory_rpc_url,
            connector_metrics_dir=config.connector_metrics_dir,
            workflow_registry_wal_path=config.workflow_registry_wal_path,
            enable_mm_prefetch=config.enable_mm_prefetch,
            strict_no_fallback=config.strict_no_fallback,
            enable_agent_state_clone=config.enable_agent_state_clone,
            high_prefill_worker_ids=tuple(config.high_prefill_worker_ids),
            low_latency_decode_worker_ids=tuple(config.low_latency_decode_worker_ids),
            standard_prefill_worker_ids=tuple(config.standard_prefill_worker_ids),
            standard_decode_worker_ids=tuple(config.standard_decode_worker_ids),
        )
    )
    app = FastAPI(lifespan=_lifespan)
    app.state.proxy_config = config
    app.state.control_plane = cp
    app.state.prefill_client_overrides = list(prefill_clients) if prefill_clients is not None else None
    app.state.decode_client_overrides = list(decode_clients) if decode_clients is not None else None
    app.state.mm_store = MMStore(
        transfer_engine=TransferEngine(protocol="local"),
        max_queue_size=max(1, int(config.mm_prefetch_queue_size)),
        dispatcher_workers=2,
        inline_fallback_on_queue_full=not config.strict_no_fallback,
    ) if config.enable_mm_prefetch else None

    @app.get("/health")
    @app.get("/healthcheck")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "prefill_clients": len(app.state.prefill_clients), "decode_clients": len(app.state.decode_clients)}

    @app.get("/metrics")
    async def metrics() -> Dict[str, Any]:
        payload = cp.snapshot()
        mm_store = getattr(app.state, "mm_store", None)
        if mm_store is not None:
            payload["mm_store"] = mm_store.stats()
        return payload

    @app.post("/mooncake_epd/agent/fork")
    async def fork_agent_state(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return cp.fork_workflow_state(
                workflow_id=str(payload.get("workflow_id") or ""),
                parent_request_id=(
                    str(payload.get("parent_request_id"))
                    if payload.get("parent_request_id") is not None
                    else None
                ),
                branch_count=int(payload.get("branch_count", 2) or 2),
                target_node_id=(
                    str(payload.get("target_node_id"))
                    if payload.get("target_node_id") is not None
                    else None
                ),
                for_write=bool(payload.get("for_write", False)),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/completions")
    async def handle_completions(request: Request):
        return await _handle_generation_request(app, "/v1/completions", request)

    @app.post("/v1/chat/completions")
    async def handle_chat_completions(request: Request):
        return await _handle_generation_request(app, "/v1/chat/completions", request)

    return app


async def _handle_generation_request(app: FastAPI, api: str, request: Request):
    req_data = await request.json()
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    req_data = _merge_control_headers(req_data, request)
    control_plane: ServingControlPlane = app.state.control_plane
    ctx = control_plane.start_request(req_data, request_id)

    try:
        prefill_decision = _admit_or_raise(control_plane, "prefill", ctx)
    except HTTPException:
        control_plane.finish_request(request_id)
        raise
    decode_peek = _peek_lowest_load_worker(control_plane, "decode")
    prefill_client = _client_for_worker(app.state.prefill_clients, prefill_decision.worker_id)
    if prefill_client is None:
        control_plane.mark_stage_complete(
            "prefill",
            prefill_decision.worker_id,
            latency_ms=0.0,
            success=False,
        )
        control_plane.finish_request(request_id)
        raise HTTPException(status_code=503, detail="no prefill client available")

    if prefill_decision.wait_ms > 0:
        await asyncio.sleep(prefill_decision.wait_ms / 1000.0)

    try:
        req_data = await _prepare_multimodal_inputs_for_prefill(
            app=app,
            req_data=req_data,
            ctx=ctx,
            target_worker_id=prefill_decision.worker_id,
        )
    except HTTPException:
        control_plane.mark_stage_complete(
            "prefill",
            prefill_decision.worker_id,
            latency_ms=0.0,
            success=False,
        )
        control_plane.finish_request(request_id)
        raise

    prefill_headers = _forward_headers(request, request_id)
    prefill_start = time.perf_counter()
    prefill_response = None
    prompt_only_prefill = _should_use_prompt_only_prefill(api)
    try:
        prefill_base_params = dict(req_data.get("kv_transfer_params") or {})
        prefill_kv_params = control_plane.build_prefill_kv_params(
            ctx,
            prefill_decision,
            decode_worker_id=(decode_peek.worker_id if decode_peek is not None else None),
            base_params=prefill_base_params,
        )
        if prefill_base_params.get("mm_feature_handles"):
            prefill_kv_params["mm_prefetch_policy"] = "feature_handle"
            prefill_kv_params["mm_feature_handles"] = prefill_base_params["mm_feature_handles"]
            prefill_kv_params["mm_feature_handle_target_worker"] = prefill_base_params.get(
                "mm_feature_handle_target_worker",
                prefill_decision.worker_id,
            )
        if prompt_only_prefill:
            prefill_json = await _dispatch_prompt_only_prefill(
                api=api,
                prefill_client=prefill_client,
                prefill_headers=prefill_headers,
                request_id=request_id,
                request_body=req_data,
                kv_transfer_params=prefill_kv_params,
            )
        else:
            prefill_payload = dict(req_data)
            prefill_payload["kv_transfer_params"] = prefill_kv_params
            prefill_payload["stream"] = False
            prefill_payload["max_tokens"] = 1
            if "max_completion_tokens" in prefill_payload:
                prefill_payload["max_completion_tokens"] = 1
            prefill_payload.pop("stream_options", None)
            prefill_response = await prefill_client["client"].post(
                api,
                json=prefill_payload,
                headers=prefill_headers,
            )
            prefill_response.raise_for_status()
            prefill_json = prefill_response.json()
    except Exception as exc:
        control_plane.mark_stage_complete(
            "prefill",
            prefill_client["worker_id"],
            latency_ms=(time.perf_counter() - prefill_start) * 1000.0,
            success=False,
        )
        control_plane.finish_request(request_id)
        raise HTTPException(status_code=502, detail=f"prefill request failed: {exc}") from exc
    finally:
        if prefill_response is not None:
            await prefill_response.aclose()

    await _release_direct_feature_buffers_after_prefill(app, req_data)

    control_plane.mark_stage_complete(
        "prefill",
        prefill_client["worker_id"],
        latency_ms=(time.perf_counter() - prefill_start) * 1000.0,
        success=True,
    )
    prefill_continuation = (
        _PrefillContinuation()
        if prompt_only_prefill
        else _extract_prefill_continuation(api, prefill_json)
    )
    if not prompt_only_prefill and _should_short_circuit_after_prefill(req_data, prefill_continuation):
        control_plane.finish_request(request_id)
        return _build_prefill_terminal_response(
            api=api,
            prefill_json=prefill_json,
            request_id=request_id,
            routing_path=ctx.routing_path,
            admission_action=prefill_decision.decision.action.value,
            degrade_level=ctx.degrade_level.value,
        )

    try:
        decode_decision = _admit_or_raise(control_plane, "decode", ctx)
    except HTTPException:
        control_plane.finish_request(request_id)
        raise
    decode_client = _client_for_worker(app.state.decode_clients, decode_decision.worker_id)
    if decode_client is None:
        control_plane.mark_stage_complete(
            "decode",
            decode_decision.worker_id,
            latency_ms=0.0,
            success=False,
        )
        control_plane.finish_request(request_id)
        raise HTTPException(status_code=503, detail="no decode client available")

    try:
        prefill_kv = control_plane.note_prefill_response(
            ctx,
            prefill_json.get("kv_transfer_params"),
            decode_worker_id=decode_client["worker_id"],
        )
    except Exception as exc:
        control_plane.mark_stage_complete(
            "decode",
            decode_decision.worker_id,
            latency_ms=0.0,
            success=False,
        )
        control_plane.finish_request(request_id)
        raise HTTPException(
            status_code=502,
            detail=f"prefill response missing usable KV handoff metadata: {exc}",
        ) from exc

    if decode_decision.wait_ms > 0:
        await asyncio.sleep(decode_decision.wait_ms / 1000.0)

    decode_payload = dict(req_data)
    decode_payload = _apply_prefill_continuation_to_decode_payload(
        api=api,
        decode_payload=decode_payload,
        continuation=prefill_continuation,
    )
    try:
        decode_payload["kv_transfer_params"] = control_plane.build_decode_kv_params(
            ctx,
            decode_decision,
            prefill_kv,
        )
        if prefill_continuation.active:
            decode_payload["kv_transfer_params"].update(
                _build_prefill_decode_semantic_hints(
                    continuation=prefill_continuation,
                    prefill_kv=prefill_kv,
                )
            )
    except Exception as exc:
        control_plane.rollback_handoff(ctx)
        control_plane.mark_stage_complete(
            "decode",
            decode_decision.worker_id,
            latency_ms=0.0,
            success=False,
        )
        control_plane.finish_request(request_id)
        raise HTTPException(
            status_code=502,
            detail=f"decode request missing usable KV transfer params: {exc}",
        ) from exc

    response_headers = {
        "X-Request-Id": request_id,
        "X-EPD-Routing-Path": ctx.routing_path,
        "X-EPD-Admission": decode_decision.decision.action.value,
        "X-EPD-Degrade-Level": ctx.degrade_level.value,
    }
    if bool(req_data.get("stream")):
        return await _dispatch_streaming_decode(
            api=api,
            control_plane=control_plane,
            ctx=ctx,
            decode_client=decode_client,
            decode_payload=decode_payload,
            decode_headers=_forward_headers(request, request_id),
            response_headers=response_headers,
            continuation=prefill_continuation,
        )
    return await _dispatch_non_streaming_decode(
        api=api,
        control_plane=control_plane,
        ctx=ctx,
        decode_client=decode_client,
        decode_payload=decode_payload,
        decode_headers=_forward_headers(request, request_id),
        response_headers=response_headers,
        continuation=prefill_continuation,
    )



async def _prepare_multimodal_inputs_for_prefill(
    *,
    app: FastAPI,
    req_data: Dict[str, Any],
    ctx,
    target_worker_id: str,
) -> Dict[str, Any]:
    config: ProxyConfig = app.state.proxy_config
    mode = str(config.mm_prefetch_mode or "asset_bytes").strip().lower()
    if mode == "asset_bytes":
        return await _prefetch_and_rewrite_multimodal_assets(
            app=app,
            req_data=req_data,
            ctx=ctx,
            target_worker_id=target_worker_id,
        )
    if mode == "feature_handle":
        return await _prepare_feature_handle_multimodal_inputs(
            app=app,
            req_data=req_data,
            ctx=ctx,
            config=config,
            target_worker_id=target_worker_id,
        )
    raise HTTPException(status_code=500, detail=f"unsupported mm_prefetch_mode: {mode}")


async def _prepare_feature_handle_multimodal_inputs(
    *,
    app: FastAPI,
    req_data: Dict[str, Any],
    ctx,
    config: ProxyConfig,
    target_worker_id: str,
) -> Dict[str, Any]:
    if not config.enable_mm_prefetch or not getattr(ctx, "mm_hashes", None):
        return req_data
    if not config.prefill_supports_feature_handles:
        raise HTTPException(
            status_code=501,
            detail=(
                "mm_prefetch_mode=feature_handle requires a Prefill runtime that "
                "can consume external multimodal hidden-state handles"
            ),
        )
    metadata = dict(req_data.get("metadata") or {})
    raw_handles = (
        metadata.get("mooncake_epd_feature_handles")
        or metadata.get("feature_handles")
        or req_data.get("mooncake_epd_feature_handles")
    )
    if not isinstance(raw_handles, list) or not raw_handles:
        raw_handles = await _request_feature_handles_from_encoder_service(
            app=app,
            req_data=req_data,
            target_worker_id=target_worker_id,
        )
    try:
        handles = [FeatureHandle.from_control_payload(dict(item)) for item in raw_handles]
        handle_payloads = [handle.as_control_payload() for handle in handles]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid feature handle payload: {exc}") from exc
    expected = len(list(getattr(ctx, "mm_hashes", []) or []))
    if expected and len(handles) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"feature handle count mismatch: handles={len(handles)} multimodal_items={expected}",
        )
    for handle, image_hash in zip(handles, list(getattr(ctx, "mm_hashes", []) or [])):
        if handle.feature_id != image_hash and handle.metadata.get("source_mm_hash") != image_hash:
            raise HTTPException(
                status_code=400,
                detail=(
                    "feature handle does not match request multimodal hash: "
                    f"handle={handle.feature_id} request={image_hash}"
                ),
            )
    rewritten = json.loads(json.dumps(req_data))
    metadata = dict(rewritten.get("metadata") or {})
    metadata["mooncake_epd_feature_handles"] = handle_payloads
    metadata["mooncake_epd_feature_handle_target_worker"] = target_worker_id
    rewritten["metadata"] = metadata
    kv = dict(rewritten.get("kv_transfer_params") or {})
    kv["mm_prefetch_policy"] = "feature_handle"
    kv["mm_feature_handles"] = handle_payloads
    kv["mm_feature_handle_target_worker"] = target_worker_id
    rewritten["kv_transfer_params"] = kv
    return rewritten


async def _request_feature_handles_from_encoder_service(
    *,
    app: FastAPI,
    req_data: Dict[str, Any],
    target_worker_id: str,
) -> List[Dict[str, Any]]:
    config: ProxyConfig = app.state.proxy_config
    if not config.encoder_service_url:
        raise HTTPException(
            status_code=400,
            detail=(
                "feature_handle mode requires metadata.mooncake_epd_feature_handles "
                "or --encoder-service-url for online E-stage encoding"
            ),
        )
    payload = json.loads(json.dumps(req_data))
    metadata = dict(payload.get("metadata") or {})
    metadata["mooncake_epd_target_worker_id"] = target_worker_id
    payload["metadata"] = metadata
    if config.prefill_direct_buffer_service_url:
        return await _request_direct_feature_handles_from_encoder_service(
            app=app,
            payload=payload,
            target_worker_id=target_worker_id,
        )
    client: httpx.AsyncClient = app.state.encoder_client
    try:
        response = await client.post("/encode", json=payload)
        response.raise_for_status()
        encoded = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"encoder service returned error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"encoder service request failed: {exc}") from exc
    handles = encoded.get("handles")
    if not isinstance(handles, list) or not handles:
        raise HTTPException(status_code=502, detail="encoder service returned no feature handles")
    return [dict(item) for item in handles]


async def _request_direct_feature_handles_from_encoder_service(
    *,
    app: FastAPI,
    payload: Dict[str, Any],
    target_worker_id: str,
) -> List[Dict[str, Any]]:
    encoder_client: httpx.AsyncClient = app.state.encoder_client
    direct_client: httpx.AsyncClient = app.state.prefill_direct_buffer_client
    try:
        described_resp = await encoder_client.post("/describe", json=payload)
        described_resp.raise_for_status()
        described = described_resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"encoder describe returned error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"encoder describe request failed: {exc}") from exc

    descriptors = described.get("descriptors")
    ticket = str(described.get("ticket") or "")
    if not ticket or not isinstance(descriptors, list) or not descriptors:
        raise HTTPException(status_code=502, detail="encoder describe returned no direct descriptors/ticket")

    try:
        alloc_resp = await direct_client.post(
            "allocate",
            json={
                "descriptors": descriptors,
                "target_worker_id": target_worker_id,
                "zero_fill": False,
            },
        )
        alloc_resp.raise_for_status()
        allocation = alloc_resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"prefill direct allocation returned error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"prefill direct allocation failed: {exc}") from exc

    targets = allocation.get("targets")
    if not isinstance(targets, list) or len(targets) != len(descriptors):
        raise HTTPException(
            status_code=502,
            detail=(
                "prefill direct allocation target count mismatch: "
                f"targets={0 if not isinstance(targets, list) else len(targets)} descriptors={len(descriptors)}"
            ),
        )

    publish_payload = {
        "ticket": ticket,
        "metadata": dict(payload.get("metadata") or {}),
        "mooncake_epd_direct_feature_targets": targets,
    }
    try:
        publish_resp = await encoder_client.post("/publish_direct", json=publish_payload)
        publish_resp.raise_for_status()
        published = publish_resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"encoder direct publish returned error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"encoder direct publish failed: {exc}") from exc
    handles = published.get("handles")
    if not isinstance(handles, list) or not handles:
        raise HTTPException(status_code=502, detail="encoder direct publish returned no feature handles")
    for handle in handles:
        if not str(dict(handle).get("uri") or "").startswith("epd-direct://"):
            raise HTTPException(status_code=502, detail="encoder direct publish returned non-direct handle")
    return [dict(item) for item in handles]


async def _release_direct_feature_buffers_after_prefill(app: FastAPI, req_data: Dict[str, Any]) -> None:
    """Release Prefill-owned E→P direct buffers once Prefill has consumed them."""

    config: ProxyConfig = app.state.proxy_config
    if not config.release_direct_feature_buffers_after_prefill:
        return
    if not config.prefill_direct_buffer_service_url:
        return
    feature_ids: List[str] = []
    kv = dict(req_data.get("kv_transfer_params") or {})
    metadata = dict(req_data.get("metadata") or {})
    raw_handles = (
        kv.get("mm_feature_handles")
        or metadata.get("mooncake_epd_feature_handles")
        or metadata.get("feature_handles")
        or []
    )
    if not isinstance(raw_handles, list):
        return
    for item in raw_handles:
        if not isinstance(item, dict):
            continue
        if not str(item.get("uri") or "").startswith("epd-direct://"):
            continue
        fid = str(item.get("feature_id") or "")
        if fid:
            feature_ids.append(fid)
    if not feature_ids:
        return
    client: httpx.AsyncClient = app.state.prefill_direct_buffer_client
    try:
        response = await client.post("release", json={"feature_ids": sorted(set(feature_ids))})
        response.raise_for_status()
    except Exception as exc:
        # Release failure is operationally serious but the P→D handoff may have
        # already succeeded. Report through logs/metrics rather than corrupting
        # the user response after Prefill has completed.
        logger.error("failed to release direct feature buffers after prefill: %s", exc)


async def _prefetch_and_rewrite_multimodal_assets(
    *,
    app: FastAPI,
    req_data: Dict[str, Any],
    ctx,
    target_worker_id: str,
) -> Dict[str, Any]:
    """Prefetch multimodal assets into MMStore and rewrite URLs to data URLs.

    This is the serving-compatible E→P prefetch path for the OpenAI/vLLM API:
    the proxy owns multimodal source bytes, moves them through the real MMStore
    event queue/TransferEngine, and sends prefill/decode identical data URLs so
    vLLM avoids repeated remote image fetches while preserving request semantics.
    """
    config: ProxyConfig = app.state.proxy_config
    mm_store: Optional[MMStore] = getattr(app.state, "mm_store", None)
    if not config.enable_mm_prefetch or mm_store is None or not getattr(ctx, "mm_hashes", None):
        return req_data

    rewritten = json.loads(json.dumps(req_data))
    items = list(_iter_mutable_mm_url_items(rewritten))
    if not items:
        return req_data

    attempted = completed = failed = worker_hits = shared_hits = recomputed = 0
    bytes_total = 0
    wait_start = time.perf_counter()
    hash_iter = iter(list(ctx.mm_hashes))

    for item in items:
        image_hash = next(hash_iter, None)
        if not image_hash:
            break
        url = _image_url_from_item(item)
        if not url:
            continue
        attempted += 1
        try:
            bundle, data_url, shared_hit = await _get_or_create_serving_mm_bundle(
                app=app,
                mm_store=mm_store,
                image_hash=image_hash,
                url=url,
                max_bytes=int(config.mm_prefetch_max_asset_bytes),
            )
            bytes_total += int(bundle.metadata.get("bytes", bundle.nbytes()) or 0)
            if shared_hit:
                shared_hits += 1
            handle = mm_store.prefetch(
                image_hash,
                target_worker_id=target_worker_id,
                target_device=torch.device("cpu"),
            )
            timeout_s = max(0.0, float(config.mm_prefetch_wait_ms)) / 1000.0
            if timeout_s > 0:
                await asyncio.to_thread(handle.wait, timeout_s)
            if handle.done.is_set() and handle.error is None:
                completed += 1
                worker_hits += int(bool(handle.worker_cache_hit))
                shared_hits += int(bool(handle.shared_store_hit and not shared_hit))
                recomputed += int(bool(handle.recomputed))
            _set_image_url_on_item(item, data_url)
        except Exception:
            failed += 1
            logger.exception("MM prefetch failed")
            if config.strict_no_fallback:
                raise HTTPException(
                    status_code=502,
                    detail="MM prefetch failed in strict-no-fallback mode",
                )
            logger.warning("preserving original multimodal URL after MM prefetch failure")

    wait_ms = (time.perf_counter() - wait_start) * 1000.0
    control_plane: ServingControlPlane = app.state.control_plane
    control_plane.record_mm_prefetch_result(
        ctx,
        attempted=attempted,
        completed=completed,
        failed=failed,
        worker_cache_hits=worker_hits,
        shared_store_hits=shared_hits,
        recomputed=recomputed,
        bytes_transferred=bytes_total,
        wait_ms=wait_ms if attempted else 0.0,
    )
    return rewritten


def _iter_mutable_mm_url_items(req_data: Dict[str, Any]):
    messages = req_data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and _image_url_from_item(item):
                        yield item
    prompt = req_data.get("prompt")
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and _image_url_from_item(item):
                yield item


def _image_url_from_item(item: Dict[str, Any]) -> Optional[str]:
    item_type = str(item.get("type", "")).strip().lower()
    if item_type not in {"image", "image_url", "input_image"}:
        return None
    image_url = item.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict) and image_url.get("url"):
        return str(image_url.get("url"))
    if item.get("url"):
        return str(item.get("url"))
    return None


def _set_image_url_on_item(item: Dict[str, Any], data_url: str) -> None:
    if isinstance(item.get("image_url"), dict):
        updated = dict(item["image_url"])
        updated["url"] = data_url
        item["image_url"] = updated
    elif "image_url" in item:
        item["image_url"] = data_url
    else:
        item["url"] = data_url


async def _get_or_create_serving_mm_bundle(
    *,
    app: FastAPI,
    mm_store: MMStore,
    image_hash: str,
    url: str,
    max_bytes: int,
) -> Tuple[FeatureBundle, str, bool]:
    cached = mm_store.shared_store.get(image_hash)
    if cached is not None:
        data_url = str(cached.metadata.get("data_url") or "")
        if data_url:
            return cached, data_url, True
    payload, content_type = await _load_mm_url_bytes(app, url, max_bytes=max_bytes)
    data_url = _bytes_to_data_url(payload, content_type)
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    bundle = FeatureBundle(
        image_hash=image_hash,
        last_hidden=tensor,
        intermediates=[],
        grid_thw=None,
        metadata={
            "kind": "serving_mm_asset_bytes",
            "content_type": content_type,
            "bytes": len(payload),
            "source_url_sha256": _stable_text_hash(url),
            "data_url": data_url,
        },
    )
    mm_store.publish(bundle)
    return bundle, data_url, False


async def _load_mm_url_bytes(app: FastAPI, url: str, *, max_bytes: int) -> Tuple[bytes, str]:
    if url.startswith("data:"):
        return _parse_data_url(url, max_bytes=max_bytes)
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"unsupported multimodal URL scheme: {url[:32]}")
    client: httpx.AsyncClient = app.state.mm_fetch_client
    response = await client.get(url)
    response.raise_for_status()
    payload = response.content
    if len(payload) > max_bytes:
        raise ValueError(f"multimodal asset too large: {len(payload)} > {max_bytes}")
    content_type = response.headers.get("content-type", "application/octet-stream").split(";")[0].strip()
    return payload, content_type or "application/octet-stream"


def _parse_data_url(url: str, *, max_bytes: int) -> Tuple[bytes, str]:
    header, sep, data = url.partition(",")
    if sep != "," or not header.startswith("data:"):
        raise ValueError("invalid data URL")
    meta = header[5:]
    parts = [part for part in meta.split(";") if part]
    content_type = parts[0] if parts and "/" in parts[0] else "application/octet-stream"
    if "base64" in parts:
        payload = base64.b64decode(data, validate=True)
    else:
        from urllib.parse import unquote_to_bytes
        payload = unquote_to_bytes(data)
    if len(payload) > max_bytes:
        raise ValueError(f"multimodal asset too large: {len(payload)} > {max_bytes}")
    return payload, content_type


def _bytes_to_data_url(payload: bytes, content_type: str) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{content_type or 'application/octet-stream'};base64,{encoded}"


def _stable_text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _peek_lowest_load_worker(control_plane: ServingControlPlane, stage: str):
    workers = control_plane.stage_workers(stage)
    if not workers:
        return None
    return min(workers, key=lambda w: (w.current_load + w.queue_size, w.avg_latency_ms))


def _merge_control_headers(req_data: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Copy stable Agent scheduling hints from HTTP headers into metadata.

    This keeps the public OpenAI request body compatible while letting real
    Agent gateways express THINKING / INTERACTIVE / HYBRID, priority and SLO.
    Body metadata wins over headers.
    """

    header_map = {
        "X-Agent-Type": "agent_type",
        "X-Agent-Priority": "priority",
        "X-Agent-Deadline-Ms": "deadline_ms",
        "X-Agent-SLO-Ms": "slo_ms",
        "X-Workflow-Id": "workflow_id",
    }
    updates: Dict[str, Any] = {}
    for header, key in header_map.items():
        value = request.headers.get(header)
        if value is not None and str(value).strip():
            updates[key] = value
    if not updates:
        return req_data
    merged = json.loads(json.dumps(req_data))
    metadata = dict(merged.get("metadata") or {})
    for key, value in updates.items():
        metadata.setdefault(key, value)
    merged["metadata"] = metadata
    return merged


def _admit_or_raise(control_plane: ServingControlPlane, stage: str, ctx) -> Any:
    try:
        decision = control_plane.admit_stage(stage, ctx)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if decision.decision.action is AdmissionAction.REJECT:
        raise HTTPException(status_code=503, detail=decision.decision.reason)
    return decision


def _client_for_worker(clients: Sequence[Dict[str, Any]], worker_id: str) -> Optional[Dict[str, Any]]:
    for client in clients:
        if client.get("worker_id") == worker_id:
            return client
    return None


def _should_use_prompt_only_prefill(api: str) -> bool:
    return api.endswith("/chat/completions") or api.endswith("/completions")


def _render_api_for(api: str) -> str:
    if api.endswith("/chat/completions"):
        return "/v1/chat/completions/render"
    if api.endswith("/completions"):
        return "/v1/completions/render"
    raise ValueError(f"unsupported render api for {api}")


def _normalize_rendered_prefill_payload(api: str, render_payload: Any) -> Dict[str, Any]:
    if api.endswith("/chat/completions"):
        if not isinstance(render_payload, dict):
            raise TypeError("chat render response must be an object")
        return dict(render_payload)
    if not isinstance(render_payload, list) or len(render_payload) != 1:
        raise ValueError("completion render response must contain exactly one prompt")
    first = render_payload[0]
    if not isinstance(first, dict):
        raise TypeError("completion render response item must be an object")
    return dict(first)


def _inject_prefill_kv_into_rendered_request(
    rendered_request: Dict[str, Any],
    *,
    request_id: str,
    kv_transfer_params: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(rendered_request)
    sampling_params = dict(payload.get("sampling_params") or {})
    extra_args = dict(sampling_params.get("extra_args") or {})
    extra_args["kv_transfer_params"] = dict(kv_transfer_params)
    sampling_params["extra_args"] = extra_args
    sampling_params["max_tokens"] = 0
    sampling_params["min_tokens"] = 0
    payload["sampling_params"] = sampling_params
    payload["request_id"] = request_id
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["kv_transfer_params"] = dict(kv_transfer_params)
    return payload


async def _dispatch_prompt_only_prefill(
    *,
    api: str,
    prefill_client: Dict[str, Any],
    prefill_headers: Dict[str, str],
    request_id: str,
    request_body: Dict[str, Any],
    kv_transfer_params: Dict[str, Any],
) -> Dict[str, Any]:
    render_response = await prefill_client["client"].post(
        _render_api_for(api),
        json=request_body,
        headers=prefill_headers,
    )
    try:
        render_response.raise_for_status()
        rendered_payload = _normalize_rendered_prefill_payload(
            api,
            render_response.json(),
        )
    finally:
        await render_response.aclose()

    prefill_payload = _inject_prefill_kv_into_rendered_request(
        rendered_payload,
        request_id=request_id,
        kv_transfer_params=kv_transfer_params,
    )
    generate_response = await prefill_client["client"].post(
        "/inference/v1/generate",
        json=prefill_payload,
        headers=prefill_headers,
    )
    try:
        generate_response.raise_for_status()
        payload = dict(generate_response.json())
    finally:
        await generate_response.aclose()
    payload.setdefault("kv_transfer_params", dict(kv_transfer_params))
    return payload


def _forward_headers(request: Request, request_id: str) -> Dict[str, str]:
    auth = request.headers.get("Authorization") or f"Bearer {os.environ.get('OPENAI_API_KEY', 'sk-local')}"
    headers = {
        "Authorization": auth,
        "X-Request-Id": request_id,
    }
    workflow_id = request.headers.get("X-Workflow-Id")
    if workflow_id:
        headers["X-Workflow-Id"] = workflow_id
    return headers


def _extract_prefill_continuation(api: str, payload: Dict[str, Any]) -> _PrefillContinuation:
    usage = dict(payload.get("usage") or {})
    choices = list(payload.get("choices") or [])
    choice = dict(choices[0] or {}) if choices else {}
    text = _extract_choice_text(api, choice)
    finish_reason = choice.get("finish_reason")
    completion_tokens = int(usage.get("completion_tokens", payload.get("completion_tokens", 0)) or 0)
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    return _PrefillContinuation(
        text=text,
        completion_tokens=completion_tokens,
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
    )


def _requested_completion_budget(req_data: Dict[str, Any]) -> Optional[int]:
    for field_name in ("max_completion_tokens", "max_tokens"):
        value = req_data.get(field_name)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None
    return None


def _should_short_circuit_after_prefill(
    req_data: Dict[str, Any],
    continuation: _PrefillContinuation,
) -> bool:
    requested_budget = _requested_completion_budget(req_data)
    if continuation.finish_reason and continuation.finish_reason != "length":
        return True
    if requested_budget is not None and requested_budget <= max(0, continuation.completion_tokens):
        return True
    return False


def _build_prefill_terminal_response(
    *,
    api: str,
    prefill_json: Dict[str, Any],
    request_id: str,
    routing_path: str,
    admission_action: str,
    degrade_level: str,
) -> Response:
    headers = {
        "X-Request-Id": request_id,
        "X-EPD-Routing-Path": routing_path,
        "X-EPD-Admission": admission_action,
        "X-EPD-Degrade-Level": degrade_level,
    }
    return JSONResponse(prefill_json, headers=headers)


def _apply_prefill_continuation_to_decode_payload(
    *,
    api: str,
    decode_payload: Dict[str, Any],
    continuation: _PrefillContinuation,
) -> Dict[str, Any]:
    if not continuation.active:
        return decode_payload
    requested_budget = _requested_completion_budget(decode_payload)
    if requested_budget is not None:
        remaining_budget = max(1, requested_budget - continuation.completion_tokens)
        if "max_tokens" in decode_payload:
            decode_payload["max_tokens"] = remaining_budget
        if "max_completion_tokens" in decode_payload:
            decode_payload["max_completion_tokens"] = remaining_budget
    if api.endswith("/chat/completions"):
        decode_payload["messages"] = _append_chat_assistant_prefix(
            decode_payload.get("messages"),
            continuation.text,
        )
        decode_payload["continue_final_message"] = True
        decode_payload["add_generation_prompt"] = False
        return decode_payload
    if api.endswith("/completions"):
        prompt = decode_payload.get("prompt")
        decode_payload["prompt"] = _append_completion_prefix(prompt, continuation.text)
    return decode_payload


def _append_chat_assistant_prefix(messages: Any, prefix_text: str) -> List[Dict[str, Any]]:
    existing_messages = [dict(message or {}) for message in list(messages or [])]
    if not existing_messages:
        return [{"role": "assistant", "content": prefix_text}]
    last = dict(existing_messages[-1] or {})
    if str(last.get("role", "")).strip().lower() == "assistant":
        last["content"] = _append_textual_content(last.get("content"), prefix_text)
        existing_messages[-1] = last
        return existing_messages
    existing_messages.append({"role": "assistant", "content": prefix_text})
    return existing_messages


def _append_completion_prefix(prompt: Any, prefix_text: str) -> Any:
    if isinstance(prompt, str):
        return prompt + prefix_text
    if isinstance(prompt, list):
        if not prompt:
            return [prefix_text]
        updated = list(prompt)
        last = updated[-1]
        if isinstance(last, str):
            updated[-1] = last + prefix_text
            return updated
    return prompt


def _append_textual_content(content: Any, prefix_text: str) -> Any:
    if isinstance(content, str):
        return content + prefix_text
    if isinstance(content, list):
        appended = False
        updated: List[Any] = []
        for item in content:
            if (
                not appended
                and isinstance(item, dict)
                and str(item.get("type", "")).strip().lower() == "text"
            ):
                merged = dict(item)
                merged["text"] = str(merged.get("text", "")) + prefix_text
                updated.append(merged)
                appended = True
            else:
                updated.append(item)
        if not appended:
            updated.append({"type": "text", "text": prefix_text})
        return updated
    return prefix_text


def _build_prefill_decode_semantic_hints(
    *,
    continuation: _PrefillContinuation,
    prefill_kv: Dict[str, Any],
) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "remote_prefill_prompt_tokens": continuation.prompt_tokens,
        "remote_prefill_completion_tokens": continuation.completion_tokens,
        "remote_prefill_semantic_continuation": True,
    }
    remote_block_ids = prefill_kv.get("remote_block_ids")
    if isinstance(remote_block_ids, list):
        hints["remote_prefill_block_counts"] = [
            len(group) if isinstance(group, list) else 0 for group in remote_block_ids
        ]
    return hints


async def _dispatch_non_streaming_decode(
    *,
    api: str,
    control_plane: ServingControlPlane,
    ctx,
    decode_client: Dict[str, Any],
    decode_payload: Dict[str, Any],
    decode_headers: Dict[str, str],
    response_headers: Dict[str, str],
    continuation: _PrefillContinuation,
) -> Response:
    decode_start = time.perf_counter()
    decode_response = None
    success = False
    try:
        decode_response = await decode_client["client"].post(api, json=decode_payload, headers=decode_headers)
        decode_response.raise_for_status()
        decode_json = decode_response.json()
        patched_json = _patch_non_stream_payload(api, decode_json, continuation)
        control_plane.commit_handoff(ctx)
        success = True
        return JSONResponse(
            patched_json,
            status_code=decode_response.status_code,
            headers=response_headers,
        )
    except Exception as exc:
        control_plane.rollback_handoff(ctx)
        raise HTTPException(status_code=502, detail=f"decode request failed: {exc}") from exc
    finally:
        if decode_response is not None:
            await decode_response.aclose()
        control_plane.mark_stage_complete(
            "decode",
            decode_client["worker_id"],
            latency_ms=(time.perf_counter() - decode_start) * 1000.0,
            success=success,
        )
        control_plane.finish_request(ctx.request_id)


async def _dispatch_streaming_decode(
    *,
    api: str,
    control_plane: ServingControlPlane,
    ctx,
    decode_client: Dict[str, Any],
    decode_payload: Dict[str, Any],
    decode_headers: Dict[str, str],
    response_headers: Dict[str, str],
    continuation: _PrefillContinuation,
) -> StreamingResponse:
    decode_start = time.perf_counter()
    stream_ctx = decode_client["client"].stream("POST", api, json=decode_payload, headers=decode_headers)
    decode_response = None
    stream_entered = False
    try:
        decode_response = await stream_ctx.__aenter__()
        stream_entered = True
        decode_response.raise_for_status()
    except Exception as exc:
        control_plane.rollback_handoff(ctx)
        if decode_response is not None:
            try:
                await decode_response.aclose()
            except Exception:
                logger.exception("failed to close decode response after startup error")
        if stream_entered:
            try:
                await stream_ctx.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.exception("failed to close decode stream context after startup error")
        control_plane.mark_stage_complete(
            "decode",
            decode_client["worker_id"],
            latency_ms=(time.perf_counter() - decode_start) * 1000.0,
            success=False,
        )
        control_plane.finish_request(ctx.request_id)
        raise HTTPException(status_code=502, detail=f"decode request failed: {exc}") from exc

    async def generate_stream():
        success = True
        first_line_seen = False
        pending_prefix = continuation.text if continuation.active else ""
        try:
            async for raw_line in decode_response.aiter_lines():
                if raw_line is None:
                    continue
                if raw_line == "":
                    continue
                line_out = raw_line
                if raw_line.startswith("data:"):
                    payload = raw_line[5:].strip()
                    if payload == "[DONE]":
                        line_out = "data: [DONE]"
                    else:
                        try:
                            packet = json.loads(payload)
                        except Exception:
                            packet = None
                        if isinstance(packet, dict):
                            packet, pending_prefix = _patch_stream_packet(
                                api=api,
                                packet=packet,
                                continuation=continuation,
                                pending_prefix=pending_prefix,
                            )
                            line_out = f"data: {json.dumps(packet, ensure_ascii=False)}"
                if not first_line_seen:
                    control_plane.commit_handoff(ctx)
                    first_line_seen = True
                yield (line_out + "\n\n").encode("utf-8")
        except Exception:
            success = False
            if not first_line_seen:
                control_plane.rollback_handoff(ctx)
            raise
        finally:
            if success and not first_line_seen:
                success = False
                control_plane.rollback_handoff(ctx)
            try:
                await decode_response.aclose()
            finally:
                await stream_ctx.__aexit__(None, None, None)
                control_plane.mark_stage_complete(
                    "decode",
                    decode_client["worker_id"],
                    latency_ms=(time.perf_counter() - decode_start) * 1000.0,
                    success=success,
                )
                control_plane.finish_request(ctx.request_id)

    media_type = decode_response.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        generate_stream(),
        status_code=decode_response.status_code,
        headers=response_headers,
        media_type=media_type,
    )


def _patch_stream_packet(
    *,
    api: str,
    packet: Dict[str, Any],
    continuation: _PrefillContinuation,
    pending_prefix: str,
) -> tuple[Dict[str, Any], str]:
    choices = list(packet.get("choices") or [])
    if choices and pending_prefix:
        choice = dict(choices[0] or {})
        merged, consumed = _merge_choice_prefix(api, choice, pending_prefix)
        choices[0] = merged
        packet["choices"] = choices
        if consumed:
            pending_prefix = ""
    if "usage" in packet:
        packet["usage"] = _patch_usage(dict(packet.get("usage") or {}), continuation)
    return packet, pending_prefix


def _patch_non_stream_payload(
    api: str,
    payload: Dict[str, Any],
    continuation: _PrefillContinuation,
) -> Dict[str, Any]:
    patched = dict(payload)
    choices = list(patched.get("choices") or [])
    if choices and continuation.active:
        choice = dict(choices[0] or {})
        choice, _ = _merge_choice_prefix(api, choice, continuation.text)
        choices[0] = choice
        patched["choices"] = choices
    if "usage" in patched:
        patched["usage"] = _patch_usage(dict(patched.get("usage") or {}), continuation)
    return patched


def _merge_choice_prefix(api: str, choice: Dict[str, Any], prefix_text: str) -> tuple[Dict[str, Any], bool]:
    if not prefix_text:
        return choice, False
    merged = dict(choice)
    if api.endswith("/chat/completions"):
        delta = merged.get("delta")
        if isinstance(delta, dict):
            delta = dict(delta)
            content = delta.get("content")
            if content is not None:
                delta["content"] = _prepend_stream_content(content, prefix_text)
                merged["delta"] = delta
                return merged, True
            if merged.get("finish_reason") is not None:
                delta["content"] = prefix_text
                merged["delta"] = delta
                return merged, True
        message = merged.get("message")
        if isinstance(message, dict):
            message = dict(message)
            content = message.get("content")
            message["content"] = _prepend_message_content(content, prefix_text)
            merged["message"] = message
            return merged, True
        return merged, False
    text = merged.get("text")
    if text is not None:
        merged["text"] = prefix_text + str(text)
        return merged, True
    return merged, False


def _prepend_stream_content(content: Any, prefix_text: str) -> Any:
    if isinstance(content, str):
        return prefix_text + content
    if isinstance(content, list):
        if content:
            first = content[0]
            if isinstance(first, dict) and str(first.get("type", "")).strip().lower() == "text":
                first = dict(first)
                first["text"] = prefix_text + str(first.get("text", ""))
                return [first, *content[1:]]
        return [{"type": "text", "text": prefix_text}, *content]
    return prefix_text


def _prepend_message_content(content: Any, prefix_text: str) -> Any:
    if isinstance(content, str):
        return prefix_text + content
    if isinstance(content, list):
        if content:
            first = content[0]
            if isinstance(first, dict) and str(first.get("type", "")).strip().lower() == "text":
                first = dict(first)
                first["text"] = prefix_text + str(first.get("text", ""))
                return [first, *content[1:]]
        return [{"type": "text", "text": prefix_text}, *content]
    return prefix_text


def _patch_usage(usage: Dict[str, Any], continuation: _PrefillContinuation) -> Dict[str, Any]:
    if not continuation.active:
        return usage
    patched = dict(usage)
    completion_tokens = int(patched.get("completion_tokens", 0) or 0) + continuation.completion_tokens
    if continuation.prompt_tokens is not None:
        prompt_tokens = continuation.prompt_tokens
    else:
        prompt_tokens = int(patched.get("prompt_tokens", 0) or 0)
    patched["prompt_tokens"] = prompt_tokens
    patched["completion_tokens"] = completion_tokens
    patched["total_tokens"] = prompt_tokens + completion_tokens
    return patched


def _extract_choice_text(api: str, choice: Dict[str, Any]) -> str:
    if api.endswith("/chat/completions"):
        return _flatten_message_content(
            (choice.get("message") or {}).get("content")
        )
    return str(choice.get("text") or "")


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = parse_args()
    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
