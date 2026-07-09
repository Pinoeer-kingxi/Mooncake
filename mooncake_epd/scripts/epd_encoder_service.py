#!/usr/bin/env python3
"""Online E-stage service for Mooncake EPD FeatureHandle generation.

The service is intentionally thin: it owns the real vision encoder process,
publishes hidden-state FeatureBundles, and returns lightweight FeatureHandle
control payloads for the vLLM prefill hot path.  It can publish either to the
local file transport used by same-node development or to a real Mooncake Store
(``--publish-backend mooncake``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from mooncake_epd.core.epd_workers import EncoderWorker  # noqa: E402
from mooncake_epd.core.omni_encoder_worker import Qwen25OmniImageEncoderWorker  # noqa: E402
from mooncake_epd.core.state import (  # noqa: E402
    FeatureHandle,
    MooncakeFeatureBundleStore,
    MooncakeFeatureBundleStoreConfig,
    publish_feature_bundle_to_dir,
)
from mooncake_epd.core.transfer import TransferEngine  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class EncoderServiceConfig:
    model: str = "/home/songbinbin/Proj/Proj_LWX/Qwen3-VL-8B-Instruct"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    encoder_family: str = "auto"  # auto | qwen3_vl | qwen2_5_omni
    publish_backend: str = "file"  # file | mooncake | direct_engine
    store_dir: str = "/tmp/mooncake_epd_feature_handle_store"
    mooncake_store_url: Optional[str] = None
    mooncake_store_id: str = "mooncake-mm-store"
    mooncake_config: Optional[str] = None
    mooncake_timeout_s: float = 30.0
    mooncake_protocol: str = "tcp"
    mooncake_local_hostname: str = "localhost"
    mooncake_metadata_server: str = "P2PHANDSHAKE"
    mooncake_device_name: str = ""
    direct_source_mode: str = "registered_tensor"
    checksum: bool = False
    max_image_bytes: int = 32 * 1024 * 1024
    request_timeout_s: float = 15.0
    enable_omni_hidden_prefix_cache: bool = True
    omni_hidden_prefix_cache_metrics: Optional[str] = None
    omni_allow_partial_prefix: bool = False


class _LazyEncoder:
    def __init__(self, config: EncoderServiceConfig):
        self.config = config
        self._lock = asyncio.Lock()
        self._worker: Optional[Any] = None
        self._processor = None
        self._model = None

    async def worker(self) -> Any:
        if self._worker is not None:
            return self._worker
        async with self._lock:
            if self._worker is None:
                self._worker = await asyncio.to_thread(self._load_sync)
            return self._worker

    def _load_sync(self) -> Any:
        from transformers import AutoConfig, AutoProcessor

        dtype = _torch_dtype(self.config.dtype)
        device = torch.device(self.config.device)
        family = str(self.config.encoder_family or "auto").lower()
        if family == "auto":
            cfg = AutoConfig.from_pretrained(self.config.model, trust_remote_code=True)
            model_type = str(getattr(cfg, "model_type", "") or "").lower()
            family = "qwen2_5_omni" if model_type == "qwen2_5_omni" else "qwen3_vl"
        processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        if family in {"qwen2_5_omni", "qwen25_omni", "omni"}:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration

            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.config.model,
                dtype=dtype,
                device_map={"": device},
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
            )
            model.eval()
            self._processor = processor
            self._model = model
            return Qwen25OmniImageEncoderWorker(
                model,
                processor,
                device=device,
                enable_hidden_prefix_cache=bool(self.config.enable_omni_hidden_prefix_cache),
                allow_partial_prefix_reuse=bool(self.config.omni_allow_partial_prefix),
                cache_metrics_path=self.config.omni_hidden_prefix_cache_metrics,
            )
        if family in {"qwen3_vl", "qwen_vl", "qwen3vl"}:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model,
                dtype=dtype,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
            model.eval()
            self._processor = processor
            self._model = model
            return EncoderWorker(model, processor, device=device)
        raise ValueError(f"unsupported encoder_family: {self.config.encoder_family}")

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError("encoder is not loaded yet")
        return self._processor


def create_app(
    config: Optional[EncoderServiceConfig] = None,
    *,
    encoder: Optional[Any] = None,
    direct_transfer_engine: Optional[TransferEngine] = None,
) -> FastAPI:
    """Create the online encoder FastAPI app.

    ``encoder`` is an optional dependency injection point for local tests and
    process managers that already own a loaded EncoderWorker-compatible object.
    Production code should leave it unset so the real Qwen-VL model is loaded
    lazily on the configured GPU.
    """

    config = config or EncoderServiceConfig()
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        try:
            yield
        finally:
            http = getattr(app.state, "http", None)
            if http is not None:
                await http.aclose()

    app = FastAPI(title="Mooncake EPD Encoder Service", lifespan=_lifespan)
    app.state.config = config
    app.state.lazy_encoder = None if encoder is not None else _LazyEncoder(config)
    app.state.encoder = encoder
    app.state.direct_transfer_engine = direct_transfer_engine
    app.state.pending_direct_bundles = {}
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(config.request_timeout_s, connect=5.0),
        follow_redirects=True,
        trust_env=False,
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": config.model,
            "device": config.device,
            "encoder_family": config.encoder_family,
            "publish_backend": config.publish_backend,
            "omni_hidden_prefix_cache": bool(config.enable_omni_hidden_prefix_cache),
            "loaded": app.state.encoder is not None,
        }

    async def _worker_and_processor() -> Tuple[Any, Any]:
        worker = app.state.encoder
        if worker is None:
            worker = await app.state.lazy_encoder.worker()
            app.state.encoder = worker
        processor = getattr(worker, "processor", None) or getattr(app.state.lazy_encoder, "processor", None)
        if processor is None:
            raise HTTPException(status_code=500, detail="encoder worker has no processor")
        return worker, processor

    async def _encode_records(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
        items = list(_iter_mm_image_items(payload))
        if not items:
            raise HTTPException(status_code=400, detail="no image/image_url items found")
        worker, processor = await _worker_and_processor()

        loaded: List[Tuple[Dict[str, Any], Image.Image, str, str, str]] = []
        for item in items:
            image, content_type, source_digest = await _load_image_item(
                app,
                item,
                max_bytes=int(config.max_image_bytes),
            )
            loaded.append((item, image, content_type, source_digest, _stable_mm_hash(item)))

        records: List[Dict[str, Any]] = []
        encode_ms_total = 0.0
        cache_stats: Dict[str, Any] = {}
        if hasattr(worker, "encode_images"):
            prompt = _prompt_for_processor(payload)
            batch_out = await asyncio.to_thread(
                worker.encode_images,
                [image for _, image, _, _, _ in loaded],
                image_ids=[source_mm_hash for _, _, _, _, source_mm_hash in loaded],
                prompt=prompt,
            )
            encode_ms_total += float(getattr(batch_out, "encode_time_ms", 0.0) or 0.0)
            cache_stats = dict(getattr(batch_out, "cache_stats", {}) or {})
            for index, (enc_out, (_, _, content_type, source_digest, source_mm_hash)) in enumerate(
                zip(batch_out.outputs, loaded)
            ):
                metadata = {
                    "source_mm_hash": source_mm_hash,
                    "source_digest": source_digest,
                    "source_content_type": content_type,
                    "source_index": index,
                    "encoder_service_model": config.model,
                    "encoder_service_device": config.device,
                    "encoder_family": config.encoder_family,
                    "encode_time_ms": enc_out.encode_time_ms,
                    "publish_backend": config.publish_backend,
                    "omni_hidden_prefix_cache": cache_stats,
                }
                records.append({"bundle": enc_out.bundle, "metadata": metadata})
        else:
            prompt = _prompt_for_processor(payload)
            for index, item in enumerate(items):
                _, image, content_type, source_digest, source_mm_hash = loaded[index]
                inputs = await asyncio.to_thread(_processor_inputs, processor, image, prompt)
                if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
                    raise HTTPException(status_code=500, detail="processor did not produce pixel_values/image_grid_thw")

                enc_out = await asyncio.to_thread(
                    worker.encode,
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    image_id=source_mm_hash,
                )
                encode_ms_total += float(enc_out.encode_time_ms)
                metadata = {
                    "source_mm_hash": source_mm_hash,
                    "source_digest": source_digest,
                    "source_content_type": content_type,
                    "source_index": index,
                    "encoder_service_model": config.model,
                    "encoder_service_device": config.device,
                    "encoder_family": config.encoder_family,
                    "encode_time_ms": enc_out.encode_time_ms,
                    "publish_backend": config.publish_backend,
                }
                records.append({"bundle": enc_out.bundle, "metadata": metadata})
        return records, encode_ms_total, cache_stats

    @app.post("/describe")
    async def describe(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Encode once and return descriptors for Prefill direct-buffer allocation.

        For ``publish_backend=direct_engine`` the bundle is held in encoder
        memory under a short-lived ticket until ``/publish_direct`` supplies
        Prefill-owned peer-buffer targets. This avoids the bad double-encode
        control flow while still keeping Prefill as the owner of destination
        tensors.
        """

        started = time.perf_counter()
        records, encode_ms_total, cache_stats = await _encode_records(payload)
        ticket = uuid.uuid4().hex
        app.state.pending_direct_bundles[ticket] = {
            "records": records,
            "created_at": time.monotonic(),
            "request_metadata": dict(payload.get("metadata") or {}),
        }
        descriptors = [record["bundle"].descriptor(checksum=bool(config.checksum)).to_dict() for record in records]
        return {
            "ticket": ticket,
            "descriptors": descriptors,
            "count": len(records),
            "encode_time_ms": encode_ms_total,
            "total_time_ms": (time.perf_counter() - started) * 1000.0,
            "publish_backend": config.publish_backend,
            "encoder_family": config.encoder_family,
            "omni_hidden_prefix_cache": cache_stats,
        }

    @app.post("/publish_direct")
    async def publish_direct(payload: Dict[str, Any]) -> Dict[str, Any]:
        if config.publish_backend != "direct_engine":
            raise HTTPException(status_code=400, detail="/publish_direct requires publish_backend=direct_engine")
        ticket = str(payload.get("ticket") or "")
        pending = app.state.pending_direct_bundles.pop(ticket, None)
        if pending is None:
            raise HTTPException(status_code=404, detail=f"unknown or already consumed direct publish ticket: {ticket}")
        direct_targets = _direct_feature_targets_for_payload(payload)
        records = list(pending.get("records") or [])
        if len(direct_targets) != len(records):
            raise HTTPException(
                status_code=400,
                detail=f"direct target count mismatch: targets={len(direct_targets)} descriptors={len(records)}",
            )
        handles: List[FeatureHandle] = []
        try:
            for index, record in enumerate(records):
                handles.append(
                    await asyncio.to_thread(
                        _publish_bundle,
                        config,
                        record["bundle"],
                        dict(record["metadata"]),
                        direct_targets[index],
                        app.state.direct_transfer_engine,
                    )
                )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"direct publish failed: {exc}") from exc
        return {
            "handles": [handle.as_control_payload() for handle in handles],
            "count": len(handles),
            "publish_backend": config.publish_backend,
            "encoder_family": config.encoder_family,
        }

    @app.post("/encode")
    async def encode(payload: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        records, encode_ms_total, cache_stats = await _encode_records(payload)
        direct_targets = _direct_feature_targets_for_payload(payload)
        handles: List[FeatureHandle] = []
        for index, record in enumerate(records):
            try:
                handle = await asyncio.to_thread(
                    _publish_bundle,
                    config,
                    record["bundle"],
                    dict(record["metadata"]),
                    direct_targets[index] if index < len(direct_targets) else None,
                    app.state.direct_transfer_engine,
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"feature publish failed: {exc}") from exc
            handles.append(handle)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "handles": [handle.as_control_payload() for handle in handles],
            "count": len(handles),
            "encode_time_ms": encode_ms_total,
            "total_time_ms": elapsed_ms,
            "publish_backend": config.publish_backend,
            "encoder_family": config.encoder_family,
            "omni_hidden_prefix_cache": cache_stats,
        }

    return app


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name or "").lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _iter_mm_image_items(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and _image_url_from_item(item):
                        yield item
    prompt = payload.get("prompt")
    if isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and _image_url_from_item(item):
                yield item
    images = payload.get("images")
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict):
                yield image
            elif isinstance(image, str):
                yield {"type": "image_url", "image_url": {"url": image}}


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


async def _load_image_item(app: FastAPI, item: Dict[str, Any], *, max_bytes: int) -> Tuple[Image.Image, str, str]:
    url = _image_url_from_item(item)
    if not url:
        raise HTTPException(status_code=400, detail="image item has no URL")
    payload, content_type = await _load_url_bytes(app, url, max_bytes=max_bytes)
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image bytes: {exc}") from exc
    return image, content_type, hashlib.sha256(payload).hexdigest()


async def _load_url_bytes(app: FastAPI, url: str, *, max_bytes: int) -> Tuple[bytes, str]:
    if url.startswith("data:"):
        return _parse_data_url(url, max_bytes=max_bytes)
    if url.startswith("file://"):
        path = Path(url[7:]).expanduser()
        payload = path.read_bytes()
        if len(payload) > max_bytes:
            raise HTTPException(status_code=413, detail=f"image too large: {len(payload)} > {max_bytes}")
        return payload, _guess_content_type(path)
    if not (url.startswith("http://") or url.startswith("https://")):
        path = Path(url).expanduser()
        if path.exists():
            payload = path.read_bytes()
            if len(payload) > max_bytes:
                raise HTTPException(status_code=413, detail=f"image too large: {len(payload)} > {max_bytes}")
            return payload, _guess_content_type(path)
        raise HTTPException(status_code=400, detail=f"unsupported image URL scheme: {url[:64]}")
    response = await app.state.http.get(url)
    response.raise_for_status()
    payload = response.content
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail=f"image too large: {len(payload)} > {max_bytes}")
    content_type = response.headers.get("content-type", "application/octet-stream").split(";")[0].strip()
    return payload, content_type or "application/octet-stream"


def _parse_data_url(url: str, *, max_bytes: int) -> Tuple[bytes, str]:
    header, sep, data = url.partition(",")
    if sep != "," or not header.startswith("data:"):
        raise HTTPException(status_code=400, detail="invalid data URL")
    meta = header[5:]
    parts = [part for part in meta.split(";") if part]
    content_type = parts[0] if parts and "/" in parts[0] else "application/octet-stream"
    if "base64" in parts:
        payload = base64.b64decode(data, validate=True)
    else:
        from urllib.parse import unquote_to_bytes
        payload = unquote_to_bytes(data)
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail=f"image too large: {len(payload)} > {max_bytes}")
    return payload, content_type


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _prompt_for_processor(payload: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and str(item.get("type", "text")) in {"text", "input_text"}:
                    pieces.append(str(item.get("text") or item.get("content") or ""))
    if not pieces and isinstance(payload.get("prompt"), str):
        pieces.append(str(payload["prompt"]))
    return " ".join(part for part in pieces if part).strip() or "Describe the image."


def _processor_inputs(processor: Any, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def _stable_mm_hash(item: Dict[str, Any]) -> str:
    payload = {k: item.get(k) for k in sorted(item) if k not in {"detail"}}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _direct_feature_targets_for_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = dict(payload.get("metadata") or {})
    raw = (
        metadata.get("mooncake_epd_direct_feature_targets")
        or metadata.get("direct_feature_targets")
        or payload.get("mooncake_epd_direct_feature_targets")
        or payload.get("direct_feature_targets")
        or []
    )
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="direct feature targets must be a list")
    return [dict(item or {}) for item in raw]


def _publish_bundle(
    config: EncoderServiceConfig,
    bundle,
    metadata: Dict[str, Any],
    direct_target: Optional[Dict[str, Any]] = None,
    direct_transfer_engine: Optional[TransferEngine] = None,
) -> FeatureHandle:
    if config.publish_backend == "file":
        return publish_feature_bundle_to_dir(
            bundle,
            config.store_dir,
            checksum=bool(config.checksum),
            metadata=metadata,
        )
    if config.publish_backend == "mooncake":
        store = MooncakeFeatureBundleStore(
            MooncakeFeatureBundleStoreConfig(
                store_id=config.mooncake_store_id,
                store_url=config.mooncake_store_url,
                config_path=config.mooncake_config,
                timeout_s=float(config.mooncake_timeout_s),
            )
        )
        try:
            return store.publish_bundle(
                bundle,
                checksum=bool(config.checksum),
                metadata=metadata,
            )
        finally:
            store.close()
    if config.publish_backend == "direct_engine":
        if not isinstance(direct_target, dict) or not direct_target:
            raise ValueError(
                "publish_backend=direct_engine requires metadata.mooncake_epd_direct_feature_targets"
            )
        remote_session = str(direct_target.get("remote_session") or "")
        remote_pointers = direct_target.get("remote_pointers")
        if not isinstance(remote_pointers, dict):
            raise ValueError("direct feature target requires remote_pointers dict")
        engine = direct_transfer_engine or TransferEngine(
            protocol=config.mooncake_protocol,
            local_hostname=config.mooncake_local_hostname,
            metadata_server=config.mooncake_metadata_server,
            device_name=config.mooncake_device_name,
        )
        plan = engine.build_feature_bundle_peer_buffer_plan(
            bundle,
            remote_session=remote_session,
            remote_pointers={str(k): int(v) for k, v in remote_pointers.items()},
            checksum=bool(config.checksum),
        )
        result = engine.transfer_feature_bundle_peer_buffer_plan(
            bundle,
            plan,
            source_memory_mode=config.direct_source_mode,
        )
        descriptor = bundle.descriptor(checksum=bool(config.checksum))
        md = dict(metadata)
        md.update(
            {
                "backend": "direct_engine",
                "direct_backend": result.backend_label,
                "direct_remote_session": remote_session,
                "direct_tensor_count": result.tensor_count,
                "direct_descriptor_count": result.descriptor_count,
                "direct_bytes": result.nbytes,
                "direct_plan": {
                    "feature_id": plan.feature_id,
                    "targets": [
                        {
                            "name": target.name,
                            "remote_pointer": target.remote_pointer,
                            "nbytes": target.nbytes,
                        }
                        for target in plan.targets
                    ],
                },
            }
        )
        return FeatureHandle(
            handle_id=f"direct-{bundle.image_hash}-{int(time.time() * 1_000_000)}",
            feature_id=str(bundle.image_hash),
            store_id=config.mooncake_store_id,
            uri=f"epd-direct://{config.mooncake_store_id}/{bundle.image_hash}",
            descriptor=descriptor,
            metadata=md,
        )
    raise ValueError(f"unsupported publish backend: {config.publish_backend}")


def parse_args() -> EncoderServiceConfig:
    ap = argparse.ArgumentParser(description="Mooncake EPD online encoder service")
    ap.add_argument("--model", default=os.getenv("MOONCAKE_EPD_ENCODER_MODEL", EncoderServiceConfig.model))
    ap.add_argument("--device", default=os.getenv("MOONCAKE_EPD_ENCODER_DEVICE", EncoderServiceConfig.device))
    ap.add_argument("--dtype", default=os.getenv("MOONCAKE_EPD_ENCODER_DTYPE", EncoderServiceConfig.dtype))
    ap.add_argument(
        "--encoder-family",
        choices=["auto", "qwen3_vl", "qwen2_5_omni"],
        default=os.getenv("MOONCAKE_EPD_ENCODER_FAMILY", "auto"),
    )
    ap.add_argument("--publish-backend", choices=["file", "mooncake", "direct_engine"], default=os.getenv("MOONCAKE_EPD_ENCODER_PUBLISH_BACKEND", "file"))
    ap.add_argument("--mooncake-local-hostname", default=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"))
    ap.add_argument("--mooncake-metadata-server", default=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"))
    ap.add_argument("--mooncake-device-name", default=os.getenv("MOONCAKE_DEVICE_NAME", ""))
    ap.add_argument("--direct-source-mode", choices=["registered_tensor", "managed_buffer"], default=os.getenv("MOONCAKE_EPD_DIRECT_SOURCE_MODE", "registered_tensor"))
    ap.add_argument("--store-dir", default=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_DIR", EncoderServiceConfig.store_dir))
    ap.add_argument("--mooncake-store-url", default=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_URL"))
    ap.add_argument("--mooncake-store-id", default=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_ID", EncoderServiceConfig.mooncake_store_id))
    ap.add_argument("--mooncake-config", default=os.getenv("MOONCAKE_CONFIG_PATH"))
    ap.add_argument("--mooncake-timeout-s", type=float, default=float(os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_TIMEOUT_S", "30")))
    ap.add_argument("--mooncake-protocol", default=os.getenv("MOONCAKE_PROTOCOL", EncoderServiceConfig.mooncake_protocol))
    ap.add_argument("--checksum", action=argparse.BooleanOptionalAction, default=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_CHECKSUM", "0") in {"1", "true", "TRUE"})
    ap.add_argument("--max-image-bytes", type=int, default=int(os.getenv("MOONCAKE_EPD_ENCODER_MAX_IMAGE_BYTES", str(32 * 1024 * 1024))))
    ap.add_argument("--request-timeout-s", type=float, default=float(os.getenv("MOONCAKE_EPD_ENCODER_REQUEST_TIMEOUT_S", "15")))
    ap.add_argument(
        "--enable-omni-hidden-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MOONCAKE_EPD_OMNI_HIDDEN_PREFIX_CACHE", "1").lower()
        not in {"0", "false", "no", "off"},
    )
    ap.add_argument(
        "--omni-hidden-prefix-cache-metrics",
        default=os.getenv("MOONCAKE_EPD_OMNI_HIDDEN_PREFIX_CACHE_METRICS"),
    )
    ap.add_argument(
        "--omni-allow-partial-prefix",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MOONCAKE_EPD_OMNI_HIDDEN_PREFIX_CACHE_ALLOW_PARTIAL", "0").lower()
        in {"1", "true", "yes", "on"},
    )
    args = ap.parse_args()
    return EncoderServiceConfig(**vars(args))


def main() -> None:
    import uvicorn

    config = parse_args()
    host = os.getenv("MOONCAKE_EPD_ENCODER_HOST", "127.0.0.1")
    port = int(os.getenv("MOONCAKE_EPD_ENCODER_PORT", "8300"))
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
