"""vLLM-side FeatureHandle resolver for Mooncake E→P hidden-state reuse.

This module is deliberately independent from vLLM imports so it can be unit
-tested in the repo and imported from ``sitecustomize`` inside real vLLM worker
processes.  It resolves lightweight control-plane FeatureHandle payloads into
validated Qwen-VL-compatible hidden-state tensors (``image_embeds`` plus optional
``image_grid_thw`` / deep-stack intermediates).

The runtime contract is fail-open by default: if a feature handle cannot be
resolved or validated, callers receive ``None`` and vLLM should execute its
normal vision encoder path.  Set ``MOONCAKE_EPD_VLLM_FEATURE_HANDLE_STRICT=1``
to make resolution errors explicit during controlled validation.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch

from .feature_handle import FeatureHandle, FeatureHandleError, FeatureHandleRegistry
from .feature_store import FeatureBundle, FeatureBundleDescriptor, TensorSpec
from .mooncake_feature_store import (
    MooncakeFeatureBundleStore,
    MooncakeFeatureBundleStoreConfig,
    MooncakeFeatureStoreError,
    parse_mooncake_feature_uri,
)
from .direct_feature_buffer import (
    get_direct_feature_buffer_registry,
    iter_direct_feature_buffer_registries,
)
from .vllm_mm_hidden_cache import (
    get_current_mm_hidden_cache_keys,
    record_vllm_precomputed_image_embeds_hit,
    trace_vllm_mm_hidden_event,
)
from ..transfer.engine import TransferEngine
from ..strict_mode import strict_no_fallback_enabled

_CURRENT_KV_TRANSFER_PARAMS: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("mooncake_epd_vllm_kv_transfer_params", default=None)
)

_REGISTRIES: Dict[str, FeatureHandleRegistry] = {}
_REGISTRY_LOCK = threading.RLock()
_DIRECT_READ_ENGINE: Optional[TransferEngine] = None
_DIRECT_READ_ENGINE_LOCK = threading.RLock()


@dataclass(frozen=True)
class ResolvedFeatureHandles:
    """Tensor payload that can be injected into a vLLM multimodal model call."""

    image_embeds: torch.Tensor
    image_grid_thw: Optional[torch.Tensor] = None
    deepstack_image_embeds: Tuple[Tuple[int, torch.Tensor], ...] = field(default_factory=tuple)
    handles: Tuple[FeatureHandle, ...] = field(default_factory=tuple)
    source: str = "unknown"

    @property
    def count(self) -> int:
        return len(self.handles)

    def as_model_kwargs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"image_embeds": self.image_embeds}
        if self.image_grid_thw is not None:
            out["image_grid_thw"] = self.image_grid_thw
        if self.deepstack_image_embeds:
            # vLLM/Qwen variants disagree on naming.  The patch layer only
            # forwards keys that the target callable accepts.
            out["deepstack_image_embeds"] = [tensor for _, tensor in self.deepstack_image_embeds]
            out["deepstack_image_hidden_states"] = [tensor for _, tensor in self.deepstack_image_embeds]
        return out


@dataclass(frozen=True)
class FeatureHandleProviderConfig:
    worker_id: str = "prefill"
    device: str = "cpu"
    timeout_s: float = 30.0
    strict: bool = False
    store_dirs: Tuple[Path, ...] = field(default_factory=tuple)
    expected_model_fingerprint: Optional[str] = None
    expected_processor_fingerprint: Optional[str] = None
    require_checksum: bool = False
    mooncake_store_url: Optional[str] = None
    mooncake_store_id: str = "mooncake-mm-store"
    mooncake_config_path: Optional[str] = None
    allow_file_fallback_for_mooncake_uri: bool = False

    @classmethod
    def from_env(cls) -> "FeatureHandleProviderConfig":
        raw_dirs = (
            os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_DIRS")
            or os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_DIR")
            or ""
        )
        dirs = tuple(
            Path(item).expanduser()
            for item in raw_dirs.split(os.pathsep)
            if item.strip()
        )
        return cls(
            worker_id=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_WORKER_ID", "prefill"),
            device=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
            timeout_s=_env_float("MOONCAKE_EPD_FEATURE_HANDLE_TIMEOUT_S", 30.0, minimum=0.0),
            strict=(
                _env_bool("MOONCAKE_EPD_VLLM_FEATURE_HANDLE_STRICT", False)
                or strict_no_fallback_enabled()
            ),
            store_dirs=dirs,
            expected_model_fingerprint=_empty_to_none(os.getenv("MOONCAKE_EPD_MODEL_FINGERPRINT")),
            expected_processor_fingerprint=_empty_to_none(os.getenv("MOONCAKE_EPD_PROCESSOR_FINGERPRINT")),
            require_checksum=_env_bool("MOONCAKE_EPD_FEATURE_HANDLE_REQUIRE_CHECKSUM", False),
            mooncake_store_url=(
                _empty_to_none(os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_URL"))
                or _empty_to_none(os.getenv("MOONCAKE_STORE_URL"))
            ),
            mooncake_store_id=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_ID", "mooncake-mm-store"),
            mooncake_config_path=(
                _empty_to_none(os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_CONFIG"))
                or _empty_to_none(os.getenv("MOONCAKE_CONFIG_PATH"))
            ),
            allow_file_fallback_for_mooncake_uri=(
                False
                if strict_no_fallback_enabled()
                else _env_bool("MOONCAKE_EPD_FEATURE_HANDLE_ALLOW_FILE_FALLBACK", False)
            ),
        )


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = float(default)
    return max(float(minimum), value)


def _dtype_from_spec(spec: TensorSpec) -> torch.dtype:
    dtype = str(spec.dtype).replace("torch.", "")
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if dtype in aliases:
        return aliases[dtype]
    candidate = getattr(torch, dtype, None)
    if isinstance(candidate, torch.dtype):
        return candidate
    raise FeatureHandleError(f"unsupported tensor dtype in FeatureHandle descriptor: {spec.dtype}")


def _tensor_from_direct_bytes(raw: bytes, spec: TensorSpec, *, device: torch.device | str) -> torch.Tensor:
    if len(raw) != int(spec.nbytes):
        raise FeatureHandleError(
            f"direct peer-buffer read size mismatch: got={len(raw)} expected={spec.nbytes}"
        )
    dtype = _dtype_from_spec(spec)
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(spec.shape))
    return tensor.to(device=device, dtype=dtype, non_blocking=False)


def _feature_handle_direct_local_hostname() -> str:
    explicit = _empty_to_none(os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_DIRECT_LOCAL_HOSTNAME"))
    if explicit:
        return explicit
    host = _empty_to_none(os.getenv("MOONCAKE_LOCAL_HOSTNAME")) or "127.0.0.1"
    if ":" in host:
        host = host.rsplit(":", 1)[0]
    port = int(
        os.getenv(
            "MOONCAKE_EPD_FEATURE_HANDLE_DIRECT_PORT",
            str(20000 + (os.getpid() % 20000)),
        )
    )
    return f"{host}:{port}"


def _get_direct_read_engine() -> TransferEngine:
    global _DIRECT_READ_ENGINE
    with _DIRECT_READ_ENGINE_LOCK:
        if _DIRECT_READ_ENGINE is None:
            _DIRECT_READ_ENGINE = TransferEngine(
                protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
                local_hostname=_feature_handle_direct_local_hostname(),
                metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
                device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
            )
        return _DIRECT_READ_ENGINE


def register_feature_handle_registry(registry: FeatureHandleRegistry) -> None:
    """Expose an in-process MMStore registry to vLLM FeatureHandle resolution.

    Real serving usually runs Encoder and Prefill in separate processes, so the
    file/Mooncake URI path is the production boundary.  This registry hook is
    still useful for same-process integration tests and colocated deployments.
    """

    with _REGISTRY_LOCK:
        _REGISTRIES[str(registry.store_id)] = registry


def unregister_feature_handle_registry(store_id: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRIES.pop(str(store_id), None)


@contextlib.contextmanager
def use_kv_transfer_params(params: Optional[Mapping[str, Any]]) -> Iterator[None]:
    token = _CURRENT_KV_TRANSFER_PARAMS.set(dict(params or {}) if params else None)
    try:
        yield
    finally:
        _CURRENT_KV_TRANSFER_PARAMS.reset(token)


def get_current_kv_transfer_params() -> Optional[Dict[str, Any]]:
    current = _CURRENT_KV_TRANSFER_PARAMS.get()
    return dict(current) if current else None


def extract_feature_handle_payloads(*sources: Any) -> List[Dict[str, Any]]:
    """Find ``mm_feature_handles`` in common vLLM/proxy metadata containers."""

    for source in sources:
        found = _extract_from_source(source)
        if found:
            return found
    current = get_current_kv_transfer_params()
    found = _extract_from_source(current)
    return found or []


def _extract_from_source(source: Any) -> List[Dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, FeatureHandle):
        return [source.as_control_payload()]
    if isinstance(source, Mapping):
        for key in (
            "mm_feature_handles",
            "mooncake_epd_feature_handles",
            "feature_handles",
        ):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [dict(item.as_control_payload() if isinstance(item, FeatureHandle) else item) for item in value]
        for nested_key in ("kv_transfer_params", "metadata", "extra_args"):
            nested = source.get(nested_key)
            found = _extract_from_source(nested)
            if found:
                return found
    if hasattr(source, "kv_transfer_params"):
        found = _extract_from_source(getattr(source, "kv_transfer_params", None))
        if found:
            return found
    if hasattr(source, "metadata"):
        return _extract_from_source(getattr(source, "metadata", None))
    return []


def publish_feature_bundle_to_dir(
    bundle: FeatureBundle,
    store_dir: str | os.PathLike[str],
    *,
    checksum: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureHandle:
    """Persist a FeatureBundle and return a JSON-safe FeatureHandle.

    This is a real local multi-process transport path for development and
    single-node serving: Encoder writes tensor payloads atomically, Prefill loads
    them by the handle URI and validates the descriptor before skipping vision.
    """

    store = Path(store_dir).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    feature_id = _safe_feature_id(bundle.image_hash)
    descriptor = bundle.descriptor(checksum=checksum)
    payload = {
        "version": 1,
        "bundle": bundle,
        "descriptor": descriptor.to_dict(),
        "written_at": time.time(),
    }
    path = store / f"{feature_id}.pt"
    tmp = store / f".{feature_id}.{os.getpid()}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return FeatureHandle(
        handle_id=f"file-{feature_id}-{int(time.time() * 1_000_000)}",
        feature_id=str(bundle.image_hash),
        store_id=f"file:{store}",
        uri=path.as_uri(),
        descriptor=descriptor,
        metadata=dict(metadata or {}),
    )


class FeatureHandleProvider:
    def __init__(self, config: Optional[FeatureHandleProviderConfig] = None):
        self.config = config or FeatureHandleProviderConfig.from_env()

    def resolve_from_sources(
        self,
        *sources: Any,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[ResolvedFeatureHandles]:
        payloads = extract_feature_handle_payloads(*sources)
        if not payloads:
            return None
        try:
            handles = tuple(FeatureHandle.from_control_payload(dict(item)) for item in payloads)
            bundles = [self._resolve_one(handle) for handle in handles]
            return self._merge(handles, bundles, device=device, dtype=dtype, source="feature_handle")
        except Exception as exc:
            trace_vllm_mm_hidden_event(
                "feature_handle_resolve_failed",
                error=f"{type(exc).__name__}: {exc}",
                strict=self.config.strict,
            )
            if self.config.strict:
                raise
            return None

    def _resolve_one(self, handle: FeatureHandle) -> FeatureBundle:
        with _REGISTRY_LOCK:
            registry = _REGISTRIES.get(handle.store_id)
        if registry is not None:
            prefetch = registry.prefetch(
                handle,
                target_worker_id=self.config.worker_id,
                target_device=torch.device(self.config.device),
            )
            return registry.wait_and_validate(
                handle,
                prefetch,
                timeout=self.config.timeout_s,
                expected_model_fingerprint=self.config.expected_model_fingerprint,
                expected_processor_fingerprint=self.config.expected_processor_fingerprint,
                require_checksum=self.config.require_checksum,
            )

        bundle = self._load_from_uri_or_dirs(handle)
        handle.descriptor.validate_bundle(
            bundle,
            expected_model_fingerprint=self.config.expected_model_fingerprint,
            expected_processor_fingerprint=self.config.expected_processor_fingerprint,
            require_checksum=self.config.require_checksum,
        )
        return bundle

    def _load_from_uri_or_dirs(self, handle: FeatureHandle) -> FeatureBundle:
        candidates: List[Path] = []
        uri = str(handle.uri or "")
        mooncake_error: Optional[BaseException] = None
        if uri.startswith("epd-direct://"):
            registry = get_direct_feature_buffer_registry(self.config.worker_id)
            if registry is not None:
                return registry.resolve_handle(handle)
            for candidate in iter_direct_feature_buffer_registries():
                if candidate.get(handle.feature_id) is not None:
                    return candidate.resolve_handle(handle)
            return self._load_epd_direct_remote_bundle(handle)
        if uri.startswith("mooncake://"):
            try:
                store_id, _ = parse_mooncake_feature_uri(uri)
                store = MooncakeFeatureBundleStore(
                    MooncakeFeatureBundleStoreConfig(
                        store_id=store_id or self.config.mooncake_store_id,
                        store_url=self.config.mooncake_store_url,
                        config_path=self.config.mooncake_config_path,
                        timeout_s=self.config.timeout_s,
                    )
                )
                try:
                    return store.load_bundle(uri)
                finally:
                    store.close()
            except Exception as exc:
                mooncake_error = exc
                trace_vllm_mm_hidden_event(
                    "feature_handle_mooncake_resolve_failed",
                    handle_id=handle.handle_id,
                    uri=uri,
                    error=f"{type(exc).__name__}: {exc}",
                    allow_file_fallback=self.config.allow_file_fallback_for_mooncake_uri,
                )
                if not self.config.allow_file_fallback_for_mooncake_uri:
                    raise FeatureHandleError(
                        f"Mooncake feature handle cannot be resolved via store: {uri}; "
                        f"set MOONCAKE_STORE_URL/MOONCAKE_CONFIG_PATH or enable explicit file fallback"
                    ) from exc
        if uri.startswith("file://"):
            candidates.append(Path(uri[7:]))
        elif uri and not uri.startswith(("mmstore://", "mooncake://")):
            candidates.append(Path(uri).expanduser())
        for store in self.config.store_dirs:
            candidates.append(store / f"{_safe_feature_id(handle.feature_id)}.pt")
            candidates.append(store / f"{_safe_feature_id(handle.descriptor.feature_id)}.pt")
        # Metadata can carry a concrete path without changing the public handle
        # schema.  This is useful when the control plane keeps mmstore:// URIs.
        for key in ("feature_path", "bundle_path", "path"):
            value = handle.metadata.get(key)
            if value:
                candidates.insert(0, Path(str(value)).expanduser())

        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.exists():
                continue
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, FeatureBundle):
                return loaded
            if isinstance(loaded, Mapping):
                bundle = loaded.get("bundle")
                if isinstance(bundle, FeatureBundle):
                    return bundle
            raise FeatureHandleError(f"invalid feature bundle file: {path}")
        extra = f"; mooncake_error={mooncake_error!r}" if mooncake_error is not None else ""
        raise FeatureHandleError(
            f"feature handle {handle.handle_id or handle.feature_id} cannot be resolved; "
            f"uri={handle.uri!r} store_dirs={[str(p) for p in self.config.store_dirs]}{extra}"
        )

    def _load_epd_direct_remote_bundle(self, handle: FeatureHandle) -> FeatureBundle:
        """Materialize an ``epd-direct://`` handle from remote peer buffers.

        In real vLLM serving the direct allocation HTTP route is installed in
        the API-server process, while multimodal encoder execution happens in
        the EngineCore worker process.  The in-process registry is therefore
        only an optimization/test path.  Production EngineCore consumption must
        use the Mooncake direct engine to read API-owned peer buffers described
        by the handle metadata.
        """

        descriptor: FeatureBundleDescriptor = handle.descriptor
        metadata = dict(handle.metadata or {})
        remote_session = str(metadata.get("direct_remote_session") or "").strip()
        plan = dict(metadata.get("direct_plan") or {})
        targets = list(plan.get("targets") or [])
        if not remote_session:
            raise FeatureHandleError("epd-direct handle missing direct_remote_session")
        if not targets:
            raise FeatureHandleError("epd-direct handle missing direct_plan.targets")

        by_name: Dict[str, Dict[str, Any]] = {}
        for item in targets:
            if isinstance(item, Mapping) and item.get("name") is not None:
                by_name[str(item.get("name"))] = dict(item)

        specs: List[Tuple[str, TensorSpec, Optional[int]]] = [
            ("last_hidden", descriptor.last_hidden, None),
        ]
        if descriptor.grid_thw is not None:
            specs.append(("grid_thw", descriptor.grid_thw, None))
        for ordinal, (layer, spec) in enumerate(descriptor.intermediates):
            specs.append((f"intermediate:{int(layer)}:{ordinal}", spec, int(layer)))

        remote_pointers: List[int] = []
        lengths: List[int] = []
        ordered: List[Tuple[str, TensorSpec, Optional[int]]] = []
        for name, spec, layer in specs:
            target = by_name.get(name)
            if target is None:
                raise FeatureHandleError(f"epd-direct plan missing target {name}")
            nbytes = int(spec.nbytes)
            target_nbytes = int(target.get("nbytes", -1))
            if target_nbytes < nbytes:
                raise FeatureHandleError(
                    f"epd-direct target undersized for {name}: target={target_nbytes} required={nbytes}"
                )
            remote_pointers.append(int(target.get("remote_pointer")))
            lengths.append(nbytes)
            ordered.append((name, spec, layer))

        try:
            raw_payloads = _get_direct_read_engine().read_remote_peer_buffers(
                remote_session=remote_session,
                remote_pointers=remote_pointers,
                lengths=lengths,
            )
        except Exception as exc:
            raise FeatureHandleError(
                f"epd-direct FeatureHandle remote peer-buffer materialization failed: {exc}"
            ) from exc
        tensors: Dict[str, torch.Tensor] = {}
        intermediates: List[Tuple[int, torch.Tensor]] = []
        for (name, spec, layer), raw in zip(ordered, raw_payloads):
            tensor = _tensor_from_direct_bytes(raw, spec, device=self.config.device)
            tensors[name] = tensor
            if layer is not None:
                intermediates.append((int(layer), tensor))

        bundle_metadata = dict(descriptor.metadata or {})
        if descriptor.model_fingerprint:
            bundle_metadata["model_fingerprint"] = descriptor.model_fingerprint
        if descriptor.processor_fingerprint:
            bundle_metadata["processor_fingerprint"] = descriptor.processor_fingerprint
        bundle_metadata.update(
            {
                "resolved_from": "epd-direct-peer-buffer",
                "direct_remote_session": remote_session,
            }
        )
        bundle = FeatureBundle(
            image_hash=descriptor.feature_id,
            last_hidden=tensors["last_hidden"],
            intermediates=intermediates,
            grid_thw=tensors.get("grid_thw"),
            metadata=bundle_metadata,
        )
        descriptor.validate_bundle(
            bundle,
            expected_model_fingerprint=self.config.expected_model_fingerprint,
            expected_processor_fingerprint=self.config.expected_processor_fingerprint,
            require_checksum=bool(self.config.require_checksum),
        )
        trace_vllm_mm_hidden_event(
            "feature_handle_direct_remote_resolved",
            feature_id=handle.feature_id,
            remote_session=remote_session,
            tensor_count=len(raw_payloads),
            nbytes=sum(lengths),
        )
        return bundle

    def _merge(
        self,
        handles: Sequence[FeatureHandle],
        bundles: Sequence[FeatureBundle],
        *,
        device: Optional[torch.device | str],
        dtype: Optional[torch.dtype],
        source: str,
    ) -> ResolvedFeatureHandles:
        target_device = torch.device(device or self.config.device)
        main_embeds = torch.cat(
            [b.last_hidden.to(device=target_device, dtype=dtype, non_blocking=True) for b in bundles],
            dim=0,
        )
        grids = [b.grid_thw for b in bundles if b.grid_thw is not None]
        image_grid_thw = None
        if len(grids) == len(bundles):
            image_grid_thw = torch.cat(
                [g.to(device=target_device, non_blocking=True) for g in grids if g is not None],
                dim=0,
            )

        deep_by_layer: Dict[int, List[torch.Tensor]] = {}
        for bundle in bundles:
            for layer, tensor in bundle.intermediates:
                deep_by_layer.setdefault(int(layer), []).append(tensor)
        deepstack: List[Tuple[int, torch.Tensor]] = []
        for layer in sorted(deep_by_layer):
            tensors = deep_by_layer[layer]
            if len(tensors) != len(bundles):
                continue
            deepstack.append(
                (
                    layer,
                    torch.cat(
                        [t.to(device=target_device, dtype=dtype, non_blocking=True) for t in tensors],
                        dim=0,
                    ),
                )
            )

        image_embeds = main_embeds
        if deepstack:
            # vLLM Qwen3-VL's native `image_embeds` branch expects each visual
            # token to carry the main visual hidden state followed by all
            # deepstack/multiscale hidden states on the last dimension.  The
            # downstream `_compute_deepstack_embeds()` splits this packed tensor
            # as [visual_dim, deepstack_num_level * visual_dim].  HF workers keep
            # these tensors separate, so the vLLM provider packs them here.
            try:
                if all(t.shape[:-1] == main_embeds.shape[:-1] for _, t in deepstack):
                    image_embeds = torch.cat([main_embeds] + [t for _, t in deepstack], dim=-1)
            except Exception:
                image_embeds = main_embeds

        stable_keys = [h.metadata.get("source_mm_hash") or h.feature_id for h in handles]
        record_vllm_precomputed_image_embeds_hit(len(handles), stable_keys=stable_keys)
        trace_vllm_mm_hidden_event(
            "feature_handle_resolved",
            count=len(handles),
            source=source,
            image_embeds_shape=list(image_embeds.shape),
            main_embeds_shape=list(main_embeds.shape),
            has_grid=image_grid_thw is not None,
            deepstack_layers=[layer for layer, _ in deepstack],
        )
        return ResolvedFeatureHandles(
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
            deepstack_image_embeds=tuple(deepstack),
            handles=tuple(handles),
            source=source,
        )


def resolve_feature_handles_for_vllm(
    *sources: Any,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    provider: Optional[FeatureHandleProvider] = None,
) -> Optional[ResolvedFeatureHandles]:
    return (provider or FeatureHandleProvider()).resolve_from_sources(*sources, device=device, dtype=dtype)


def maybe_inject_feature_handle_kwargs(
    kwargs: Mapping[str, Any],
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    provider: Optional[FeatureHandleProvider] = None,
) -> Dict[str, Any]:
    """Return a copy of kwargs augmented with resolved image hidden states.

    Existing ``image_embeds`` always wins; this function never overwrites user or
    native vLLM tensors.
    """

    out = dict(kwargs)
    if out.get("image_embeds") is not None:
        try:
            n = int(out["image_embeds"].shape[0]) if hasattr(out["image_embeds"], "shape") else 1
        except Exception:
            n = 1
        record_vllm_precomputed_image_embeds_hit(n, stable_keys=get_current_mm_hidden_cache_keys())
        return out
    resolved = resolve_feature_handles_for_vllm(out, device=device, dtype=dtype, provider=provider)
    if resolved is None:
        return out
    out.update(resolved.as_model_kwargs())
    return out



def _match_handle_for_mm_hash(
    handles: Sequence[FeatureHandle],
    mm_hash: str,
    fallback_index: int,
) -> Optional[FeatureHandle]:
    for handle in handles:
        if handle.feature_id == mm_hash or str(handle.metadata.get("source_mm_hash") or "") == mm_hash:
            return handle
    if 0 <= fallback_index < len(handles):
        return handles[fallback_index]
    return None


def inject_feature_handles_into_vllm_mm_kwargs(
    *,
    mm_hashes: Sequence[str],
    mm_kwargs: Sequence[Tuple[str, Any]],
    mm_lora_refs: Sequence[Tuple[str, Any]],
    requests: Mapping[str, Any],
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    provider: Optional[FeatureHandleProvider] = None,
) -> Tuple[List[str], List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    """Replace vLLM per-image pixel inputs with external ``image_embeds``.

    This is the bridge from request-level ``kv_transfer_params.mm_feature_handles``
    to vLLM's real encoder hot path.  It operates before
    ``group_and_batch_mm_kwargs`` so the downstream Qwen model sees its native
    ``image_embeds`` input and skips ``self.visual(...)``.  Any failure leaves the
    original item untouched unless the provider is strict.
    """

    provider = provider or FeatureHandleProvider()
    out_kwargs: List[Tuple[str, Any]] = list(mm_kwargs)
    per_request_seen: Dict[str, int] = {}
    for idx, item in enumerate(list(mm_kwargs)):
        try:
            modality, original_item = item
            if modality != "image":
                continue
            req_id = str(mm_lora_refs[idx][0]) if idx < len(mm_lora_refs) else ""
            req = requests.get(req_id) if hasattr(requests, "get") else None
            kv_params = getattr(req, "kv_transfer_params", None)
            if not kv_params:
                continue
            raw_payloads = extract_feature_handle_payloads(kv_params)
            if not raw_payloads:
                continue
            handles = [FeatureHandle.from_control_payload(dict(payload)) for payload in raw_payloads]
            req_item_index = per_request_seen.get(req_id, 0)
            per_request_seen[req_id] = req_item_index + 1
            mm_hash = str(mm_hashes[idx]) if idx < len(mm_hashes) else ""
            handle = _match_handle_for_mm_hash(handles, mm_hash, req_item_index)
            if handle is None:
                continue
            resolved = provider.resolve_from_sources(
                {"mm_feature_handles": [handle.as_control_payload()]},
                device=device,
                dtype=dtype,
            )
            if resolved is None:
                continue
            converted = _build_vllm_image_embedding_item(original_item, resolved)
            if converted is not None:
                out_kwargs[idx] = (modality, converted)
                trace_vllm_mm_hidden_event(
                    "feature_handle_injected_mm_item",
                    req_id=req_id,
                    mm_hash=mm_hash,
                    handle_id=handle.handle_id,
                )
        except Exception as exc:
            trace_vllm_mm_hidden_event(
                "feature_handle_inject_mm_item_failed",
                index=idx,
                error=f"{type(exc).__name__}: {exc}",
                strict=provider.config.strict,
            )
            if provider.config.strict:
                raise
            continue
    return list(mm_hashes), out_kwargs, list(mm_lora_refs)


def _build_vllm_image_embedding_item(original_item: Any, resolved: ResolvedFeatureHandles) -> Optional[Any]:
    try:
        from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem  # type: ignore
    except Exception:
        return None
    if not hasattr(original_item, "items"):
        return None
    original = dict(original_item.items())
    embed_field = None
    if "image_embeds" in original:
        embed_field = original["image_embeds"].field
    elif "pixel_values" in original:
        embed_field = original["pixel_values"].field
    elif original:
        embed_field = next(iter(original.values())).field
    if embed_field is None:
        return None

    new_item: Dict[str, Any] = {}
    new_item["image_embeds"] = MultiModalFieldElem(
        data=resolved.image_embeds.detach(),
        field=embed_field,
    )
    if resolved.image_grid_thw is not None:
        grid_field = original.get("image_grid_thw").field if "image_grid_thw" in original else embed_field
        grid = resolved.image_grid_thw.detach()
        try:
            old_grid = original.get("image_grid_thw").data if "image_grid_thw" in original else None
            if isinstance(old_grid, torch.Tensor) and old_grid.ndim == 1 and grid.ndim == 2 and grid.shape[0] == 1:
                grid = grid[0]
        except Exception:
            pass
        new_item["image_grid_thw"] = MultiModalFieldElem(data=grid, field=grid_field)
    # Preserve non-pixel metadata fields required by some vLLM processors, but
    # drop raw pixel data so Qwen takes the native image_embeds branch.
    for key, elem in original.items():
        if key in {"pixel_values", "image_embeds", "image_grid_thw"}:
            continue
        new_item[key] = elem
    return MultiModalKwargsItem(new_item)

def _safe_feature_id(value: Any) -> str:
    raw = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)[:240]
