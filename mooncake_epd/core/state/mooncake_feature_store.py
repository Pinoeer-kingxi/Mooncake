"""Mooncake-backed FeatureBundle object store.

This module is the production E→P boundary for multimodal hidden-state
FeatureHandles.  Encoder workers serialize a validated FeatureBundle into the
real Mooncake Store (HTTP service or Python SDK); Prefill workers resolve the
``mooncake://`` URI and validate the descriptor before injecting hidden states
into vLLM.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import requests
import torch

from .feature_handle import FeatureHandle
from .feature_store import FeatureBundle


class MooncakeFeatureStoreError(RuntimeError):
    """Raised when a FeatureBundle cannot be published/resolved via Mooncake."""


@dataclass(frozen=True)
class MooncakeFeatureBundleStoreConfig:
    store_id: str = "mooncake-mm-store"
    store_url: Optional[str] = None
    config_path: Optional[str] = None
    timeout_s: float = 30.0
    key_prefix: str = "epd-mm-feature"
    cleanup_on_close: bool = False

    @classmethod
    def from_env(cls) -> "MooncakeFeatureBundleStoreConfig":
        timeout = os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_TIMEOUT_S", "30")
        try:
            timeout_s = max(0.1, float(timeout))
        except Exception:
            timeout_s = 30.0
        return cls(
            store_id=(
                os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_ID")
                or os.getenv("MOONCAKE_EPD_MOONCAKE_STORE_ID")
                or "mooncake-mm-store"
            ),
            store_url=(
                _empty_to_none(os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_URL"))
                or _empty_to_none(os.getenv("MOONCAKE_STORE_URL"))
            ),
            config_path=(
                _empty_to_none(os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_STORE_CONFIG"))
                or _empty_to_none(os.getenv("MOONCAKE_CONFIG_PATH"))
            ),
            timeout_s=timeout_s,
            key_prefix=os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_KEY_PREFIX", "epd-mm-feature"),
        )


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _safe_key_component(value: Any) -> str:
    raw = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)[:220]


def parse_mooncake_feature_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(str(uri))
    if parsed.scheme != "mooncake":
        raise ValueError(f"not a mooncake feature uri: {uri!r}")
    store_id = parsed.netloc
    key = parsed.path.lstrip("/")
    if not store_id or not key:
        raise ValueError(f"invalid mooncake feature uri: {uri!r}")
    return unquote(store_id), unquote(key)


def build_mooncake_feature_uri(store_id: str, key: str) -> str:
    return f"mooncake://{quote(str(store_id), safe='')}/{quote(str(key), safe='')}"


class MooncakeFeatureBundleStore:
    """FeatureBundle object store backed by real Mooncake Store APIs.

    Backend selection order:
    1. HTTP Mooncake store service (`MOONCAKE_STORE_URL`, `/api/put` + `/api/get`).
    2. Python `MooncakeDistributedStore` SDK configured by `MOONCAKE_CONFIG_PATH`
       or the standard Mooncake env vars.

    No filesystem fallback is implemented here; callers that want local dev
    transport should use `publish_feature_bundle_to_dir` explicitly.
    """

    def __init__(self, config: Optional[MooncakeFeatureBundleStoreConfig] = None):
        self.config = config or MooncakeFeatureBundleStoreConfig.from_env()
        self._session: Optional[requests.Session] = None
        self._store = None
        self._store_initialized = False

    @property
    def store_id(self) -> str:
        return self.config.store_id

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._store is not None:
            close = getattr(self._store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._store = None
            self._store_initialized = False

    def publish_bundle(
        self,
        bundle: FeatureBundle,
        *,
        checksum: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> FeatureHandle:
        descriptor = bundle.descriptor(checksum=checksum)
        object_key = key or self.make_key(bundle.image_hash, descriptor=descriptor)
        payload = self._serialize_bundle(bundle, descriptor=descriptor)
        self._put_bytes(object_key, payload)
        md = dict(metadata or {})
        md.update(
            {
                "backend": "mooncake_store",
                "store_id": self.store_id,
                "object_key": object_key,
                "published_at_unix": time.time(),
                "payload_bytes": len(payload),
            }
        )
        return FeatureHandle(
            handle_id=f"mooncake-{_safe_key_component(bundle.image_hash)}-{int(time.time() * 1_000_000)}",
            feature_id=str(bundle.image_hash),
            store_id=self.store_id,
            uri=build_mooncake_feature_uri(self.store_id, object_key),
            descriptor=descriptor,
            metadata=md,
        )

    def load_bundle(self, uri_or_key: str) -> FeatureBundle:
        key = str(uri_or_key)
        if key.startswith("mooncake://"):
            store_id, parsed_key = parse_mooncake_feature_uri(key)
            if store_id != self.store_id:
                raise MooncakeFeatureStoreError(
                    f"Mooncake feature store mismatch: uri store={store_id}, configured={self.store_id}"
                )
            key = parsed_key
        payload = self._get_bytes(key)
        return self._deserialize_bundle(payload)

    def make_key(self, feature_id: str, *, descriptor=None) -> str:
        prefix = _safe_key_component(self.config.key_prefix)
        if descriptor is None:
            return f"{prefix}-{_safe_key_component(feature_id)}.ptb64"
        # Object keys must not be based on source image hash alone.  The same
        # image can be encoded by different model/processor versions, with or
        # without Qwen3-VL DeepStack tensors, or with different dtypes.  Several
        # real Mooncake Store deployments also keep the first value for a key
        # long enough that a same-key publish can resolve stale bytes.  Include
        # the descriptor digest in the object key so the control-plane handle is
        # content/schema-addressed while still preserving feature_id as the
        # cross-request reuse key.
        digest_payload = json.dumps(
            descriptor.to_dict(),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(digest_payload).hexdigest()[:16]
        return f"{prefix}-{_safe_key_component(feature_id)}-v2-{digest}.ptb64"

    @staticmethod
    def _serialize_bundle(bundle: FeatureBundle, *, checksum: bool = False, descriptor=None) -> bytes:
        descriptor = descriptor or bundle.descriptor(checksum=checksum)
        buf = io.BytesIO()
        torch.save(
            {
                "version": 1,
                "kind": "mooncake_epd_feature_bundle",
                "bundle": bundle,
                "descriptor": descriptor.to_dict(),
                "written_at": time.time(),
            },
            buf,
        )
        return base64.b64encode(buf.getvalue())

    @staticmethod
    def _deserialize_bundle(payload: bytes) -> FeatureBundle:
        raw = payload.strip()
        try:
            raw = base64.b64decode(raw, validate=True)
        except Exception:
            # Python SDK callers may store raw torch.save bytes in future.
            pass
        loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if isinstance(loaded, FeatureBundle):
            return loaded
        if isinstance(loaded, dict) and isinstance(loaded.get("bundle"), FeatureBundle):
            return loaded["bundle"]
        raise MooncakeFeatureStoreError("Mooncake object does not contain a FeatureBundle")

    def _put_bytes(self, key: str, payload: bytes) -> None:
        if self.config.store_url:
            session = self._http_session()
            resp = session.put(
                f"{self.config.store_url.rstrip('/')}/api/put",
                json={"key": key, "value": payload.decode("ascii")},
                timeout=self.config.timeout_s,
            )
            resp.raise_for_status()
            return
        store = self._python_store()
        rc = store.put(key, payload)
        if rc != 0:
            raise MooncakeFeatureStoreError(f"MooncakeDistributedStore.put failed: rc={rc}, key={key}")

    def _get_bytes(self, key: str) -> bytes:
        if self.config.store_url:
            session = self._http_session()
            resp = session.get(
                f"{self.config.store_url.rstrip('/')}/api/get/{quote(key, safe='')}",
                timeout=self.config.timeout_s,
            )
            resp.raise_for_status()
            if not resp.content:
                raise MooncakeFeatureStoreError(f"Mooncake HTTP store returned empty object: {key}")
            return resp.content
        store = self._python_store()
        payload = store.get(key)
        if not payload:
            raise MooncakeFeatureStoreError(f"MooncakeDistributedStore missing object: {key}")
        return bytes(payload)

    def _http_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.trust_env = False
        return self._session

    def _python_store(self):
        if self._store_initialized and self._store is not None:
            return self._store
        try:
            from mooncake.store import MooncakeDistributedStore
        except Exception as exc:
            raise MooncakeFeatureStoreError(
                "Mooncake Python SDK is unavailable and no MOONCAKE_STORE_URL was configured"
            ) from exc
        store = MooncakeDistributedStore()
        config = self._load_store_config()
        if config is None:
            raise MooncakeFeatureStoreError(
                "Mooncake store config is missing; set MOONCAKE_STORE_URL or MOONCAKE_CONFIG_PATH/standard Mooncake env vars"
            )
        setup = getattr(store, "setup")
        try:
            rc = setup(config)
        except TypeError:
            rc = setup(
                config["local_hostname"],
                config["metadata_server"],
                int(config["global_segment_size"]),
                int(config["local_buffer_size"]),
                config["protocol"],
                config.get("device_name", ""),
                config["master_server_address"],
            )
        if rc != 0:
            raise MooncakeFeatureStoreError(f"MooncakeDistributedStore.setup failed: rc={rc}")
        self._store = store
        self._store_initialized = True
        return store

    def _load_store_config(self) -> Optional[Dict[str, Any]]:
        path = self.config.config_path
        if path and Path(path).expanduser().exists():
            return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        metadata_server = os.getenv("MOONCAKE_TE_META_DATA_SERVER")
        master_server = os.getenv("MOONCAKE_MASTER")
        if not metadata_server or not master_server:
            return None
        return {
            "local_hostname": os.getenv("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            "metadata_server": metadata_server,
            "global_segment_size": int(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(16 * 1024 * 1024))),
            "local_buffer_size": int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(16 * 1024 * 1024))),
            "protocol": os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            "device_name": os.getenv("MOONCAKE_DEVICE_NAME", ""),
            "master_server_address": master_server,
        }
