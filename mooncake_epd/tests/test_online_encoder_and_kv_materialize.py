from __future__ import annotations

import base64
import io

import httpx
import pytest
import torch
from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image

from mooncake_epd.core.state import (
    FeatureBundle,
    FeatureHandle,
    FeatureHandleError,
    FeatureHandleProvider,
    FeatureHandleProviderConfig,
    MooncakeKVStateStore,
    MooncakeRemoteKVMaterializer,
    PagedKVManager,
)
from mooncake_epd.core.transfer import TransferEngine
from mooncake_epd.scripts.epd_encoder_service import EncoderServiceConfig, create_app as create_encoder_app
from mooncake_epd.scripts.vllm_disagg_proxy import ProxyConfig, create_app as create_proxy_app
from mooncake_epd.tests.test_vllm_disagg_proxy_semantics import (
    _build_decode_app,
    _build_prefill_app,
    _client_override,
)


def _png_data_url() -> str:
    image = Image.new("RGB", (4, 4), color=(32, 64, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class _DummyProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return {
            "pixel_values": torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class _DummyEncoder:
    processor = _DummyProcessor()

    def encode(self, *, pixel_values, image_grid_thw, image_id=None):
        class Out:
            pass

        out = Out()
        out.encode_time_ms = 1.25
        out.bundle = FeatureBundle(
            image_hash=image_id or "img",
            last_hidden=torch.ones((4, 8), dtype=torch.float32),
            intermediates=[(1, torch.full((4, 8), 2.0, dtype=torch.float32))],
            grid_thw=image_grid_thw.detach().cpu(),
            metadata={"kind": "dummy_qwen_vl_hidden_state"},
        )
        return out


class _FakeMooncakeEngine:
    def __init__(self):
        self.calls = []

    def batch_transfer_sync_write(self, remote_session, local_ptrs, remote_ptrs, lengths):
        self.calls.append((remote_session, list(local_ptrs), list(remote_ptrs), list(lengths)))
        return 0

    def transfer_sync_write(self, remote_session, local_ptr, remote_ptr, length):
        self.calls.append((remote_session, [int(local_ptr)], [int(remote_ptr)], [int(length)]))
        return 0


def test_online_encoder_service_publishes_file_feature_handle(tmp_path):
    app = create_encoder_app(
        EncoderServiceConfig(
            publish_backend="file",
            store_dir=str(tmp_path / "feature-store"),
            device="cpu",
        ),
        encoder=_DummyEncoder(),
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _png_data_url()}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
    }

    with TestClient(app) as client:
        response = client.post("/encode", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()

    assert payload["count"] == 1
    handle = payload["handles"][0]
    assert handle["uri"].startswith("file:")
    assert handle["metadata"]["source_mm_hash"] == handle["feature_id"]
    provider = FeatureHandleProvider(FeatureHandleProviderConfig(store_dirs=(tmp_path / "feature-store",)))
    resolved = provider.resolve_from_sources({"mm_feature_handles": [handle]}, device="cpu", dtype=torch.float32)
    assert resolved is not None
    assert tuple(resolved.image_embeds.shape) == (4, 16)  # main + deepstack packed
    assert resolved.image_grid_thw.tolist() == [[1, 2, 2]]


def test_online_encoder_service_publishes_direct_engine_feature_handle():
    direct_engine = TransferEngine(protocol="tcp")
    fake = _FakeMooncakeEngine()
    direct_engine.bind_mooncake_backend(fake, initialized=True, owns_backend=False)
    app = create_encoder_app(
        EncoderServiceConfig(
            publish_backend="direct_engine",
            device="cpu",
        ),
        encoder=_DummyEncoder(),
        direct_transfer_engine=direct_engine,
    )
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _png_data_url()}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        "metadata": {
            "mooncake_epd_direct_feature_targets": [
                {
                    "remote_session": "prefill-session",
                    "remote_pointers": {
                        "last_hidden": 10000,
                        "last_hidden:nbytes": 4 * 8 * 4,
                        "grid_thw": 20000,
                        "grid_thw:nbytes": 1 * 3 * 8,
                        "intermediate:1:0": 30000,
                        "intermediate:1:0:nbytes": 4 * 8 * 4,
                    },
                }
            ]
        },
    }

    with TestClient(app) as client:
        response = client.post("/encode", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()

    handle = payload["handles"][0]
    assert handle["uri"].startswith("epd-direct://")
    assert handle["metadata"]["backend"] == "direct_engine"
    assert handle["metadata"]["direct_backend"] == "feature_peer_buffer_direct"
    assert handle["metadata"]["direct_tensor_count"] == 3
    assert handle["metadata"]["direct_bytes"] == 280
    assert fake.calls and fake.calls[0][0] == "prefill-session"

    provider = FeatureHandleProvider(FeatureHandleProviderConfig(device="cpu", strict=True))
    with pytest.raises(FeatureHandleError, match="epd-direct FeatureHandle"):
        provider.resolve_from_sources({"mm_feature_handles": [handle]}, device="cpu", dtype=torch.float32)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_proxy_feature_handle_mode_calls_online_encoder_when_handles_absent(tmp_path):
    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)
    encoder_app = create_encoder_app(
        EncoderServiceConfig(
            publish_backend="file",
            store_dir=str(tmp_path / "feature-store"),
            device="cpu",
        ),
        encoder=_DummyEncoder(),
    )
    proxy_app = create_proxy_app(
        ProxyConfig(
            mm_prefetch_mode="feature_handle",
            prefill_supports_feature_handles=True,
            encoder_service_url="http://encoder.local",
        ),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
    )

    with TestClient(proxy_app) as client:
        proxy_app.state.encoder_client = httpx.AsyncClient(
            base_url="http://encoder.local",
            transport=httpx.ASGITransport(app=encoder_app),
            timeout=None,
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _png_data_url()}},
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                "metadata": {"workflow_id": "wf-online-encoder"},
            },
        )
        assert response.status_code == 200, response.text
        prefill_kv = record["prefill_generate_body"]["sampling_params"]["extra_args"]["kv_transfer_params"]
        assert prefill_kv["mm_prefetch_policy"] == "feature_handle"
        assert prefill_kv["mm_feature_handles"][0]["uri"].startswith("file:")
        assert prefill_kv["mm_feature_handle_target_worker"] == "prefill-0"


def _pm(node_id: str) -> PagedKVManager:
    return PagedKVManager(
        page_size=4,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        node_id=node_id,
    )


def test_descriptor_shared_state_materializes_for_write_on_target_node():
    src = _pm("prefill-a")
    dst = _pm("decode-b")
    refs = src.allocate_pages(1, filled=4)
    key = torch.arange(16, dtype=torch.float32).reshape(2, 1, 4, 2)
    val = key + 100
    src.write_page_slots(refs[0], key, val)

    store = MooncakeKVStateStore(
        src,
        node_id="prefill-a",
        page_managers_by_node={"decode-b": dst},
        remote_materializer=MooncakeRemoteKVMaterializer(
            src,
            dst,
            transfer_engine=TransferEngine(protocol="local"),
        ),
        allow_remote_descriptor_sharing=True,
    )
    store.register_state(refs, workflow_id="wf", state_id="parent")
    child = store.clone_state("parent", child_state_id="child", target_node_id="decode-b", share_remote_descriptor=True)
    assert src.refcount(refs[0].global_block_id) == 2
    assert store.resolve_remote_refs("child", target_node_id="decode-b")[0].physical_id == refs[0].physical_id
    with pytest.raises(RuntimeError, match="materialized before write"):
        store.resolve_remote_refs("child", target_node_id="decode-b", for_write=True)

    materialized = store.materialize_for_write("child", target_node_id="decode-b")
    assert materialized.owner_node_id == "decode-b"
    assert not materialized.metadata.get("remote_descriptor_shared", False)
    assert src.refcount(refs[0].global_block_id) == 1
    child_refs = store.resolve_remote_refs("child", target_node_id="decode-b", for_write=True)
    assert child_refs[0].physical_node_id == "decode-b"
    moved_key, moved_val = dst.get_page_slice(child_refs[0])
    assert torch.equal(moved_key, key)
    assert torch.equal(moved_val, val)
    assert store.release_state("child") == 1
    assert store.release_state("parent") == 1

class _CopyingMooncakeEngine:
    def __init__(self):
        self.calls = []

    def batch_transfer_sync_write(self, remote_session, local_ptrs, remote_ptrs, lengths):
        import ctypes

        self.calls.append((remote_session, list(local_ptrs), list(remote_ptrs), list(lengths)))
        for src, dst, n in zip(local_ptrs, remote_ptrs, lengths):
            ctypes.memmove(int(dst), int(src), int(n))
        return 0

    def transfer_sync_write(self, remote_session, local_ptr, remote_ptr, length):
        import ctypes

        self.calls.append((remote_session, [int(local_ptr)], [int(remote_ptr)], [int(length)]))
        ctypes.memmove(int(remote_ptr), int(local_ptr), int(length))
        return 0


def test_direct_feature_buffer_service_allocates_and_releases():
    from mooncake_epd.core.state import DirectFeatureBufferRegistry
    from mooncake_epd.scripts.direct_feature_buffer_service import create_app as create_direct_app

    bundle = FeatureBundle(
        image_hash="img-direct",
        last_hidden=torch.ones((2, 3), dtype=torch.float32),
        intermediates=[(7, torch.ones((2, 3), dtype=torch.float32))],
        grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.long),
    )
    registry = DirectFeatureBufferRegistry(
        worker_id="prefill-0",
        device="cpu",
        remote_session="prefill-session",
        register_memory=False,
    )
    app = create_direct_app(registry=registry)
    with TestClient(app) as client:
        response = client.post("/allocate", json={"descriptors": [bundle.descriptor().to_dict()]})
        assert response.status_code == 200, response.text
        target = response.json()["targets"][0]
        assert target["remote_session"] == "prefill-session"
        assert set(target["remote_pointers"]) >= {
            "last_hidden",
            "last_hidden:nbytes",
            "grid_thw",
            "grid_thw:nbytes",
            "intermediate:7:0",
            "intermediate:7:0:nbytes",
        }
        assert client.get("/stats").json()["allocations"] == 1
        released = client.post("/release", json={"feature_ids": ["img-direct"]})
        assert released.status_code == 200
        assert released.json()["stats"]["allocations"] == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_proxy_online_encoder_direct_engine_handshake_materializes_prefill_buffers():
    from mooncake_epd.core.state import DirectFeatureBufferRegistry
    from mooncake_epd.scripts.direct_feature_buffer_service import create_app as create_direct_app

    record: dict = {}
    prefill_app = _build_prefill_app(record)
    decode_app = _build_decode_app(record)

    direct_engine = TransferEngine(protocol="tcp")
    fake_engine = _CopyingMooncakeEngine()
    direct_engine.bind_mooncake_backend(fake_engine, initialized=True, owns_backend=False)
    encoder_app = create_encoder_app(
        EncoderServiceConfig(
            publish_backend="direct_engine",
            device="cpu",
        ),
        encoder=_DummyEncoder(),
        direct_transfer_engine=direct_engine,
    )
    registry = DirectFeatureBufferRegistry(
        worker_id="prefill-0",
        device="cpu",
        remote_session="prefill-session",
        register_memory=False,
    )
    direct_app = create_direct_app(registry=registry)
    proxy_app = create_proxy_app(
        ProxyConfig(
            mm_prefetch_mode="feature_handle",
            prefill_supports_feature_handles=True,
            encoder_service_url="http://encoder.local",
            prefill_direct_buffer_service_url="http://prefill-direct.local",
            release_direct_feature_buffers_after_prefill=False,
        ),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
    )

    with TestClient(proxy_app) as client:
        proxy_app.state.encoder_client = httpx.AsyncClient(
            base_url="http://encoder.local",
            transport=httpx.ASGITransport(app=encoder_app),
            timeout=None,
        )
        proxy_app.state.prefill_direct_buffer_client = httpx.AsyncClient(
            base_url="http://prefill-direct.local",
            transport=httpx.ASGITransport(app=direct_app),
            timeout=None,
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _png_data_url()}},
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                "metadata": {"workflow_id": "wf-online-direct"},
            },
        )
        assert response.status_code == 200, response.text

    prefill_kv = record["prefill_generate_body"]["sampling_params"]["extra_args"]["kv_transfer_params"]
    handle_payload = prefill_kv["mm_feature_handles"][0]
    assert prefill_kv["mm_prefetch_policy"] == "feature_handle"
    assert handle_payload["uri"].startswith("epd-direct://")
    assert handle_payload["metadata"]["backend"] == "direct_engine"
    assert handle_payload["metadata"]["direct_backend"] == "feature_peer_buffer_direct"
    assert fake_engine.calls and fake_engine.calls[0][0] == "prefill-session"

    provider = FeatureHandleProvider(
        FeatureHandleProviderConfig(worker_id="prefill-0", device="cpu", strict=True)
    )
    resolved = provider.resolve_from_sources({"mm_feature_handles": [handle_payload]}, device="cpu", dtype=torch.float32)
    assert resolved is not None
    assert tuple(resolved.image_embeds.shape) == (4, 16)
    assert torch.allclose(resolved.image_embeds[:, :8], torch.ones((4, 8)))
    assert torch.allclose(resolved.image_embeds[:, 8:], torch.full((4, 8), 2.0))

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_proxy_releases_direct_feature_buffers_after_prefill_consumes_them():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from mooncake_epd.core.state import DirectFeatureBufferRegistry
    from mooncake_epd.scripts.direct_feature_buffer_service import create_app as create_direct_app

    record: dict = {}
    prefill_app = FastAPI()

    @prefill_app.post("/v1/chat/completions/render")
    async def render_chat(request: Request):
        body = await request.json()
        record["prefill_render_body"] = body
        return JSONResponse(
            {
                "request_id": "rendered-prefill-direct-release",
                "token_ids": [1, 2, 3, 4],
                "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16, "min_tokens": 0},
                "model": "fake-model",
                "stream": False,
                "priority": 0,
            }
        )

    @prefill_app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()
        record["prefill_generate_body"] = body
        kv = dict((body.get("sampling_params") or {}).get("extra_args", {}).get("kv_transfer_params") or {})
        provider = FeatureHandleProvider(FeatureHandleProviderConfig(worker_id="prefill-0", device="cpu", strict=True))
        resolved = provider.resolve_from_sources(kv, device="cpu", dtype=torch.float32)
        assert resolved is not None
        record["resolved_image_embeds_shape"] = list(resolved.image_embeds.shape)
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
                "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
                "kv_transfer_params": kv,
            }
        )

    decode_app = _build_decode_app(record)
    direct_engine = TransferEngine(protocol="tcp")
    fake_engine = _CopyingMooncakeEngine()
    direct_engine.bind_mooncake_backend(fake_engine, initialized=True, owns_backend=False)
    encoder_app = create_encoder_app(
        EncoderServiceConfig(publish_backend="direct_engine", device="cpu"),
        encoder=_DummyEncoder(),
        direct_transfer_engine=direct_engine,
    )
    registry = DirectFeatureBufferRegistry(
        worker_id="prefill-0",
        device="cpu",
        remote_session="prefill-session",
        register_memory=False,
    )
    direct_app = create_direct_app(registry=registry)
    proxy_app = create_proxy_app(
        ProxyConfig(
            mm_prefetch_mode="feature_handle",
            prefill_supports_feature_handles=True,
            encoder_service_url="http://encoder.local",
            prefill_direct_buffer_service_url="http://prefill-direct.local",
        ),
        prefill_clients=[_client_override(prefill_app, "prefill-0", "prefill.local", 8100)],
        decode_clients=[_client_override(decode_app, "decode-0", "decode.local", 8200)],
    )

    with TestClient(proxy_app) as client:
        proxy_app.state.encoder_client = httpx.AsyncClient(
            base_url="http://encoder.local",
            transport=httpx.ASGITransport(app=encoder_app),
            timeout=None,
        )
        proxy_app.state.prefill_direct_buffer_client = httpx.AsyncClient(
            base_url="http://prefill-direct.local",
            transport=httpx.ASGITransport(app=direct_app),
            timeout=None,
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": _png_data_url()}},
                            {"type": "text", "text": "describe"},
                        ],
                    }
                ],
                "metadata": {"workflow_id": "wf-online-direct-release"},
            },
        )
        assert response.status_code == 200, response.text

    assert record["resolved_image_embeds_shape"] == [4, 16]
    assert registry.stats()["allocations"] == 0
