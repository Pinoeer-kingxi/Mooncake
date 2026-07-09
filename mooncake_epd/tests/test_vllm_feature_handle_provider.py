from __future__ import annotations

import os

import pytest
import torch

from mooncake_epd.core.state import (
    FeatureBundle,
    FeatureHandleProvider,
    FeatureHandleProviderConfig,
    FeatureHandleRegistry,
    MMStore,
    maybe_inject_feature_handle_kwargs,
    publish_feature_bundle_to_dir,
    register_feature_handle_registry,
    resolve_feature_handles_for_vllm,
    unregister_feature_handle_registry,
)


def _bundle(feature_id: str, base: float) -> FeatureBundle:
    return FeatureBundle(
        image_hash=feature_id,
        last_hidden=torch.tensor([[base, base + 1.0], [base + 2.0, base + 3.0]]),
        intermediates=[(3, torch.tensor([[base + 10.0], [base + 11.0]]))],
        grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
        metadata={"model_fingerprint": "model-a", "processor_fingerprint": "processor-a"},
    )


def test_file_backed_feature_handles_resolve_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONCAKE_EPD_VLLM_MM_HIDDEN_CACHE", "1")
    monkeypatch.setenv("MOONCAKE_EPD_CONNECTOR_METRICS_DIR", str(tmp_path / "metrics"))
    store_dir = tmp_path / "feature_store"
    h1 = publish_feature_bundle_to_dir(_bundle("img-a", 1.0), store_dir)
    h2 = publish_feature_bundle_to_dir(_bundle("img-b", 5.0), store_dir)

    provider = FeatureHandleProvider(
        FeatureHandleProviderConfig(
            worker_id="prefill-test",
            device="cpu",
            store_dirs=(store_dir,),
            expected_model_fingerprint="model-a",
            expected_processor_fingerprint="processor-a",
        )
    )
    resolved = provider.resolve_from_sources(
        {"kv_transfer_params": {"mm_feature_handles": [h1.as_control_payload(), h2.as_control_payload()]}}
    )

    assert resolved is not None
    assert resolved.count == 2
    # vLLM Qwen3-VL consumes packed image embeds: main hidden || deepstack hidden.
    assert torch.equal(
        resolved.image_embeds,
        torch.tensor([[1.0, 2.0, 11.0], [3.0, 4.0, 12.0], [5.0, 6.0, 15.0], [7.0, 8.0, 16.0]]),
    )
    assert torch.equal(resolved.image_grid_thw, torch.tensor([[1, 2, 2], [1, 2, 2]]))
    assert len(resolved.deepstack_image_embeds) == 1
    assert resolved.deepstack_image_embeds[0][0] == 3
    assert torch.equal(resolved.deepstack_image_embeds[0][1], torch.tensor([[11.0], [12.0], [15.0], [16.0]]))


def test_in_process_registry_feature_handle_resolution():
    mm_store = MMStore()
    registry = FeatureHandleRegistry(mm_store, store_id="unit-mm-store")
    bundle = _bundle("img-reg", 2.0)
    handle = registry.publish_bundle(bundle)
    register_feature_handle_registry(registry)
    try:
        resolved = resolve_feature_handles_for_vllm(
            {"mm_feature_handles": [handle.as_control_payload()]},
            device="cpu",
            provider=FeatureHandleProvider(FeatureHandleProviderConfig(worker_id="prefill", device="cpu")),
        )
    finally:
        unregister_feature_handle_registry("unit-mm-store")
        mm_store.stop()

    assert resolved is not None
    assert torch.equal(resolved.image_embeds, torch.cat([bundle.last_hidden, bundle.intermediates[0][1]], dim=-1))


def test_feature_handle_provider_is_fail_open_unless_strict(tmp_path):
    missing = publish_feature_bundle_to_dir(_bundle("missing", 1.0), tmp_path)
    os.remove(tmp_path / "missing.pt")
    fail_open = FeatureHandleProvider(
        FeatureHandleProviderConfig(device="cpu", store_dirs=(tmp_path,), strict=False)
    )
    assert fail_open.resolve_from_sources({"mm_feature_handles": [missing.as_control_payload()]}) is None

    strict = FeatureHandleProvider(
        FeatureHandleProviderConfig(device="cpu", store_dirs=(tmp_path,), strict=True)
    )
    with pytest.raises(Exception):
        strict.resolve_from_sources({"mm_feature_handles": [missing.as_control_payload()]})


def test_maybe_inject_feature_handle_kwargs_preserves_existing_embeds(tmp_path):
    handle = publish_feature_bundle_to_dir(_bundle("img-a", 1.0), tmp_path)
    existing = torch.ones(1, 2)
    provider = FeatureHandleProvider(FeatureHandleProviderConfig(device="cpu", store_dirs=(tmp_path,)))

    unchanged = maybe_inject_feature_handle_kwargs(
        {"image_embeds": existing, "kv_transfer_params": {"mm_feature_handles": [handle.as_control_payload()]}},
        provider=provider,
    )
    assert unchanged["image_embeds"] is existing

    injected = maybe_inject_feature_handle_kwargs(
        {"kv_transfer_params": {"mm_feature_handles": [handle.as_control_payload()]}},
        provider=provider,
    )
    expected_bundle = _bundle("img-a", 1.0)
    assert torch.equal(injected["image_embeds"], torch.cat([expected_bundle.last_hidden, expected_bundle.intermediates[0][1]], dim=-1))
    assert "image_grid_thw" in injected


def test_vllm_mm_kwargs_injection_replaces_pixels_when_vllm_available(tmp_path):
    pytest.importorskip("vllm")
    from types import SimpleNamespace

    from vllm.multimodal.inputs import (  # type: ignore
        MultiModalFieldConfig,
        MultiModalFieldElem,
        MultiModalKwargsItem,
    )

    from mooncake_epd.core.state import inject_feature_handles_into_vllm_mm_kwargs

    bundle = _bundle("stable-mm-hash", 3.0)
    handle = publish_feature_bundle_to_dir(
        bundle,
        tmp_path,
        metadata={"source_mm_hash": "stable-mm-hash"},
    )
    flat_field = MultiModalFieldConfig.flat("image", [slice(0, 4)]).field
    batched_field = MultiModalFieldConfig.batched("image").field
    original = MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(data=torch.zeros(4, 2), field=flat_field),
            "image_grid_thw": MultiModalFieldElem(data=torch.tensor([1, 2, 2]), field=batched_field),
        }
    )
    req = SimpleNamespace(
        kv_transfer_params={"mm_feature_handles": [handle.as_control_payload()]}
    )
    provider = FeatureHandleProvider(FeatureHandleProviderConfig(device="cpu", store_dirs=(tmp_path,)))

    _, converted_kwargs, _ = inject_feature_handles_into_vllm_mm_kwargs(
        mm_hashes=["stable-mm-hash"],
        mm_kwargs=[("image", original)],
        mm_lora_refs=[("req-1", object())],
        requests={"req-1": req},
        device="cpu",
        provider=provider,
    )

    converted = converted_kwargs[0][1]
    assert "pixel_values" not in converted
    assert "image_embeds" in converted
    assert torch.equal(converted["image_embeds"].data, torch.cat([bundle.last_hidden, bundle.intermediates[0][1]], dim=-1))
    assert torch.equal(converted["image_grid_thw"].data, torch.tensor([1, 2, 2]))
