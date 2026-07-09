"""Workspace-scoped runtime patches for Mooncake EPD real vLLM serving.

This module is loaded automatically by Python when the workspace root is on
``PYTHONPATH``. The generated vLLM serving scripts already export that path,
so the patch stays local to this project instead of mutating site-packages.

Patch scope:
1. Allow ``SamplingParams(max_tokens=0)`` for prompt-only prefill.
2. Finalize prompt-only prefill requests after the prompt forward completes so
   KV handoff metadata is emitted without sampling a first decode token.
3. Install safe Mooncake FeatureHandle helpers for vLLM Qwen-VL multimodal
   hidden-state reuse.  The helper is fail-open by default for compatibility,
   but re-raises in MOONCAKE_EPD_STRICT / strict-no-fallback evaluation mode.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable


logger = logging.getLogger("mooncake_epd.sitecustomize")


def _strict_no_fallback() -> bool:
    try:
        from mooncake_epd.core.strict_mode import strict_no_fallback_enabled

        return strict_no_fallback_enabled()
    except Exception:
        return False


def _patch_vllm_prompt_only_prefill() -> None:
    try:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.utils import remove_all
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
        from vllm.v1.request import RequestStatus
    except Exception:
        return

    if getattr(SamplingParams, "_mooncake_epd_prompt_only_patch", False):
        return

    original_verify_args = SamplingParams._verify_args

    def _patched_verify_args(self: SamplingParams) -> None:
        if self.max_tokens == 0:
            if self.min_tokens != 0:
                raise ValueError(
                    "min_tokens must be 0 when prompt-only prefill uses max_tokens=0."
                )
            original_max_tokens = self.max_tokens
            self.max_tokens = 1
            try:
                original_verify_args(self)
            finally:
                self.max_tokens = original_max_tokens
            return
        original_verify_args(self)

    SamplingParams._verify_args = _patched_verify_args  # type: ignore[method-assign]

    original_update_from_output = Scheduler.update_from_output

    def _patched_update_from_output(
        self: Scheduler,
        scheduler_output: Any,
        model_runner_output: Any,
    ) -> dict[int, EngineCoreOutputs]:
        engine_core_outputs = original_update_from_output(
            self,
            scheduler_output,
            model_runner_output,
        )

        prompt_only_requests = []
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            if getattr(request, "max_tokens", None) != 0:
                continue
            if request.num_output_tokens != 0:
                continue
            if request.num_computed_tokens < request.num_tokens:
                continue
            prompt_only_requests.append(request)

        for request in prompt_only_requests:
            status_before_stop = request.status
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            finish_reason = request.get_finished_reason()
            finished = self._handle_stopped_request(request)
            kv_transfer_params = self._free_request(request) if finished else None

            if status_before_stop == RequestStatus.RUNNING:
                self.running = remove_all(self.running, {request})
            else:
                self.waiting.remove_requests({request})

            eco = engine_core_outputs.setdefault(
                request.client_index,
                EngineCoreOutputs(),
            )
            eco.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=finish_reason,
                    events=request.take_events(),
                    prefill_stats=request.take_prefill_stats(),
                    kv_transfer_params=kv_transfer_params,
                    trace_headers=request.trace_headers,
                    num_nans_in_logits=request.num_nans_in_logits,
                )
            )

        return engine_core_outputs

    Scheduler.update_from_output = _patched_update_from_output  # type: ignore[method-assign]
    SamplingParams._mooncake_epd_prompt_only_patch = True  # type: ignore[attr-defined]
    logger.info("enabled Mooncake EPD prompt-only prefill patch for vLLM")


def _has_var_kwargs(fn: Callable[..., Any]) -> bool:
    try:
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in inspect.signature(fn).parameters.values()
        )
    except Exception:
        return False


def _accepts_kw(fn: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return True
    return name in sig.parameters or _has_var_kwargs(fn)


def _patch_callable_for_feature_handles(cls: type, method_name: str) -> bool:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_mooncake_epd_feature_handle_patch", False):
        return False
    if not callable(original):
        return False

    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            from mooncake_epd.core.state.vllm_feature_handle_provider import (
                maybe_inject_feature_handle_kwargs,
            )

            if "image_embeds" not in kwargs and (
                kwargs.get("kv_transfer_params") is not None
                or kwargs.get("mm_feature_handles") is not None
                or kwargs.get("mooncake_epd_feature_handles") is not None
            ):
                injected = maybe_inject_feature_handle_kwargs(
                    kwargs,
                    device=getattr(getattr(self, "device", None), "type", None) or getattr(self, "device", None),
                    dtype=getattr(self, "dtype", None),
                )
                if injected is not kwargs:
                    # Only forward keys accepted by this vLLM version.  Unknown
                    # names are removed before calling upstream code.
                    kwargs = {
                        key: value
                        for key, value in injected.items()
                        if key in kwargs or _accepts_kw(original, key)
                    }
        except Exception as exc:
            logger.debug("Mooncake FeatureHandle injection skipped: %s", exc)
            if _strict_no_fallback():
                raise
        return original(self, *args, **kwargs)

    _patched._mooncake_epd_feature_handle_patch = True  # type: ignore[attr-defined]
    setattr(cls, method_name, _patched)
    return True


def _patch_vllm_feature_handle_injection() -> None:
    """Best-effort vLLM multimodal hidden-state injection.

    vLLM model internals move across versions.  We patch only known Qwen-VL
    multimodal methods and only when the method can accept an ``image_embeds``
    style kwarg (or arbitrary kwargs).  If no compatible method exists, this
    leaves serving untouched; ``scripts/check_vllm_feature_handle_patch.py`` can
    be used to make that explicit before a real EPD hidden-state run.
    """

    targets = (
        ("vllm.model_executor.models.qwen3_vl", "Qwen3VLForConditionalGeneration"),
        ("vllm.model_executor.models.qwen3_vl", "Qwen3VLModel"),
        ("vllm.model_executor.models.qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("vllm.model_executor.models.qwen2_vl", "Qwen2VLForConditionalGeneration"),
    )
    methods = (
        "embed_multimodal",
        "get_multimodal_embeddings",
        "get_input_embeddings",
        "forward",
    )
    patched = []
    for module_name, class_name in targets:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception:
            continue
        for method_name in methods:
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            if not (_accepts_kw(method, "image_embeds") or _has_var_kwargs(method)):
                continue
            if _patch_callable_for_feature_handles(cls, method_name):
                patched.append(f"{module_name}.{class_name}.{method_name}")
    if patched:
        logger.info("enabled Mooncake EPD FeatureHandle vLLM hooks: %s", ", ".join(patched))


def _patch_vllm_gpu_model_runner_feature_handles() -> None:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return
    original = getattr(GPUModelRunner, "_batch_mm_inputs_from_scheduler", None)
    if original is None or getattr(original, "_mooncake_epd_feature_handle_patch", False):
        return

    def _patched_batch_mm_inputs_from_scheduler(self: Any, scheduler_output: Any) -> Any:
        result = original(self, scheduler_output)
        try:
            from mooncake_epd.core.state.vllm_feature_handle_provider import (
                inject_feature_handles_into_vllm_mm_kwargs,
            )

            mm_hashes, mm_kwargs, mm_lora_refs = result
            return inject_feature_handles_into_vllm_mm_kwargs(
                mm_hashes=mm_hashes,
                mm_kwargs=mm_kwargs,
                mm_lora_refs=mm_lora_refs,
                requests=getattr(self, "requests", {}),
                device=getattr(self, "device", None),
                dtype=getattr(self, "dtype", None),
            )
        except Exception as exc:
            logger.debug("Mooncake FeatureHandle runner injection skipped: %s", exc)
            if _strict_no_fallback():
                raise
            return result

    _patched_batch_mm_inputs_from_scheduler._mooncake_epd_feature_handle_patch = True  # type: ignore[attr-defined]
    GPUModelRunner._batch_mm_inputs_from_scheduler = _patched_batch_mm_inputs_from_scheduler  # type: ignore[method-assign]
    logger.info("enabled Mooncake EPD FeatureHandle GPUModelRunner hook")


def _patch_vllm_gpu_model_runner_kv_params() -> None:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        return
    original = getattr(GPUModelRunner, "_update_states", None)
    if original is None or getattr(original, "_mooncake_epd_kv_params_patch", False):
        return

    def _patched_update_states(self: Any, scheduler_output: Any) -> Any:
        result = original(self, scheduler_output)
        try:
            for new_req_data in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
                req_id = getattr(new_req_data, "req_id", None)
                if req_id is None:
                    continue
                req_state = getattr(self, "requests", {}).get(req_id)
                if req_state is None:
                    continue
                sampling_params = getattr(new_req_data, "sampling_params", None)
                extra_args = getattr(sampling_params, "extra_args", None) if sampling_params is not None else None
                kv_transfer_params = None
                if isinstance(extra_args, dict):
                    kv_transfer_params = extra_args.get("kv_transfer_params")
                if kv_transfer_params:
                    setattr(req_state, "kv_transfer_params", dict(kv_transfer_params))
                    try:
                        from mooncake_epd.core.state.vllm_mm_hidden_cache import trace_vllm_mm_hidden_event

                        trace_vllm_mm_hidden_event(
                            "gpu_runner_request_kv_params_attached",
                            req_id=req_id,
                            has_feature_handles=bool(kv_transfer_params.get("mm_feature_handles")),
                        )
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("Mooncake kv_transfer_params attach skipped: %s", exc)
            if _strict_no_fallback():
                raise
        return result

    _patched_update_states._mooncake_epd_kv_params_patch = True  # type: ignore[attr-defined]
    GPUModelRunner._update_states = _patched_update_states  # type: ignore[method-assign]
    logger.info("enabled Mooncake EPD GPUModelRunner kv_transfer_params attach hook")


_patch_vllm_prompt_only_prefill()
_patch_vllm_feature_handle_injection()
_patch_vllm_gpu_model_runner_kv_params()
_patch_vllm_gpu_model_runner_feature_handles()

# ---------------------------------------------------------------------------
# Prefill-owned E→P direct FeatureBundle allocation routes
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    import os

    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _install_direct_feature_buffer_routes(app: Any) -> None:
    """Install Prefill-process-owned direct FeatureBundle allocation routes.

    This must run inside the same vLLM Prefill API process that later executes
    the FeatureHandleProvider hook.  It is disabled unless
    ``MOONCAKE_EPD_ENABLE_DIRECT_FEATURE_BUFFER=1`` to avoid exposing extra
    endpoints on unrelated vLLM servers.
    """

    if getattr(app, "_mooncake_epd_direct_feature_buffer_routes", False):
        return
    if not _env_bool("MOONCAKE_EPD_ENABLE_DIRECT_FEATURE_BUFFER", False):
        return
    try:
        import os
        from fastapi import HTTPException
        from mooncake_epd.core.state import (
            DirectFeatureBufferRegistry,
            FeatureBundleDescriptor,
            register_direct_feature_buffer_registry,
        )
        from mooncake_epd.core.transfer import TransferEngine
    except Exception as exc:
        logger.error("Mooncake direct feature buffer route install failed: %s", exc)
        if _strict_no_fallback():
            raise
        return

    worker_id = os.getenv(
        "MOONCAKE_EPD_DIRECT_BUFFER_WORKER_ID",
        os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_WORKER_ID", "prefill"),
    )
    device = os.getenv(
        "MOONCAKE_EPD_DIRECT_BUFFER_DEVICE",
        os.getenv("MOONCAKE_EPD_FEATURE_HANDLE_DEVICE", "cuda"),
    )
    engine = TransferEngine(
        protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
        local_hostname=os.getenv(
            "MOONCAKE_EPD_DIRECT_LOCAL_HOSTNAME",
            os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
        ),
        metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
        device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
    )
    registry = DirectFeatureBufferRegistry(
        worker_id=worker_id,
        device=device,
        transfer_engine=engine,
        remote_session=os.getenv("MOONCAKE_EPD_DIRECT_REMOTE_SESSION"),
        register_memory=_env_bool("MOONCAKE_EPD_DIRECT_REGISTER_MEMORY", True),
        target_memory_mode=os.getenv("MOONCAKE_EPD_DIRECT_TARGET_MODE", "auto"),
    )
    register_direct_feature_buffer_registry(registry)
    setattr(app, "_mooncake_epd_direct_feature_buffer_registry", registry)

    async def _allocate(payload: dict[str, Any]) -> dict[str, Any]:
        raw_descriptors = payload.get("descriptors")
        if raw_descriptors is None and isinstance(payload.get("descriptor"), dict):
            raw_descriptors = [payload.get("descriptor")]
        if not isinstance(raw_descriptors, list) or not raw_descriptors:
            raise HTTPException(status_code=400, detail="allocate requires descriptors[]")
        targets = []
        try:
            for raw in raw_descriptors:
                descriptor = FeatureBundleDescriptor.from_dict(dict(raw or {}))
                allocation = registry.allocate_for_descriptor(
                    descriptor,
                    zero_fill=bool(payload.get("zero_fill", False)),
                )
                targets.append(allocation.as_direct_target())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"direct buffer allocation failed: {exc}") from exc
        return {"targets": targets, "count": len(targets), "worker_id": registry.worker_id}

    async def _release(payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("feature_ids") or payload.get("features") or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="release requires feature_ids[]")
        for feature_id in raw:
            registry.release(str(feature_id))
        return {"released": len(raw), "stats": dict(registry.stats())}

    async def _stats() -> dict[str, Any]:
        return dict(registry.stats())

    prefix = "/mooncake_epd/direct_feature_buffer"
    app.post(f"{prefix}/allocate")(_allocate)
    app.post(f"{prefix}/release")(_release)
    app.get(f"{prefix}/stats")(_stats)
    if _env_bool("MOONCAKE_EPD_DIRECT_BUFFER_ROOT_ROUTES", True):
        app.post("/allocate")(_allocate)
        app.post("/release")(_release)
        app.get("/direct_feature_buffer_stats")(_stats)
    setattr(app, "_mooncake_epd_direct_feature_buffer_routes", True)
    logger.info(
        "enabled Mooncake EPD direct FeatureBuffer routes in vLLM process: worker=%s device=%s",
        worker_id,
        device,
    )


def _patch_vllm_openai_app_direct_feature_buffer_routes() -> None:
    try:
        from vllm.entrypoints.openai import api_server
    except Exception:
        return
    original = getattr(api_server, "build_app", None)
    if original is None or getattr(original, "_mooncake_epd_direct_feature_buffer_patch", False):
        return

    def _patched_build_app(*args: Any, **kwargs: Any) -> Any:
        app = original(*args, **kwargs)
        try:
            _install_direct_feature_buffer_routes(app)
        except Exception:
            if _strict_no_fallback():
                raise
            logger.exception("Mooncake direct FeatureBuffer route install skipped")
        return app

    _patched_build_app._mooncake_epd_direct_feature_buffer_patch = True  # type: ignore[attr-defined]
    api_server.build_app = _patched_build_app  # type: ignore[assignment]
    logger.info("enabled Mooncake EPD vLLM build_app direct FeatureBuffer hook")


_patch_vllm_openai_app_direct_feature_buffer_routes()
