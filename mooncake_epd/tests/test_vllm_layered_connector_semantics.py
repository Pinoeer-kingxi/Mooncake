from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("vllm")

from mooncake_epd.core.control.vllm_mooncake_connector import (  # noqa: E402
    LayeredMooncakeXferResponse,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeXferResponseStatus,
    EPDMooncakeConnectorScheduler,
    UpstreamMooncakeConnectorWorker,
    _LayeredReceiveState,
    _LayeredSendState,
    EPDMooncakeConnectorWorker,
)
from mooncake_epd.core.control.connector_metrics import (  # noqa: E402
    ConnectorMetricsReader,
    ConnectorMetricsSink,
)
from mooncake_epd.core.control.vllm_transfer_primitives import LayeredTransferWorkerMeta  # noqa: E402


def _make_consumer_worker() -> EPDMooncakeConnectorWorker:
    worker = object.__new__(EPDMooncakeConnectorWorker)
    worker.layered_kv_transfer = True
    worker.trace_layered_kv = False
    worker.is_kv_producer = False
    worker.is_kv_consumer = True
    worker._worker_meta = LayeredTransferWorkerMeta()
    worker._layer_to_group = {"layer0": 0, "layer1": 0, "layer2": 1}
    worker.layer_load_timeout_seconds = 1.0
    worker._layered_recv_lock = threading.RLock()
    worker._layered_recv_states = {
        "req-0": _LayeredReceiveState.create(
            request_id="req-0",
            transfer_id="xfer-0",
            total_groups=2,
            expected_tasks=1,
        )
    }
    worker._current_recv_req_ids = ["req-0"]
    worker.finished_recving_reqs = set()
    worker._connector_metrics_pending_by_path = {}
    worker._recv_request_routing_paths = {}
    worker._recv_transfer_routing_paths = {}
    worker._send_request_routing_paths = {}
    worker._send_transfer_routing_paths = {}
    worker.async_zmq_ctx = type("Ctx", (), {"term": lambda self: None})()
    worker.receiver_loop = type(
        "Loop",
        (),
        {"is_running": lambda self: False, "call_soon_threadsafe": lambda *args, **kwargs: None},
    )()
    worker.shutdown = lambda: None
    return worker


def _make_producer_worker() -> EPDMooncakeConnectorWorker:
    worker = object.__new__(EPDMooncakeConnectorWorker)
    worker.layered_kv_transfer = True
    worker.trace_layered_kv = False
    worker.is_kv_consumer = False
    worker.is_kv_producer = True
    worker.layers_per_group = 2
    worker._sender_group_count = 2
    worker._layer_names = ["layer0", "layer1", "layer2"]
    worker._layer_to_group = {"layer0": 0, "layer1": 0, "layer2": 1}
    worker._layer_to_index = {"layer0": 0, "layer1": 1, "layer2": 2}
    worker._layered_send_lock = threading.RLock()
    worker._layered_send_states = {
        "xfer-0": _LayeredSendState.create("xfer-0", 2),
    }
    worker._current_send_transfer_ids = set()
    worker.reqs_need_send = {}
    worker.async_zmq_ctx = type("Ctx", (), {"term": lambda self: None})()
    worker.sender_loop = type(
        "Loop",
        (),
        {"is_running": lambda self: False, "call_soon_threadsafe": lambda *args, **kwargs: None},
    )()
    worker._sender_executor = type("Exec", (), {"shutdown": lambda *args, **kwargs: None})()
    worker.transport_backend = "mooncake_engine_direct"
    worker.group_delay_ms = 0.0
    worker.max_group_bytes = 0
    worker._registered_region_count = 4
    worker._worker_meta = LayeredTransferWorkerMeta()
    worker._connector_metrics_pending_by_path = {}
    worker._recv_request_routing_paths = {}
    worker._recv_transfer_routing_paths = {}
    worker._send_request_routing_paths = {}
    worker._send_transfer_routing_paths = {}
    worker.xfer_stats = type(
        "Stats",
        (),
        {
            "record_transfer": lambda *args, **kwargs: None,
            "record_failed_transfer": lambda *args, **kwargs: None,
        },
    )()
    worker._trace = lambda *args, **kwargs: None
    worker.shutdown = lambda: None
    return worker


def test_layered_connector_requires_piecewise_for_cudagraph():
    assert MooncakeConnector.requires_piecewise_for_cudagraph({}) is False
    assert MooncakeConnector.requires_piecewise_for_cudagraph(
        {"layered_kv_transfer": True}
    ) is True


def test_wait_for_layer_load_blocks_until_group_event_arrives():
    worker = _make_consumer_worker()
    state = worker._layered_recv_states["req-0"]  # noqa: SLF001

    def _ack():
        time.sleep(0.05)
        state.ack_group(0)

    thread = threading.Thread(target=_ack, daemon=True)
    thread.start()
    worker.wait_for_layer_load("layer0")
    thread.join(timeout=1.0)
    assert state.group_events[0].is_set()
    assert worker._worker_meta.layer_wait_calls == 1  # noqa: SLF001
    assert worker._worker_meta.layer_wait_ms > 0.0  # noqa: SLF001


def test_save_kv_layer_only_announces_group_tail():
    worker = _make_producer_worker()
    state = worker._layered_send_states["xfer-0"]  # noqa: SLF001

    worker.save_kv_layer("layer0", None, None)
    assert state.group_ready_events[0].is_set() is False

    worker.save_kv_layer("layer1", None, None)
    assert state.group_ready_events[0].is_set() is True

    worker.save_kv_layer("layer2", None, None)
    assert state.group_ready_events[1].is_set() is True


def test_layered_receive_state_can_resize_groups_without_losing_progress():
    state = _LayeredReceiveState.create(
        request_id="req-0",
        transfer_id="xfer-0",
        total_groups=2,
        expected_tasks=1,
    )
    state.ack_group(0)
    assert state.group_events[0].is_set() is True

    state.ensure_total_groups(3)

    assert state.total_groups == 3
    assert len(state.group_events) == 3
    assert state.group_events[0].is_set() is True
    assert state.group_events[1].is_set() is False
    assert state.group_events[2].is_set() is False


def test_process_pulling_result_aligns_total_groups_from_producer():
    worker = _make_consumer_worker()
    response = LayeredMooncakeXferResponse(
        status=MooncakeXferResponseStatus.CONTINUE,
        ok_reqs=["req-0"],
        group_index=0,
        total_groups=3,
    )

    worker.process_pulling_result(response, {"req-0": None})  # type: ignore[arg-type]

    state = worker._layered_recv_states["req-0"]  # noqa: SLF001
    assert state.total_groups == 3
    assert state.group_events[0].is_set() is True
    assert worker._worker_meta.received_group_batches == 1  # noqa: SLF001


def test_process_pulling_result_uses_decode_request_id_and_does_not_crash_on_trace():
    worker = _make_consumer_worker()
    worker._layered_recv_states = {
        "prefill-req-0": _LayeredReceiveState.create(
            request_id="prefill-req-0",
            transfer_id="xfer-0",
            total_groups=1,
            expected_tasks=1,
        )
    }
    worker._current_recv_req_ids = ["prefill-req-0"]
    worker.finished_recving_reqs = set()
    worker._recv_request_routing_paths = {"prefill-req-0": "EPD"}

    pull_meta = type("PullMeta", (), {"d_req_id": "decode-req-0"})()
    response = LayeredMooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["prefill-req-0"],
        group_index=0,
        total_groups=1,
    )

    worker.process_pulling_result(response, {"prefill-req-0": pull_meta})  # type: ignore[arg-type]

    assert "decode-req-0" in worker.finished_recving_reqs
    assert worker._worker_meta.received_finished_reqs == 1  # noqa: SLF001
    assert worker._connector_metrics_pending_by_path["EPD"].received_finished_reqs == 1  # noqa: SLF001
    assert "UNKNOWN" not in worker._connector_metrics_pending_by_path  # noqa: SLF001


def test_process_pulling_result_does_not_double_count_finished_request():
    worker = _make_consumer_worker()
    response0 = LayeredMooncakeXferResponse(
        status=MooncakeXferResponseStatus.CONTINUE,
        ok_reqs=["req-0"],
        group_index=0,
        total_groups=2,
    )
    response1 = LayeredMooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["req-0"],
        group_index=1,
        total_groups=2,
    )

    worker.process_pulling_result(response0, {"req-0": None})  # type: ignore[arg-type]
    worker.process_pulling_result(response1, {"req-0": None})  # type: ignore[arg-type]
    worker.process_pulling_result(response1, {"req-0": None})  # type: ignore[arg-type]

    assert worker._worker_meta.received_finished_reqs == 1  # noqa: SLF001


def test_record_send_reqs_creates_ready_placeholder_for_early_layered_send():
    worker = _make_producer_worker()
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_to_send["req-0"] = ("xfer-ready", [[10, 11], [20]])

    asyncio.run(worker.record_send_reqs(metadata))

    send_meta = worker.reqs_need_send["xfer-ready"]
    assert send_meta.p_req_id == "req-0"
    assert send_meta.local_block_ids == [[10, 11], [20]]
    assert send_meta.ready.is_set() is True
    assert "xfer-ready" in worker._layered_send_states  # noqa: SLF001


def test_record_send_reqs_keeps_pending_placeholder_unready():
    worker = _make_producer_worker()
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_to_send["req-0"] = ("xfer-pending", [])

    asyncio.run(worker.record_send_reqs(metadata))

    send_meta = worker.reqs_need_send["xfer-pending"]
    assert send_meta.p_req_id == "req-0"
    assert send_meta.local_block_ids == []
    assert send_meta.ready.is_set() is False


def test_record_send_reqs_keeps_all_empty_groups_unready():
    worker = _make_producer_worker()
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_to_send["req-0"] = ("xfer-empty-groups", [[], []])

    asyncio.run(worker.record_send_reqs(metadata))

    send_meta = worker.reqs_need_send["xfer-empty-groups"]
    assert send_meta.local_block_ids == [[], []]
    assert send_meta.ready.is_set() is False


def test_record_send_reqs_does_not_drop_ready_transfer_without_matching_not_processed_flag():
    worker = _make_producer_worker()
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_to_send["req-ready"] = ("xfer-ready", [[10, 11]])
    metadata.reqs_not_processed.add("xfer-other")

    asyncio.run(worker.record_send_reqs(metadata))

    assert "xfer-ready" in worker.reqs_need_send
    assert worker.reqs_need_send["xfer-ready"].ready.is_set() is True
    assert worker.reqs_need_send.get("xfer-other") is None


def test_scheduler_build_connector_meta_captures_routing_paths():
    scheduler = object.__new__(EPDMooncakeConnectorScheduler)
    scheduler.is_kv_producer = True
    scheduler.is_kv_consumer = False
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {
        "req-epd": (
            type(
                "Req",
                (),
                {
                    "kv_transfer_params": {
                        "transfer_id": "xfer-epd",
                        "routing_path": "EPD",
                    }
                },
            )(),
            [[1, 2]],
        )
    }
    scheduler._reqs_not_processed = set()

    meta = scheduler.build_connector_meta(None)

    assert meta.request_routing_paths["req-epd"] == "EPD"
    assert meta.transfer_routing_paths["xfer-epd"] == "EPD"


def test_batched_transfer_regions_uses_peer_buffer_direct_path():
    worker = _make_producer_worker()
    worker._transfer_region_descriptors_via_peer_engine = lambda *args, **kwargs: 0
    worker.engine = type(
        "Engine",
        (),
        {"batch_transfer_sync_write": lambda *args, **kwargs: 0},
    )()

    result = worker._batched_transfer_regions(
        "peer-0",
        [1, 2],
        [11, 12],
        [16, 32],
    )

    assert result.ret_code == 0
    assert result.backend_label == "peer_buffer_direct"
    assert result.used_fallback is False


def test_batched_transfer_regions_falls_back_to_raw_batch_write_on_peer_failure():
    worker = _make_producer_worker()
    worker._transfer_region_descriptors_via_peer_engine = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("peer path failed")
    )
    calls = []
    worker.engine = type(
        "Engine",
        (),
        {
            "batch_transfer_sync_write": lambda self, remote, src, dst, lengths: calls.append(
                (remote, list(src), list(dst), list(lengths))
            )
            or 0
        },
    )()

    result = worker._batched_transfer_regions(
        "peer-1",
        [3, 4],
        [13, 14],
        [64, 128],
    )

    assert result.ret_code == 0
    assert result.backend_label == "batch_transfer_fallback"
    assert result.used_fallback is True
    assert calls == [("peer-1", [3, 4], [13, 14], [64, 128])]


def test_send_region_group_records_peer_buffer_path_metrics():
    worker = _make_producer_worker()
    worker._transfer_region_descriptors_via_peer_engine = lambda *args, **kwargs: 0
    worker.engine = type(
        "Engine",
        (),
        {"batch_transfer_sync_write": lambda *args, **kwargs: 0},
    )()

    ret = worker._send_region_group(
        "peer-2",
        [5, 6],
        [15, 16],
        [32, 32],
    )

    assert ret == 0
    assert worker._worker_meta.grouped_batches == 1  # noqa: SLF001
    assert worker._worker_meta.peer_buffer_batches == 1  # noqa: SLF001
    assert worker._worker_meta.peer_buffer_bytes == 64  # noqa: SLF001
    assert worker._worker_meta.backend_counts["peer_buffer_direct"] == 1  # noqa: SLF001


def test_send_blocks_uses_direct_dispatch_even_when_not_layered(monkeypatch):
    worker = _make_producer_worker()
    worker.layered_kv_transfer = False
    calls = []

    def _super_should_not_run(*args, **kwargs):
        raise AssertionError("unexpected upstream _send_blocks fallback")

    monkeypatch.setattr(UpstreamMooncakeConnectorWorker, "_send_blocks", _super_should_not_run)
    worker._send_region_group = lambda remote, src, dst, lengths, **kwargs: calls.append(  # type: ignore[method-assign]
        (remote, list(src), list(dst), list(lengths))
    ) or 0

    ret = worker._send_blocks("peer-3", [7], [17], [96])

    assert ret == 0
    assert calls == [("peer-3", [7], [17], [96])]


def test_send_blocks_with_descriptor_paths_preserves_path_totals(tmp_path):
    worker = _make_producer_worker()
    worker._connector_metrics_sink = ConnectorMetricsSink(  # noqa: SLF001
        tmp_path,
        engine_id="engine-path-aware",
        role="producer",
        hostname="host-4",
        rpc_port=9004,
        tp_rank=0,
    )
    worker.layers_per_group = 1
    worker._batched_transfer_regions = lambda *args, **kwargs: type(  # noqa: SLF001
        "Dispatch",
        (),
        {"ret_code": 0, "backend_label": "peer_buffer_direct"},
    )()

    ret = worker._send_blocks(
        "peer-path-aware",
        [1, 2, 3, 4, 5, 6],
        [11, 12, 13, 14, 15, 16],
        [32, 32, 64, 64, 96, 96],
        ["PD", "PD", "EPD", "EPD", "EPD", "EPD"],
    )

    assert ret == 0
    aggregate = ConnectorMetricsReader(tmp_path).aggregate()
    totals = aggregate.totals
    path_totals = aggregate.path_totals
    assert totals.grouped_batches == 3
    assert sum(meta.grouped_batches for meta in path_totals.values()) == totals.grouped_batches
    assert sum(meta.grouped_bytes for meta in path_totals.values()) == totals.grouped_bytes
    assert sum(meta.grouped_descriptors for meta in path_totals.values()) == totals.grouped_descriptors
    assert sum(meta.peer_buffer_batches for meta in path_totals.values()) == totals.peer_buffer_batches
    assert sum(meta.peer_buffer_bytes for meta in path_totals.values()) == totals.peer_buffer_bytes
    assert path_totals["PD"].grouped_batches == 1
    assert path_totals["EPD"].grouped_batches == 2


def test_build_connector_worker_meta_flushes_shared_metrics(tmp_path):
    worker = _make_producer_worker()
    worker._connector_metrics_sink = ConnectorMetricsSink(  # noqa: SLF001
        tmp_path,
        engine_id="engine-test",
        role="producer",
        hostname="host-0",
        rpc_port=9000,
        tp_rank=0,
    )
    worker._worker_meta = LayeredTransferWorkerMeta(  # noqa: SLF001
        grouped_batches=2,
        grouped_bytes=128,
        grouped_descriptors=4,
        peer_buffer_batches=2,
        peer_buffer_bytes=128,
        backend_counts={"peer_buffer_direct": 2},
    )
    worker._connector_metrics_pending = worker._worker_meta  # noqa: SLF001
    worker._connector_metrics_pending_by_path = {  # noqa: SLF001
        "EPD": LayeredTransferWorkerMeta(
            grouped_batches=2,
            grouped_bytes=128,
            grouped_descriptors=4,
        )
    }

    meta = worker.build_connector_worker_meta()

    assert meta is not None
    assert "rank0" in worker._connector_metrics_sink.path.name  # noqa: SLF001
    payload = worker._connector_metrics_sink.path.read_text(encoding="utf-8")  # noqa: SLF001
    assert '"peer_buffer_direct": 2' in payload
    assert '"path_totals"' in payload
    assert '"EPD"' in payload


def test_consumer_build_connector_worker_meta_flushes_receive_metrics(tmp_path):
    worker = _make_consumer_worker()
    worker._connector_metrics_sink = ConnectorMetricsSink(  # noqa: SLF001
        tmp_path,
        engine_id="engine-consumer",
        role="consumer",
        hostname="host-1",
        rpc_port=9001,
        tp_rank=0,
    )
    worker._worker_meta = LayeredTransferWorkerMeta(  # noqa: SLF001
        received_group_batches=2,
        received_finished_reqs=1,
        layer_wait_calls=3,
        layer_wait_ms=4.5,
    )
    worker._connector_metrics_pending = worker._worker_meta  # noqa: SLF001

    meta = worker.build_connector_worker_meta()

    assert meta is not None
    payload = worker._connector_metrics_sink.path.read_text(encoding="utf-8")  # noqa: SLF001
    assert '"received_group_batches": 2' in payload


def test_process_pulling_result_publishes_shared_metrics_immediately(tmp_path):
    worker = _make_consumer_worker()
    worker._connector_metrics_sink = ConnectorMetricsSink(  # noqa: SLF001
        tmp_path,
        engine_id="engine-consumer-live",
        role="consumer",
        hostname="host-2",
        rpc_port=9002,
        tp_rank=0,
    )
    worker._recv_request_routing_paths = {"req-0": "EPD"}  # noqa: SLF001
    response = LayeredMooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH,
        ok_reqs=["req-0"],
        group_index=0,
        total_groups=1,
    )

    worker.process_pulling_result(response, {"req-0": None})  # type: ignore[arg-type]

    payload = worker._connector_metrics_sink.path.read_text(encoding="utf-8")  # noqa: SLF001
    assert '"received_group_batches": 1' in payload
    assert '"received_finished_reqs": 1' in payload
    assert '"path_totals"' in payload
    assert '"EPD"' in payload


def test_send_region_group_publishes_shared_metrics_immediately(tmp_path):
    worker = _make_producer_worker()
    worker._connector_metrics_sink = ConnectorMetricsSink(  # noqa: SLF001
        tmp_path,
        engine_id="engine-producer-live",
        role="producer",
        hostname="host-3",
        rpc_port=9003,
        tp_rank=0,
    )
    worker._batched_transfer_regions = lambda *args, **kwargs: type(  # noqa: SLF001
        "Dispatch",
        (),
        {"ret_code": 0, "backend_label": "peer_buffer_direct"},
    )()

    ret = worker._send_region_group("peer-live", [1, 2], [3, 4], [32, 32])

    assert ret == 0
    payload = worker._connector_metrics_sink.path.read_text(encoding="utf-8")  # noqa: SLF001
    assert '"grouped_batches": 1' in payload
    assert '"peer_buffer_direct": 1' in payload


def test_send_blocks_applies_transport_safety_chunking_for_large_descriptor_burst():
    worker = _make_producer_worker()
    worker.layered_kv_transfer = True
    worker.layers_per_group = 128
    worker._registered_region_count = 128
    worker.max_group_bytes = 0
    worker.max_transfer_descriptors = 4
    worker.max_transfer_bytes = 64
    calls = []

    def _dispatch(remote, src, dst, lengths, path_stats=None, **kwargs):
        calls.append((list(src), list(dst), list(lengths), dict(path_stats or {})))
        return type("Dispatch", (), {"ret_code": 0, "backend_label": "peer_buffer_direct"})()

    worker._send_region_group_dispatch = _dispatch  # type: ignore[method-assign]

    ret = worker._send_blocks(
        "peer-chunked",
        list(range(10)),
        list(range(100, 110)),
        [16] * 10,
        ["EPD"] * 10,
    )

    assert ret == 0
    assert [len(src) for src, _, _, _ in calls] == [4, 4, 2]
    assert [sum(lengths) for _, _, lengths, _ in calls] == [64, 64, 32]


def test_send_blocks_applies_transport_safety_chunking_even_without_layered_mode():
    worker = _make_producer_worker()
    worker.layered_kv_transfer = False
    worker.max_transfer_descriptors = 2
    worker.max_transfer_bytes = 0
    calls = []

    def _dispatch(remote, src, dst, lengths, path_stats=None, **kwargs):
        calls.append((list(src), list(dst), list(lengths)))
        return type("Dispatch", (), {"ret_code": 0, "backend_label": "batch_transfer_native"})()

    worker._send_region_group_dispatch = _dispatch  # type: ignore[method-assign]

    ret = worker._send_blocks("peer-nonlayered", [1, 2, 3], [11, 12, 13], [8, 8, 8])

    assert ret == 0
    assert [len(src) for src, _, _ in calls] == [2, 1]


def test_direct_peer_failure_can_be_fail_fast_when_fallback_disabled():
    worker = _make_producer_worker()
    worker.allow_transfer_fallback = False
    worker._transfer_region_descriptors_via_peer_engine = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("direct path failed")
    )
    worker.engine = type(
        "Engine",
        (),
        {"batch_transfer_sync_write": lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run"))},
    )()

    result = worker._batched_transfer_regions("peer-strict", [1], [2], [3])

    assert result.ret_code != 0
    assert result.backend_label == "peer_buffer_direct"
    assert "direct path failed" in (result.error_message or "")


def test_save_kv_layer_only_marks_active_transfer_scope():
    worker = _make_producer_worker()
    worker._layered_send_states["xfer-1"] = _LayeredSendState.create("xfer-1", 2)  # noqa: SLF001
    worker._current_send_transfer_ids = {"xfer-1"}  # noqa: SLF001

    worker.save_kv_layer("layer1", None, None)

    assert worker._layered_send_states["xfer-0"].group_ready_events[0].is_set() is False  # noqa: SLF001
    assert worker._layered_send_states["xfer-1"].group_ready_events[0].is_set() is True  # noqa: SLF001


def test_save_kv_layer_marks_earlier_groups_when_late_tail_is_first():
    worker = _make_producer_worker()

    worker.save_kv_layer("layer2", None, None)

    state = worker._layered_send_states["xfer-0"]  # noqa: SLF001
    assert state.group_ready_events[0].is_set() is True
    assert state.group_ready_events[1].is_set() is True
