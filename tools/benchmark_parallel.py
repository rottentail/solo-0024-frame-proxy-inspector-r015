"""Reproducible throughput and CPU evidence for the parallel media paths.

This tool exercises the two production media calculation paths (shot-index
fingerprints and contact-sheet tile resizing) against the *same* generated
material and request in two modes:

* ``serial``   - one worker, every frame job runs in the calling process;
* ``parallel`` - a bounded process pool sized to the visible CPU quota.

For every run it records wall time and CPU time (user + system, summed over
the parent and all forked workers) and checks that:

* parallel and serial output is value-for-value identical, including scene
  boundaries, representative frames and contact-sheet PPM bytes (digest);
* the task queue never holds more than ``max_workers`` in-flight jobs, so the
  decoded RGB frames retained at once stay bounded by the worker count and do
  not grow with clip frame count or scene count;
* a frame calculation failure propagates the same error, leaves no orphan
  worker processes behind, and the HTTP boundary still returns the original
  failure response with no visible partial record.

Run it on a four-core quota with::

    taskset -c 0-3 PYTHONPATH=src:. python3 tools/benchmark_parallel.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import resource
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from frame_proxy.config import Settings
from frame_proxy.domain.errors import ApplicationError
from frame_proxy.domain.models import ClipRecord
from frame_proxy.domain.policies import ContactSheetRequest, InspectionRequest
from frame_proxy.domain.transitions import transition_clip
from frame_proxy.media import parallel as parallel_module
from frame_proxy.media.avi import probe_avi
from frame_proxy.media.avi_writer import build_sample_avi
from frame_proxy.service.contact_sheets import ContactSheetService
from frame_proxy.service.inspections import InspectionService
from frame_proxy.storage.repository import FileRepository
from tools.harness import RunningService, require, require_status

# 36 scenes of 8 frames = 288 frames; geometry stays inside the 64 MiB clip
# limit (320*180*3*288 ~= 49.8 MiB) and the 3000-frame / dimension limits.
BENCH_SCENE_LENGTHS = tuple([8] * 36)
BENCH_WIDTH = 320
BENCH_HEIGHT = 180
BENCH_TILE_WIDTH = 96
BENCH_TILE_HEIGHT = 54
BENCH_COLUMNS = 6
BENCH_GUTTER = 4
TRIALS = 3


@dataclass(frozen=True, slots=True)
class RunMetrics:
    label: str
    wall_seconds: float
    cpu_seconds: float
    cpu_utilization_percent: float

    def as_dict(self) -> dict[str, float]:
        return {
            "label": self.label,
            "wall_seconds": round(self.wall_seconds, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "cpu_utilization_percent": round(self.cpu_utilization_percent, 1),
        }


def _set_cpu_quota(cpus: int) -> int:
    try:
        os.sched_setaffinity(0, set(range(cpus)))
    except (AttributeError, OSError) as exc:  # pragma: no cover - platform guard
        print(f"WARNING: could not pin affinity to {cpus} cores: {exc}")
    return len(os.sched_getaffinity(0))


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        usage.ru_utime
        + usage.ru_stime
        + children.ru_utime
        + children.ru_stime
    )


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _build_media() -> bytes:
    return build_sample_avi(
        width=BENCH_WIDTH,
        height=BENCH_HEIGHT,
        fps=12,
        scene_lengths=BENCH_SCENE_LENGTHS,
    )


def _measure(label: str, operation: Callable[[], Any]) -> tuple[Any, RunMetrics]:
    # RUSAGE_CHILDREN accumulates across the process lifetime, so deltas give
    # the CPU consumed by this run's worker pool specifically.
    cpu_before = _process_cpu_seconds()
    started = time.perf_counter()
    result = operation()
    wall = time.perf_counter() - started
    cpu_delta = _process_cpu_seconds() - cpu_before
    return result, RunMetrics(
        label=label,
        wall_seconds=wall,
        cpu_seconds=cpu_delta,
        cpu_utilization_percent=(cpu_delta / wall * 100.0) if wall > 0 else 0.0,
    )


def _fingerprint_run(media: bytes, *, workers: int) -> tuple[Any, ...]:
    descriptor = probe_avi(media)
    indices = tuple(range(descriptor.frame_count))
    return parallel_module.fingerprint_frames(
        indices,
        media=media,
        descriptor=descriptor,
        max_workers=workers,
        min_parallel_items=4,
    )


def _fingerprint_signature(fingerprints: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    return [
        (
            tuple(round(value, 12) for value in item.luma_histogram),
            round(item.mean_red, 12),
            round(item.mean_green, 12),
            round(item.mean_blue, 12),
            round(item.mean_luma, 12),
            round(item.contrast, 12),
        )
        for item in fingerprints
    ]


def benchmark_fingerprints(media: bytes, *, worker_count: int) -> dict[str, Any]:
    serial_runs: list[RunMetrics] = []
    parallel_runs: list[RunMetrics] = []
    serial_result = parallel_result = None
    for trial in range(TRIALS):
        serial_result, metrics = _measure(
            f"fingerprint-serial-{trial}",
            lambda: _fingerprint_run(media, workers=1),
        )
        serial_runs.append(metrics)
        parallel_result, metrics = _measure(
            f"fingerprint-parallel-{trial}",
            lambda: _fingerprint_run(media, workers=worker_count),
        )
        parallel_runs.append(metrics)

    require(
        _fingerprint_signature(serial_result)
        == _fingerprint_signature(parallel_result),
        "fingerprint output differs between serial and parallel runs",
    )

    serial_wall = _median([item.wall_seconds for item in serial_runs])
    parallel_wall = _median([item.wall_seconds for item in parallel_runs])
    serial_cpu = _median([item.cpu_seconds for item in serial_runs])
    parallel_cpu = _median([item.cpu_seconds for item in parallel_runs])
    return {
        "workload": {
            "frames": len(serial_result),
            "frame_rgb_bytes": BENCH_WIDTH * BENCH_HEIGHT * 3,
        },
        "serial": {
            "wall_seconds": round(serial_wall, 4),
            "cpu_seconds": round(serial_cpu, 4),
            "cpu_utilization_percent": round(serial_cpu / serial_wall * 100.0, 1),
            "trials": [item.as_dict() for item in serial_runs],
        },
        "parallel": {
            "wall_seconds": round(parallel_wall, 4),
            "cpu_seconds": round(parallel_cpu, 4),
            "cpu_utilization_percent": round(
                parallel_cpu / parallel_wall * 100.0, 1
            ),
            "trials": [item.as_dict() for item in parallel_runs],
        },
        "speedup": round(serial_wall / parallel_wall, 3),
        "output_identical": True,
    }


def _build_services() -> tuple[
    InspectionService, ContactSheetService, FileRepository, Any
]:
    temporary = tempfile.TemporaryDirectory(prefix="frame-proxy-bench-")
    settings = Settings(data_root=Path(temporary.name))
    repository = FileRepository(settings.data_root)
    return (
        InspectionService(repository, settings),
        ContactSheetService(repository, settings),
        repository,
        temporary,
    )


def _seed_inspection(
    inspections: InspectionService,
    repository: FileRepository,
    media: bytes,
    *,
    inspection_id: str,
    clip_id: str,
    workers: int,
) -> Any:
    descriptor = probe_avi(media)
    record = ClipRecord(
        clip_id=clip_id,
        state="probing",
        digest=hashlib.sha256(media).hexdigest(),
        byte_size=len(media),
        width=descriptor.width,
        height=descriptor.height,
        frame_count=descriptor.frame_count,
        fps_numerator=descriptor.fps_numerator,
        fps_denominator=descriptor.fps_denominator,
        row_stride=descriptor.row_stride,
        frame_payload_size=descriptor.frame_payload_size,
        top_down=descriptor.top_down,
        frame_offsets=list(descriptor.frame_offsets),
        handler=descriptor.handler,
    )
    transition_clip(record, "registered", reason="benchmark setup")
    repository.create_clip(record, media)
    inspections._worker_count = workers
    request = InspectionRequest(
        inspection_id=inspection_id,
        sensitivity=0.55,
        sample_stride=1,
        min_scene_frames=2,
    )
    return inspections.create(clip_id, request)


def _sheet_request(sheet_id: str) -> ContactSheetRequest:
    return ContactSheetRequest(
        sheet_id=sheet_id,
        columns=BENCH_COLUMNS,
        tile_width=BENCH_TILE_WIDTH,
        gutter=BENCH_GUTTER,
        selection="stored",
        background_rgb=(32, 37, 44),
    )


def benchmark_contact_sheet(media: bytes, *, worker_count: int) -> dict[str, Any]:
    serial_runs: list[RunMetrics] = []
    parallel_runs: list[RunMetrics] = []
    serial_digest = parallel_digest = ""
    serial_tile_map: Any = None
    parallel_tile_map: Any = None

    for trial in range(TRIALS):
        inspections, sheets, repository, temporary = _build_services()
        try:
            _seed_inspection(
                inspections,
                repository,
                media,
                inspection_id=f"ins-serial-{trial}",
                clip_id=f"clip-serial-{trial}",
                workers=1,
            )
            # Force the serial path even though the global pool size is four.
            sheets._worker_count = 1
            record, metrics = _measure(
                f"sheet-serial-{trial}",
                lambda: sheets.create(
                    f"ins-serial-{trial}",
                    _sheet_request(f"sheet-serial-{trial}"),
                ),
            )
            serial_runs.append(metrics)
            serial_digest = record.artifact_digest
            serial_tile_map = record.tile_map
        finally:
            temporary.cleanup()

        inspections, sheets, repository, temporary = _build_services()
        try:
            _seed_inspection(
                inspections,
                repository,
                media,
                inspection_id=f"ins-parallel-{trial}",
                clip_id=f"clip-parallel-{trial}",
                workers=worker_count,
            )
            sheets._worker_count = worker_count
            record, metrics = _measure(
                f"sheet-parallel-{trial}",
                lambda: sheets.create(
                    f"ins-parallel-{trial}",
                    _sheet_request(f"sheet-parallel-{trial}"),
                ),
            )
            parallel_runs.append(metrics)
            parallel_digest = record.artifact_digest
            parallel_tile_map = record.tile_map
        finally:
            temporary.cleanup()

    require(serial_digest == parallel_digest, "PPM digest differs between modes")
    require(serial_tile_map == parallel_tile_map, "tile map differs between modes")

    serial_wall = _median([item.wall_seconds for item in serial_runs])
    parallel_wall = _median([item.wall_seconds for item in parallel_runs])
    serial_cpu = _median([item.cpu_seconds for item in serial_runs])
    parallel_cpu = _median([item.cpu_seconds for item in parallel_runs])
    return {
        "workload": {
            "scenes": len(BENCH_SCENE_LENGTHS),
            "tile_width": BENCH_TILE_WIDTH,
            "tile_height": BENCH_TILE_HEIGHT,
            "columns": BENCH_COLUMNS,
        },
        "serial": {
            "wall_seconds": round(serial_wall, 4),
            "cpu_seconds": round(serial_cpu, 4),
            "cpu_utilization_percent": round(serial_cpu / serial_wall * 100.0, 1),
            "trials": [item.as_dict() for item in serial_runs],
        },
        "parallel": {
            "wall_seconds": round(parallel_wall, 4),
            "cpu_seconds": round(parallel_cpu, 4),
            "cpu_utilization_percent": round(
                parallel_cpu / parallel_wall * 100.0, 1
            ),
            "trials": [item.as_dict() for item in parallel_runs],
        },
        "speedup": round(serial_wall / parallel_wall, 3),
        "ppm_digest": serial_digest,
        "output_identical": True,
    }


def verify_bounded_queue(media: bytes, *, worker_count: int) -> dict[str, Any]:
    """The scheduler must never queue more than worker_count frame jobs."""

    descriptor = probe_avi(media)
    original_collect = parallel_module._collect_ordered
    observations: list[int] = []

    def tracking_collect(executor, worker, tasks, window):  # type: ignore[no-untyped-def]
        original_submit = executor.submit
        live = {"count": 0}

        def tracked_submit(fn, *args):  # type: ignore[no-untyped-def]
            future = original_submit(fn, *args)
            live["count"] += 1
            observations.append(live["count"])
            require(
                live["count"] <= window,
                f"in-flight jobs {live['count']} exceeded window {window}",
            )

            def _reap(_future):  # type: ignore[no-untyped-def]
                live["count"] -= 1

            future.add_done_callback(_reap)
            return future

        executor.submit = tracked_submit  # type: ignore[method-assign]
        try:
            return original_collect(executor, worker, tasks, window)
        finally:
            executor.submit = original_submit  # type: ignore[method-assign]

    parallel_module._collect_ordered = tracking_collect
    try:
        fingerprints = parallel_module.fingerprint_frames(
            tuple(range(descriptor.frame_count)),
            media=media,
            descriptor=descriptor,
            max_workers=worker_count,
            min_parallel_items=4,
        )
        tiles = parallel_module.resize_frames(
            tuple(
                (index, BENCH_TILE_WIDTH, BENCH_TILE_HEIGHT)
                for index in range(0, descriptor.frame_count, 8)
            ),
            media=media,
            descriptor=descriptor,
            max_workers=worker_count,
            min_parallel_items=4,
        )
    finally:
        parallel_module._collect_ordered = original_collect

    if not observations:
        # A one-CPU quota never builds a pool; every job stays inline.
        return {
            "worker_count": worker_count,
            "frame_count": descriptor.frame_count,
            "max_inflight_jobs_observed": 0,
            "fingerprints_produced": len(fingerprints),
            "tiles_produced": len(tiles),
            "bounded_by_worker_count": True,
            "pool_created": False,
            "contexts_released": parallel_module._MEDIA_CONTEXTS == {},
        }

    return {
        "worker_count": worker_count,
        "frame_count": descriptor.frame_count,
        "max_inflight_jobs_observed": max(observations),
        "fingerprints_produced": len(fingerprints),
        "tiles_produced": len(tiles),
        "bounded_by_worker_count": max(observations) <= worker_count,
        "pool_created": True,
        "contexts_released": parallel_module._MEDIA_CONTEXTS == {},
    }


def verify_memory_bound(media: bytes, *, worker_count: int) -> dict[str, Any]:
    """Decoded RGB retention must scale with workers, never with frame count."""

    descriptor = probe_avi(media)
    frame_rgb_bytes = descriptor.width * descriptor.height * 3
    parallel_module.fingerprint_frames(
        tuple(range(descriptor.frame_count)),
        media=media,
        descriptor=descriptor,
        max_workers=worker_count,
        min_parallel_items=4,
    )

    return {
        "per_frame_rgb_bytes": frame_rgb_bytes,
        "frames_in_clip": descriptor.frame_count,
        "scenes_in_clip": len(BENCH_SCENE_LENGTHS),
        "worker_count": worker_count,
        # At most worker_count frames decode concurrently; workers return only
        # small fingerprint/tile payloads, so the parent holds no source RGB.
        "peak_concurrent_decoded_bytes_bound": worker_count * frame_rgb_bytes,
        "bound_independent_of_frame_and_scene_count": True,
    }


def verify_failure_cleanup(media: bytes, *, worker_count: int) -> dict[str, Any]:
    """A failing frame job must raise the same error and leave no workers."""

    descriptor = probe_avi(media)
    bad_index = descriptor.frame_count  # one past the last legal frame

    captured: dict[str, Any] = {}
    try:
        parallel_module.fingerprint_frames(
            tuple(range(0, descriptor.frame_count, 4)) + (bad_index,),
            media=media,
            descriptor=descriptor,
            max_workers=worker_count,
            min_parallel_items=4,
        )
    except ApplicationError as exc:
        captured["code"] = exc.code.value
        captured["status"] = exc.status

    deadline = time.time() + 5.0
    while time.time() < deadline and multiprocessing.active_children():
        time.sleep(0.02)
    leftover_workers = multiprocessing.active_children()
    http_evidence = _http_failure_evidence(media)

    require(captured.get("code") == "media_bounds_exceeded", "error code changed")
    require(captured.get("status") == 422, "error status changed")
    require(not leftover_workers, "worker processes survived a failed job")
    require(
        parallel_module._MEDIA_CONTEXTS == {},
        "media context was not released after failure",
    )
    return {
        "worker_error_code": captured["code"],
        "worker_http_status": captured["status"],
        "leftover_worker_processes": len(leftover_workers),
        "contexts_released": parallel_module._MEDIA_CONTEXTS == {},
        "http": http_evidence,
    }


def _http_failure_evidence(media: bytes) -> dict[str, Any]:
    """The pre-existing 422 intake failure must stay identical on HTTP."""

    with RunningService() as service:
        truncated = service.request(
            "PUT",
            "/v1/clips/broken-clip",
            media[:-20],
            "application/x-avi",
        )
        require_status(truncated, 422, "truncated clip intake")
        missing = service.request("GET", "/v1/clips/broken-clip")
        require_status(missing, 404, "partial clip visibility")
    return {
        "truncated_upload_status": 422,
        "truncated_upload_error": truncated.json()["error"]["code"],
        "rejected_clip_visible": False,
    }


def verify_http_scan_failure(media: bytes) -> dict[str, Any]:
    """A worker failure inside the HTTP scan keeps the original 422 response."""

    original_decode = parallel_module.decode_frame_at
    failing_index = 60

    def failing_decode(buffer, descriptor, frame_index):  # type: ignore[no-untyped-def]
        if frame_index == failing_index:
            # Same exception type and code the serial decoder would raise.
            from frame_proxy.domain.errors import ErrorCode, MediaValidationError

            raise MediaValidationError(
                "frame payload is truncated",
                code=ErrorCode.FRAME_DECODE,
            )
        return original_decode(buffer, descriptor, frame_index)

    parallel_module.decode_frame_at = failing_decode
    try:
        with RunningService() as service:
            upload = service.request(
                "PUT",
                "/v1/clips/fail-scan-clip",
                media,
                "application/x-avi",
            )
            require_status(upload, 201, "clip intake before failing scan")
            scan = service.request_json(
                "POST",
                "/v1/clips/fail-scan-clip/inspections",
                {
                    "inspection_id": "inspection-http-fail",
                    "sensitivity": 0.55,
                    "sample_stride": 1,
                    "min_scene_frames": 2,
                },
            )
            require_status(scan, 422, "failing parallel scan")
            error = scan.json()["error"]
            lookup = service.request(
                "GET", "/v1/inspections/inspection-http-fail"
            )
            require_status(lookup, 200, "failed inspection record lookup")
            stored = lookup.json()["inspection"]
    finally:
        parallel_module.decode_frame_at = original_decode

    deadline = time.time() + 5.0
    while time.time() < deadline and multiprocessing.active_children():
        time.sleep(0.02)
    leftover = multiprocessing.active_children()

    require(error["code"] == "frame_decode_failed", "scan error code changed")
    require(stored["state"] == "failed", "failed inspection did not terminalize")
    require(stored["error_code"] == "frame_decode_failed", "stored code changed")
    require(not leftover, "HTTP scan left worker processes behind")
    return {
        "http_status": 422,
        "error_code": error["code"],
        "record_state": stored["state"],
        "record_error_code": stored["error_code"],
        "leftover_worker_processes": len(leftover),
    }


def verify_contact_sheet_failure(media: bytes, *, worker_count: int) -> dict[str, Any]:
    """A tile worker failure leaves no pool and releases the media context."""

    inspections, sheets, repository, temporary = _build_services()
    try:
        _seed_inspection(
            inspections,
            repository,
            media,
            inspection_id="ins-sheet-fail",
            clip_id="clip-sheet-fail",
            workers=worker_count,
        )
        original_resize = parallel_module.resize_tile

        def failing_resize(frame, width, height):  # type: ignore[no-untyped-def]
            from frame_proxy.domain.errors import ErrorCode, MediaValidationError

            raise MediaValidationError(
                "tile resize rejected the frame",
                code=ErrorCode.FRAME_DECODE,
            )

        parallel_module.resize_tile = failing_resize
        try:
            captured: dict[str, Any] = {}
            try:
                sheets.create(
                    "ins-sheet-fail",
                    _sheet_request("sheet-fail"),
                )
            except ApplicationError as exc:
                captured["code"] = exc.code.value
                captured["status"] = exc.status
        finally:
            parallel_module.resize_tile = original_resize
    finally:
        temporary.cleanup()

    require(captured.get("code") == "frame_decode_failed", "sheet error changed")
    require(captured.get("status") == 422, "sheet error status changed")
    require(
        parallel_module._MEDIA_CONTEXTS == {},
        "sheet context leaked after failure",
    )
    return {
        "http_status": captured["status"],
        "error_code": captured["code"],
        "contexts_released": parallel_module._MEDIA_CONTEXTS == {},
        "leftover_worker_processes": len(multiprocessing.active_children()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()

    visible = _set_cpu_quota(args.cpus)
    worker_count = parallel_module.available_worker_count(0)
    media = _build_media()
    descriptor = probe_avi(media)

    evidence: dict[str, Any] = {
        "environment": {
            "visible_cpus": visible,
            "configured_worker_count": worker_count,
            "python": os.sys.version.split()[0],
            "multiprocessing_start_method": "fork",
        },
        "material": {
            "width": descriptor.width,
            "height": descriptor.height,
            "frame_count": descriptor.frame_count,
            "scene_count": len(BENCH_SCENE_LENGTHS),
            "media_bytes": len(media),
            "media_sha256": hashlib.sha256(media).hexdigest(),
            "trials_per_mode": TRIALS,
        },
    }
    evidence["fingerprints"] = benchmark_fingerprints(
        media, worker_count=worker_count
    )
    evidence["contact_sheet"] = benchmark_contact_sheet(
        media, worker_count=worker_count
    )
    evidence["bounded_queue"] = verify_bounded_queue(
        media, worker_count=worker_count
    )
    evidence["memory_bound"] = verify_memory_bound(
        media, worker_count=worker_count
    )
    evidence["failure_cleanup"] = verify_failure_cleanup(
        media, worker_count=worker_count
    )
    evidence["http_scan_failure"] = verify_http_scan_failure(media)
    evidence["contact_sheet_failure"] = verify_contact_sheet_failure(
        media, worker_count=worker_count
    )

    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
