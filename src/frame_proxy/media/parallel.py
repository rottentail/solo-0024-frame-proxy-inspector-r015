"""Bounded process-level parallelism for per-frame media calculations.

Both heavy media paths (shot-index fingerprints and contact-sheet tile
resizing) are embarrassingly parallel per frame but cannot overlap while the
GIL is held.  This module runs those frame calculations in a short-lived
:class:`~concurrent.futures.ProcessPoolExecutor`.

The design is deliberately bounded:

* The source media buffer is published once under a unique context key.
  Worker processes are created with ``fork`` and inherit that buffer
  copy-on-write, so the media bytes are never pickled per task.  Contexts are
  keyed, so concurrent HTTP requests running on server threads never share or
  overwrite each other's media.
* Tasks are submitted through a sliding window of at most ``max_workers``
  in-flight futures.  All source frames are never queued up front, so the
  number of decoded RGB frames retained at any moment is independent of the
  clip frame count and the number of scenes.
* The executor is created per request and is always shut down with
  ``cancel_futures=True``; when one frame calculation fails, every queued or
  running task is cancelled or reaped before the error propagates, so no
  background worker keeps consuming CPU afterwards.
"""

from __future__ import annotations

import itertools
import multiprocessing
import multiprocessing.context
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, TypeVar

from frame_proxy.media.avi import AviDescriptor
from frame_proxy.media.contact import resize_tile
from frame_proxy.media.fingerprint import FrameFingerprint, fingerprint_frame
from frame_proxy.media.frames import decode_frame_at

ResultT = TypeVar("ResultT")

# Media contexts inherited by forked workers.  Each request registers its
# media under a unique key and releases it in a ``finally`` block, so the map
# never grows with the number of frames or scenes in a clip.
_MEDIA_CONTEXTS: dict[str, tuple[bytes, AviDescriptor]] = {}
_CONTEXT_LOCK = threading.Lock()
_CONTEXT_SEQ = itertools.count()


def _register_context(media: bytes, descriptor: AviDescriptor) -> str:
    key = f"media-{os.getpid()}-{next(_CONTEXT_SEQ)}"
    with _CONTEXT_LOCK:
        _MEDIA_CONTEXTS[key] = (media, descriptor)
    return key


def _release_context(key: str) -> None:
    with _CONTEXT_LOCK:
        _MEDIA_CONTEXTS.pop(key, None)


def _context_for(key: str) -> tuple[bytes, AviDescriptor]:
    try:
        return _MEDIA_CONTEXTS[key]
    except KeyError as exc:  # pragma: no cover - internal contract failure
        raise RuntimeError("media worker context is not initialized") from exc


def _fork_context() -> multiprocessing.context.BaseContext:
    # Fork lets workers inherit the media buffer copy-on-write instead of
    # re-sending it through a pickled initializer for every task.
    return multiprocessing.get_context("fork")


def _worker_fingerprint(task: tuple[str, int]) -> FrameFingerprint:
    key, frame_index = task
    media, descriptor = _context_for(key)
    return fingerprint_frame(decode_frame_at(media, descriptor, frame_index))


def _worker_resize_tile(
    task: tuple[str, int, int, int],
) -> tuple[int, bytes]:
    key, frame_index, tile_width, tile_height = task
    media, descriptor = _context_for(key)
    frame = decode_frame_at(media, descriptor, frame_index)
    return frame_index, resize_tile(frame, tile_width, tile_height)


def _run_ordered(
    worker: Callable[..., ResultT],
    tasks: Sequence[Any],
    *,
    max_workers: int,
    min_parallel_items: int,
) -> list[ResultT]:
    """Run tasks in order, using a pool only for sufficiently large workloads."""

    if len(tasks) < min_parallel_items or max_workers < 2 or len(tasks) < 2:
        # Small workloads stay inline so tiny clips pay no process overhead.
        return [worker(task) for task in tasks]

    executor: ProcessPoolExecutor | None = None
    worker_limit = min(max_workers, len(tasks))
    try:
        executor = ProcessPoolExecutor(
            max_workers=worker_limit,
            mp_context=_fork_context(),
        )
        return _collect_ordered(executor, worker, tasks, worker_limit)
    finally:
        # cancel_futures drops queued (not yet started) tasks and the wait
        # reaps in-flight ones, so a failure never leaves background CPU work.
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _collect_ordered(
    executor: ProcessPoolExecutor,
    worker: Callable[..., ResultT],
    tasks: Sequence[Any],
    window: int,
) -> list[ResultT]:
    """Submit at most one task per worker slot and keep output ordered."""

    pending: dict[Any, int] = {}
    results: list[ResultT | None] = [None] * len(tasks)
    next_submit = 0

    def submit_more() -> None:
        nonlocal next_submit
        while next_submit < len(tasks) and len(pending) < window:
            future = executor.submit(worker, tasks[next_submit])
            pending[future] = next_submit
            next_submit += 1

    submit_more()
    while pending:
        wait(tuple(pending), return_when=FIRST_COMPLETED)
        # Reap every ready future; waiting on an already-done future is cheap.
        ready = [future for future in pending if future.done()]
        for future in ready:
            index = pending.pop(future)
            # Re-raise worker failures here; the outer finally tears the pool
            # down and cancels everything still queued.
            results[index] = future.result()
        submit_more()

    return results  # type: ignore[return-value]


def available_worker_count(configured: int) -> int:
    """Resolve the worker limit, clamping to CPUs visible to this process."""

    if configured > 0:
        limit = configured
    else:
        try:
            limit = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux fallback
            limit = os.cpu_count() or 1
    return max(1, limit)


def fingerprint_frames(
    frame_indices: Iterable[int],
    *,
    media: bytes,
    descriptor: AviDescriptor,
    max_workers: int,
    min_parallel_items: int,
) -> tuple[FrameFingerprint, ...]:
    """Fingerprint every sampled frame, preserving frame order."""

    indices = tuple(frame_indices)
    context_key = _register_context(media, descriptor)
    try:
        tasks = tuple((context_key, frame_index) for frame_index in indices)
        return tuple(
            _run_ordered(
                _worker_fingerprint,
                tasks,
                max_workers=max_workers,
                min_parallel_items=min_parallel_items,
            )
        )
    finally:
        _release_context(context_key)


def resize_frames(
    frame_tasks: Iterable[tuple[int, int, int]],
    *,
    media: bytes,
    descriptor: AviDescriptor,
    max_workers: int,
    min_parallel_items: int,
) -> dict[int, bytes]:
    """Decode and resize selected frames into ``{frame_index: tile_bytes}``."""

    frame_tasks = tuple(frame_tasks)
    context_key = _register_context(media, descriptor)
    try:
        worker_tasks = tuple(
            (context_key, frame_index, width, height)
            for (frame_index, width, height) in frame_tasks
        )
        results = _run_ordered(
            _worker_resize_tile,
            worker_tasks,
            max_workers=max_workers,
            min_parallel_items=min_parallel_items,
        )
        return {frame_index: tile for (frame_index, tile) in results}
    finally:
        _release_context(context_key)
