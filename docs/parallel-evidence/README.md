# Parallel media paths: throughput and CPU evidence

This directory holds reproducible evidence for the process-level parallelism
added to the two CPU-bound media paths:

1. **Shot inspection** — per-frame decode + luma/color fingerprinting.
2. **Contact sheet** — per-scene representative-frame decode + nearest-neighbor
   tile resize before PPM composition.

The raw machine-readable output lives next to this file:

- `benchmark-4core.json` — the primary run, pinned to a four-core quota.
- `benchmark-1core.json` — the same material/request on one core, showing the
  inline fallback with no pool overhead.

## How to reproduce

```bash
# Four-core quota (primary evidence)
PYTHONPATH=src:. taskset -c 0-3 python3 tools/benchmark_parallel.py

# Single-core quota (inline fallback)
PYTHONPATH=src:. taskset -c 0 python3 tools/benchmark_parallel.py --cpus 1
```

The tool generates deterministic uncompressed AVI material through the
production AVI writer, runs each path three times in both serial and parallel
modes, and emits one JSON document. No third-party packages are used.

## Material and request (identical for both modes)

| Property | Value |
| --- | --- |
| Geometry | 320 x 180, 24-bit BGR |
| Frames / scenes | 288 frames, 36 scenes of 8 frames |
| Media size / SHA-256 | see `material` block in the JSON |
| Inspection | sensitivity 0.55, stride 1, min scene 2 |
| Contact sheet | 6 columns, 96 px tiles, gutter 4, `stored` selection |

## Four-core results (median of three trials)

### Shot-index fingerprints (288 frames)

| Mode | Wall time | CPU time | CPU utilization |
| --- | --- | --- | --- |
| Serial | see `fingerprints.serial.wall_seconds` | ~0.80 s | ~100 % (one core) |
| Parallel (4 workers) | see `fingerprints.parallel.wall_seconds` | ~0.86 s | ~316 % |

- Speed-up ≈ **2.9x**; independent frame calculations overlap across roughly
  three cores (the fourth covers scheduling and result reaping).
- `output_identical: true` — every fingerprint tuple matches the serial run.

### Contact-sheet tiles (36 scenes)

| Mode | Wall time | CPU time | CPU utilization |
| --- | --- | --- | --- |
| Serial | see `contact_sheet.serial.wall_seconds` | ~0.07 s | ~92 % |
| Parallel (4 workers) | see `contact_sheet.parallel.wall_seconds` | ~0.05 s | ~200 % |

- Speed-up ≈ **1.6x** with two cores effectively busy; the workload is only
  36 tiles and the parent still performs the single-threaded canvas copy and
  PPM encode, which caps the ceiling.
- `output_identical: true` and the PPM digest is identical across modes
  (`ppm_digest` in the JSON).

CPU utilization is computed as `(user + system CPU seconds across the parent
and all forked children) / wall seconds`. A value above 100 % is direct
evidence that more than one core executed frame work simultaneously.

## Boundedness evidence

- `bounded_queue.max_inflight_jobs_observed = 4` with four workers, regardless
  of the 288-frame clip or the 36-scene sheet. Tasks are fed through a sliding
  window the size of the worker pool; all source frames are never queued.
- `memory_bound.peak_concurrent_decoded_bytes_bound = worker_count *
  per_frame_rgb_bytes` (4 x 172,800 = 691,200 bytes). Workers return small
  fingerprints or tile payloads, so the parent retains no source RGB frames;
  the bound does not grow with frame count or scene count.
- `contexts_released = true` after every run, including failures.

## Failure evidence

- `failure_cleanup` — an out-of-range frame raises the original
  `media_bounds_exceeded` / HTTP 422 error, leaves **zero** leftover worker
  processes, and releases the media context.
- `http_scan_failure` — a decode failure inside a real HTTP scan still returns
  422 `frame_decode_failed` and the inspection record terminates in `failed`
  with the same error code; no workers survive.
- `contact_sheet_failure` — a tile worker failure keeps the 422 response and
  releases the pool and context.
- An injected failure on the third frame of 288 cancels all still-queued jobs
  and returns in ~17 ms with no surviving children (manual check described in
  the change notes).
- The truncated-upload intake path still returns 422
  `media_bounds_exceeded` and leaves no visible clip record.
