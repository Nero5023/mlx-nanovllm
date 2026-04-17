# continuous-batch-worker benchmark

Correctness: **PASS**. Chip: Apple M4. Max active requests: 4, batch wait: 2.0 ms, KV cache=True.

TTFT includes queueing and client notification. ITL pools consecutive token gaps; TPOT averages gaps per request. Values are milliseconds. Throughput is the median of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.

| Case | Runs | OK / all | Admission group sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| single | 3 | 3 / 3 | 1 | 185.87 / 200.16 | 39.68 / 64.03 | 42.67 | 2.55 | 22.27 | 0.671 |
| burst-1 | 3 | 3 / 3 | 1 | 147.89 / 150.05 | 39.33 / 60.93 | 41.14 | 2.54 | 23.39 | 0.671 |
| burst-2 | 3 | 6 / 6 | 2 | 265.28 / 291.49 | 39.40 / 68.50 | 41.57 | 2.53 | 43.98 | 0.780 |
| burst-4 | 3 | 12 / 12 | 4 | 360.25 / 418.46 | 46.47 / 72.93 | 45.48 | 0.02 | 77.97 | 1.028 |
| mixed-burst | 3 | 12 / 12 | 4 | 3209.08 / 3640.01 | 46.05 / 81.38 | 58.54 | 0.02 | 19.65 | 2.582 |
| staggered | 3 | 12 / 12 | 1, 3 | 409.12 / 507.71 | 37.66 / 68.61 | 47.66 | 90.88 | 30.05 | 0.883 |
| prefix-shared | 3 | 12 / 12 | 1 | 346.29 / 435.88 | 36.90 / 60.54 | 36.91 | 2.55 | 22.04 | 0.980 |
| prefix-control | 3 | 12 / 12 | 1 | 390.31 / 559.33 | 38.31 / 63.01 | 37.92 | 2.55 | 19.91 | 0.980 |
| context-128 | 3 | 3 / 3 | 1 | 159.63 / 162.36 | 40.38 / 64.13 | 42.83 | 2.55 | 21.51 | 0.671 |
| context-512 | 3 | 3 / 3 | 1 | 345.56 / 364.13 | 38.18 / 60.53 | 38.56 | 2.55 | 20.49 | 0.980 |
| context-2048 | 3 | 3 / 3 | 1 | 1863.47 / 2386.23 | 51.13 / 71.65 | 52.74 | 2.53 | 9.03 | 1.801 |
| context-burst | 3 | 12 / 12 | 4 | 7724.08 / 7777.64 | 75.65 / 104.82 | 90.85 | 0.02 | 12.14 | 4.435 |

Raw JSON includes p99 and sample counts, worker token timestamps, client arrival timestamps, submission lag, per-request input/output lengths, token IDs, errors, end-to-end latency and allocator cache memory.

These are short synthetic workloads, not steady-state production load. Small-sample tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. Queued requests have no active KV cache; active batch rows do. Concurrency is client in-flight work; admission group sizes do not track growing/shrinking decode tensors.

The largest successful context/concurrency is only the largest tested configuration. Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.

See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.
