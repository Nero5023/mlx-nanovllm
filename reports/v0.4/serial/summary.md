# fifo-worker benchmark

Correctness: **PASS**. Chip: Apple M4. Max active requests: 1, batch wait: 0.0 ms, KV cache=True.

TTFT includes queueing and client notification. ITL pools consecutive token gaps; TPOT averages gaps per request. Values are milliseconds. Throughput is the median of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.

| Case | Runs | OK / all | Admission group sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| single | 3 | 3 / 3 | 1 | 161.02 / 165.06 | 34.33 / 57.40 | 32.77 | 0.03 | 28.84 | 0.667 |
| burst-1 | 3 | 3 / 3 | 1 | 154.59 / 165.29 | 35.56 / 63.15 | 32.44 | 0.03 | 28.96 | 0.667 |
| burst-2 | 3 | 6 / 6 | 1 | 1323.14 / 2966.32 | 39.71 / 61.25 | 41.19 | 1131.47 | 24.68 | 0.667 |
| burst-4 | 3 | 12 / 12 | 1 | 3504.84 / 8398.92 | 38.41 / 61.04 | 40.18 | 3337.07 | 23.65 | 0.667 |
| mixed-burst | 3 | 12 / 12 | 1 | 6719.41 / 8114.34 | 40.56 / 65.50 | 39.76 | 6574.02 | 20.62 | 1.427 |
| staggered | 3 | 12 / 12 | 1 | 5360.17 / 6609.80 | 36.69 / 62.11 | 39.41 | 5192.56 | 25.28 | 0.667 |
| prefix-shared | 3 | 12 / 12 | 1 | 352.13 / 463.73 | 36.55 / 63.28 | 36.80 | 0.02 | 20.97 | 0.980 |
| prefix-control | 3 | 12 / 12 | 1 | 365.08 / 393.30 | 38.47 / 60.86 | 40.65 | 0.02 | 20.18 | 0.980 |
| context-128 | 3 | 3 / 3 | 1 | 143.08 / 160.65 | 35.74 / 59.05 | 33.32 | 0.02 | 27.98 | 0.667 |
| context-512 | 3 | 3 / 3 | 1 | 394.73 / 460.68 | 37.80 / 69.49 | 37.89 | 0.02 | 21.15 | 0.980 |
| context-2048 | 3 | 3 / 3 | 1 | 1593.92 / 1637.02 | 48.69 / 76.83 | 47.70 | 0.02 | 10.69 | 1.839 |
| context-burst | 3 | 12 / 12 | 1 | 5975.40 / 11007.17 | 46.25 / 74.63 | 51.28 | 4363.20 | 10.18 | 1.839 |

Raw JSON includes p99 and sample counts, worker token timestamps, client arrival timestamps, submission lag, per-request input/output lengths, token IDs, errors, end-to-end latency and allocator cache memory.

These are short synthetic workloads, not steady-state production load. Small-sample tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. Queued requests have no active KV cache; active batch rows do. Concurrency is client in-flight work; admission group sizes do not track growing/shrinking decode tensors.

The largest successful context/concurrency is only the largest tested configuration. Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.

See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.
