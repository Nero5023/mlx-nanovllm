# fifo-worker benchmark

Correctness: **PASS**. Chip: Apple M4. Max active requests: 1, batch wait: 0.0 ms, KV cache=True.

TTFT includes queueing and client notification. ITL pools consecutive token gaps; TPOT averages gaps per request. Values are milliseconds. Throughput is the median of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.

| Case | Runs | OK / all | Actual batch sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| single | 3 | 3 / 3 | 1 | 108.35 / 111.01 | 11.08 / 15.23 | 11.42 | 0.02 | 77.33 | 0.667 |
| burst-1 | 3 | 3 / 3 | 1 | 102.54 / 110.18 | 10.51 / 15.57 | 11.15 | 0.02 | 79.56 | 0.667 |
| burst-2 | 3 | 6 / 6 | 1 | 497.20 / 900.69 | 10.02 / 15.12 | 11.03 | 398.01 | 81.09 | 0.667 |
| burst-4 | 3 | 12 / 12 | 1 | 1252.97 / 2448.63 | 9.87 / 15.01 | 10.96 | 1161.29 | 81.58 | 0.667 |
| mixed-burst | 3 | 12 / 12 | 1 | 2510.69 / 2912.28 | 12.68 / 16.32 | 11.35 | 2420.81 | 57.41 | 1.427 |
| staggered | 3 | 12 / 12 | 1 | 1620.08 / 1955.96 | 9.70 / 15.00 | 10.86 | 1531.22 | 78.03 | 0.667 |
| prefix-shared | 3 | 12 / 12 | 1 | 296.83 / 316.42 | 11.43 / 15.76 | 11.99 | 0.02 | 48.12 | 0.980 |
| prefix-control | 3 | 12 / 12 | 1 | 296.48 / 310.16 | 11.33 / 15.60 | 12.09 | 0.02 | 47.92 | 0.980 |
| context-128 | 3 | 3 / 3 | 1 | 102.09 / 107.87 | 9.81 / 15.07 | 10.87 | 0.02 | 71.85 | 0.667 |
| context-512 | 3 | 3 / 3 | 1 | 308.02 / 310.25 | 11.55 / 17.50 | 12.59 | 0.02 | 45.65 | 0.980 |
| context-2048 | 3 | 3 / 3 | 1 | 1430.30 / 1479.10 | 15.90 / 20.66 | 16.54 | 0.02 | 16.47 | 1.839 |
| context-burst | 3 | 12 / 12 | 1 | 4327.01 / 7172.14 | 16.27 / 21.57 | 16.62 | 2865.88 | 17.03 | 1.839 |

Raw JSON includes p99 and sample counts, worker token timestamps, client arrival timestamps, submission lag, per-request input/output lengths, token IDs, errors, end-to-end latency and allocator cache memory.

These are short synthetic workloads, not steady-state production load. Small-sample tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. Queued requests have no active KV cache; active batch rows do. Concurrency is client in-flight work; the actual batch-size column records model execution groups.

The largest successful context/concurrency is only the largest tested configuration. Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.

See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.
