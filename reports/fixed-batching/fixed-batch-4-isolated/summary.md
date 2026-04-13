# fixed-batch-worker benchmark

Correctness: **PASS**. Chip: Apple M4. Max active requests: 4, batch wait: 2.0 ms, KV cache=True.

TTFT includes queueing and client notification. ITL pools consecutive token gaps; TPOT averages gaps per request. Values are milliseconds. Throughput is the median of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.

| Case | Runs | OK / all | Actual batch sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| single | 3 | 3 / 3 | 1 | 102.49 / 102.78 | 9.43 / 14.88 | 10.89 | 2.55 | 81.17 | 0.671 |
| burst-1 | 3 | 3 / 3 | 1 | 104.37 / 104.70 | 9.39 / 14.94 | 10.77 | 2.55 | 81.76 | 0.671 |
| burst-2 | 3 | 6 / 6 | 2 | 172.24 / 182.88 | 10.78 / 15.62 | 11.78 | 2.55 | 139.60 | 0.780 |
| burst-4 | 3 | 12 / 12 | 4 | 317.14 / 322.76 | 14.68 / 16.98 | 14.20 | 0.03 | 211.27 | 1.028 |
| mixed-burst | 3 | 12 / 12 | 4 | 2905.20 / 2985.44 | 17.51 / 22.22 | 20.13 | 0.03 | 33.53 | 2.582 |
| staggered | 3 | 12 / 12 | 1, 3 | 1592.82 / 1674.27 | 10.23 / 15.23 | 12.84 | 1367.04 | 92.19 | 0.853 |
| prefix-shared | 3 | 12 / 12 | 1 | 318.37 / 328.23 | 10.85 / 15.52 | 11.79 | 2.55 | 46.79 | 0.980 |
| prefix-control | 3 | 12 / 12 | 1 | 316.66 / 331.69 | 10.98 / 15.67 | 11.96 | 2.55 | 46.15 | 0.980 |
| context-128 | 3 | 3 / 3 | 1 | 105.34 / 107.03 | 9.79 / 14.75 | 11.02 | 2.54 | 71.49 | 0.671 |
| context-512 | 3 | 3 / 3 | 1 | 320.84 / 323.13 | 11.17 / 15.88 | 12.02 | 2.40 | 46.15 | 0.980 |
| context-2048 | 3 | 3 / 3 | 1 | 1748.14 / 1790.84 | 15.73 / 20.20 | 15.99 | 2.54 | 14.31 | 1.801 |
| context-burst | 3 | 12 / 12 | 4 | 6525.40 / 6532.30 | 33.94 / 38.42 | 36.63 | 0.02 | 16.71 | 4.435 |

Raw JSON includes p99 and sample counts, worker token timestamps, client arrival timestamps, submission lag, per-request input/output lengths, token IDs, errors, end-to-end latency and allocator cache memory.

These are short synthetic workloads, not steady-state production load. Small-sample tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. Queued requests have no active KV cache; active batch rows do. Concurrency is client in-flight work; the actual batch-size column records model execution groups.

The largest successful context/concurrency is only the largest tested configuration. Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.

See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.
