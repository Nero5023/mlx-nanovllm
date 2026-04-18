# chunked-prefill-worker benchmark

Correctness: **PASS**. Chip: Apple M4. Max active requests: 4, batch wait: 2.0 ms, KV cache=True, prefill chunk size=256 (None means whole prompt).

TTFT includes queueing and client notification. ITL pools consecutive token gaps; TPOT averages gaps per request. Values are milliseconds. Throughput is the median of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.

| Case | Runs | OK / all | Admission group sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| prefill-interference | 3 | 6 / 6 | 1 | 1171.98 / 2275.48 | 45.13 / 208.52 | 62.85 | 3.53 | 19.27 | 1.306 |
| context-burst | 3 | 12 / 12 | 4 | 5669.37 / 6100.83 | 64.07 / 111.22 | 85.12 | 0.02 | 15.20 | 1.839 |

Raw JSON includes p99 and sample counts, worker token timestamps, client arrival timestamps, submission lag, per-request input/output lengths, token IDs, errors, end-to-end latency and allocator cache memory.

These are short synthetic workloads, not steady-state production load. Small-sample tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. Queued requests have no active KV cache; active batch rows do. Concurrency is client in-flight work; admission group sizes do not track growing/shrinking decode tensors.

The largest successful context/concurrency is only the largest tested configuration. Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.

See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.
