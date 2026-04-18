# continuous-batch-worker benchmark

Correctness: **PASS**. Chip: Apple M4. Max active requests: 4, batch wait: 2.0 ms, KV cache=True, prefill chunk size=None (None means whole prompt).

TTFT includes queueing and client notification. ITL pools consecutive token gaps; TPOT averages gaps per request. Values are milliseconds. Throughput is the median of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.

| Case | Runs | OK / all | Admission group sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| prefill-interference | 3 | 6 / 6 | 1 | 847.34 / 1639.11 | 37.83 / 73.73 | 61.36 | 11.96 | 22.34 | 1.306 |
| context-burst | 3 | 12 / 12 | 4 | 6147.72 / 6300.47 | 73.82 / 102.86 | 83.36 | 0.03 | 14.41 | 1.996 |

Raw JSON includes p99 and sample counts, worker token timestamps, client arrival timestamps, submission lag, per-request input/output lengths, token IDs, errors, end-to-end latency and allocator cache memory.

These are short synthetic workloads, not steady-state production load. Small-sample tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. Queued requests have no active KV cache; active batch rows do. Concurrency is client in-flight work; admission group sizes do not track growing/shrinking decode tensors.

The largest successful context/concurrency is only the largest tested configuration. Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.

See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.
