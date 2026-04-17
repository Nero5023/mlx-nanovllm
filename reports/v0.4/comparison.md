# Serial vs continuous batching

Apple M4, Qwen3-0.6B 4-bit, original activation precision. Both backends use identical workloads, model files and package versions. Twelve scenarios run three times each; serial and continuous runs execute sequentially on the desktop, without exclusive machine isolation.

Throughput is the median of three workload-window rates. TTFT pools request samples; peak memory is maximum active MLX allocation across repeats. Output budgets are fixed (`ignore_eos=True`). Warmup and allocator policy are identical. These short synthetic runs are not a production capacity measurement.

| Case | Serial tok/s | Continuous tok/s | Speedup | TTFT p50 ms, serial / continuous | Peak GB, serial / continuous |
|---|---:|---:|---:|---:|---:|
| single | 28.84 | 22.27 | 0.77× | 161.02 / 185.87 | 0.667 / 0.671 |
| burst-1 | 28.96 | 23.39 | 0.81× | 154.59 / 147.89 | 0.667 / 0.671 |
| burst-2 | 24.68 | 43.98 | 1.78× | 1323.14 / 265.28 | 0.667 / 0.780 |
| burst-4 | 23.65 | 77.97 | 3.30× | 3504.84 / 360.25 | 0.667 / 1.028 |
| mixed-burst | 20.62 | 19.65 | 0.95× | 6719.41 / 3209.08 | 1.427 / 2.582 |
| staggered | 25.28 | 30.05 | 1.19× | 5360.17 / 409.12 | 0.667 / 0.883 |
| prefix-shared | 20.97 | 22.04 | 1.05× | 352.13 / 346.29 | 0.980 / 0.980 |
| prefix-control | 20.18 | 19.91 | 0.99× | 365.08 / 390.31 | 0.980 / 0.980 |
| context-128 | 27.98 | 21.51 | 0.77× | 143.08 / 159.63 | 0.667 / 0.671 |
| context-512 | 21.15 | 20.49 | 0.97× | 394.73 / 345.56 | 0.980 / 0.980 |
| context-2048 | 10.69 | 9.03 | 0.84× | 1593.92 / 1863.47 | 1.839 / 1.801 |
| context-burst | 10.18 | 12.14 | 1.19× | 5975.40 / 7724.08 | 1.839 / 4.435 |

## Correctness and scope

Both preflight suites passed. Each backend completed 93 performance requests without request errors. Exact token outputs matched for 72/93 requests; 21 differed. First divergence positions are preserved in [comparison.json](comparison.json), and full token IDs are in the raw results. Passing selected preflight checks does not establish bitwise equivalence for every workload.

Continuous batching admits new requests between decode steps, compacts finished rows and delivers individual results immediately. Incoming prefill still pauses existing decode. There is no chunked prefill, prefix reuse or paged KV cache. Different serial/batch tensor shapes can change numerical results; this run does not diagnose individual token differences.

This comparison is between v0.4 serial and v0.4 continuous backends, not a controlled comparison against the older v0.3 archived machine timings.

See [the release report](README.md) for commands and [BENCHMARKS.md](../../BENCHMARKS.md) for metric definitions.
