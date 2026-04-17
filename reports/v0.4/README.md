# v0.4: continuous batching and project organization

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

This release allows waiting requests to join active decode between steps. New
requests prefill separately, merge their padded KV histories into the active
batch, and receive results individually when complete. Finished rows compact
out of the batch. Whole-group prefill still pauses decode; there is no chunked
prefill, prefix reuse or paged KV cache.

Runtime code now lives under `mlx_nano_llm/`, the serving benchmark under
`mlx_nano_llm/benchmarks/`, usage examples under `examples/`, and tests under
`tests/`. Models and local benchmark output keep their existing directories.
The default model path still resolves to the repository's `models/` directory.

## Benchmark artifacts

Read the [serial vs continuous comparison](comparison.md) for all 12 workload
scenarios. Each backend runs three repeats on the same Apple M4 with the local
Qwen3-0.6B 4-bit model. The runs execute sequentially on a desktop; the machine
is not exclusively isolated for benchmarking. Results characterize these runs,
not a hardware capacity limit or a controlled speed comparison to older tags.

- [Serial summary](serial/summary.md) and [raw results](serial/results.json).
- [Four-slot continuous summary](continuous-batch-4/summary.md) and
  [raw results](continuous-batch-4/results.json).
- [Comparison metrics and exact-token differences](comparison.json).

Raw results preserve model hashes, token timing,
per-request token IDs, errors and memory measurements. Historical
reports under `reports/fixed-batching/` remain unchanged.

## Measured results

| Workload | Serial tok/s | Continuous tok/s | Ratio |
|---|---:|---:|---:|
| burst-4 | 23.65 | 77.97 | 3.30× |
| mixed-burst | 20.62 | 19.65 | 0.95× |
| staggered | 25.28 | 30.05 | 1.19× |
| context-burst | 10.18 | 12.14 | 1.19× |

Staggered-request median TTFT fell from 5360 ms to 409 ms. Mixed-length throughput was slightly lower; long-context burst peak active memory rose from 1.839 GB to 4.435 GB. Gains depend on the workload, and whole-group prefill still blocks existing decode.

Both real-model preflight suites passed. Each backend completed 93 performance requests with no request failures. Exact token sequences matched for 72/93 paired requests; 21 differed. Full outputs and first differing token positions are retained. Selected checks passing does not guarantee bitwise equivalence for every prompt. The original activation precision is unchanged.

## Reproduce

From the repository root, with model files already downloaded:

```bash
uv sync --locked
uv run python -m unittest discover -v
uv run python -m mlx_nano_llm.benchmarks.runner --batch-size 1 --profile quick --repeats 3 --output benchmarks/v04-serial
uv run python -m mlx_nano_llm.benchmarks.runner --batch-size 4 --profile quick --repeats 3 --output benchmarks/v04-continuous-batch-4
```

Use fresh output directories. Both commands use 2 ms idle collection wait,
maximum prompt length 2048 and maximum client concurrency 4. The serial path
processes one request at a time. Each case has its own client/model, warmup and
allocator reset. See [BENCHMARKS.md](../../BENCHMARKS.md) for metric definitions.

The 39 unit/regression tests cover scheduling, failure recovery, cancellation,
shutdown, per-request outputs, compaction, cache merging and tiny-Qwen logits.
Real-model benchmark preflight additionally checks serial/cache equivalence,
request isolation, mixed/reordered/partial groups and continuous refill.
