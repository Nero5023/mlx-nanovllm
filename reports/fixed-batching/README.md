# Fixed batching benchmark archive

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

This commit adds fixed-batch serving to the async client/worker engine. The
preceding serial-worker milestone is tagged `v0.2` at `652f7c4`.

Read the [serial vs fixed-batch comparison](fixed-batch-4-isolated/comparison.md)
for the complete workload table and numerical limitations. Both backends ran
12 scenarios, three repeats each, on the same Apple M4 and Qwen3-0.6B 4-bit model.
Each backend completed 93 performance requests with no request failures.

| Workload | Serial output tok/s | Fixed-batch output tok/s | Speedup |
|---|---:|---:|---:|
| Four equal-length requests | 81.58 | 211.27 | 2.59× |
| Mixed prompt/output lengths | 57.41 | 33.53 | 0.58× |
| Four long-context requests | 17.03 | 16.71 | 0.98× |

These are medians of three workload-window throughput measurements. Fixed
batches retain padded and finished rows, so they can waste computation. The
four equal-length requests increased peak active MLX memory from 0.667 to
1.028 GB. This experiment does not measure GPU utilization or capacity limits.

The selected real-model preflight checks and 32 regression tests passed. Exact
comparison across the performance workloads matched 75 of 93 request outputs;
18 differed. A controlled logit comparison reproduced two BF16 rounding/tie
differences that disappeared in FP32. This does not establish the cause of
every mismatch, and the implementation does not guarantee bitwise equivalence
between serial and batched inference. Default serving precision is unchanged.

Archived artifacts:

- [Serial report](serial-isolated/summary.md) and [raw observations](serial-isolated/results.json).
- [Fixed-batch report](fixed-batch-4-isolated/summary.md) and [raw observations](fixed-batch-4-isolated/results.json).
- [Comparison data and first differing token IDs](fixed-batch-4-isolated/comparison.json).
- [BF16/FP32 numerical investigation](batch-numerics.json).

The raw files are unchanged copies of the local benchmark results. Their
metadata preserves the environment, source/model hashes and Git state at
measurement time. The only Python source change afterward was a docstring
correction in `batch_infer.py`: “tensor shape” became “the batch dimension”.
Unused allocator buffers were cleared after warmup before each measured case
for both backends; the older interrupted run is not included in this archive.

Reproduce from the repository root using fresh output directories:

```bash
uv run python bench.py --batch-size 1 --profile quick --repeats 3 --output benchmarks/new-serial
uv run python bench.py --batch-size 4 --profile quick --repeats 3 --output benchmarks/new-fixed
```

See [BENCHMARKS.md](../../BENCHMARKS.md) for metric definitions and measurement scope.
