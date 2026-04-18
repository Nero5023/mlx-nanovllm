# Last-position vocabulary projection

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

After v0.4, the local `LastTokenQwen3` subclass runs the complete transformer,
then slices `hidden[:, -1:, :]` before the tied embedding or untied LM head.
Loaded inference models return `[batch, 1, vocabulary]` logits. Prompt processing,
KV updates, masks and position offsets are unchanged. This model is for generation,
not training or scoring every prompt position. MLX-LM dependency files are unchanged.

## Targeted validation

Apple M4, local Qwen3-0.6B 4-bit weights, original BF16 activations. Four 2048-token
prompts each generate 32 tokens with EOS ignored. The baseline uses the upstream
full-projection model through the existing worker and `run_case()` path; the
optimized result uses the production CLI and local loader. Runs are sequential
on the desktop, with identical 128-token/four-output-token warmup and allocator
reset policy. Each measurement has one repeat; no stable timing speedup is claimed.

| Check | Result |
|---|---:|
| Baseline peak active MLX memory | 4.435 GB |
| Optimized peak active MLX memory | 1.996 GB |
| Peak reduction | 54.99% |
| Complete 32-token sequences matching baseline | 4 / 4 |
| Prefill final-position logits max absolute difference | 0.125 |
| Prefill final-position logits mean absolute difference | 0.006735 |
| First-token argmax matching baseline | 4 / 4 |

The logits comparison reuses the same weights, runs full projection and the
local subclass's projection with independent fresh KV caches, then compares
the final-position logits in float32. The weights and transformer precision
are not changed. Different projection shapes can choose different numerical
kernels; these results do not guarantee bitwise logits or identical generation
for every prompt. Passing preflight checks is not a universal equivalence claim.

The full BF16 logits tensor would contain `4 * 2048 * 151936` elements, about
2.489 GB. Last-position projection produces `4 * 1 * 151936` elements, about
1.22 MB. Total peak memory still includes weights, KV and transformer temporaries.

Artifacts:

- [Baseline raw run](baseline-run.json), using v0.4's upstream full projection.
- [Optimized summary](optimized/summary.md) and [raw results](optimized/results.json),
  including model hashes and all nine passing real-model preflight checks.
- [Comparison, logits differences and generation checks](comparison.json).

Historical v0.4 reports and its release tag remain unchanged.

## Reproduce

From the repository root:

```bash
uv run python -m unittest discover -v
uv run python -m mlx_nano_llm.benchmarks.runner --batch-size 4 --case context-burst --repeats 1 --output benchmarks/last-token-projection
```

Use a fresh output directory. To measure the original implementation, run the
same benchmark command from a separate checkout of `v0.4` with the same model.
The three new regression tests cover tied and untied heads, padded prefill and
decode logits, KV contents and offsets, complete cached/uncached generation,
and batched generation with padding and compaction. All 42 tests pass.
