# Chunked prefill

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

`InferenceClient(prefill_chunk_size=256)` processes at most 256 padded prompt
positions per incoming row per scheduler iteration. Each intermediate chunk
updates/evaluates KV without a vocabulary projection or token notification.
Existing decode rows then advance one token. The last chunk samples the first
output token and merges surviving rows into decode.

This first implementation has one incoming group at a time, reserves slots for
prefilling requests, and admits the next group after the current prefill finishes.
The chunk limit is per row, not a global token budget. `None` retains whole-group
prefill. The option requires KV caching and disables per-request compute stats.
It also works with `max_batch_size=1`, where chunks have no other row to decode.

## Real-model comparison

Apple M4, Qwen3-0.6B 4-bit with original activation precision, last-position
vocabulary projection enabled for both backends. Two cases run three repeats
each, sequentially on a desktop without exclusive isolation:

- `prefill-interference`: a 128-token prompt generates 128 output tokens; a
  2048-token prompt requesting 16 output tokens arrives 250 ms later.
- `context-burst`: four simultaneous 2048-token prompts, 32 output tokens each.

EOS is ignored to keep output budgets fixed. Both backends have four reserved
request slots and a 2 ms idle collection wait. Source hashes, workload prompts,
model files and package versions match; only chunk configuration differs.

| Interference measurement | Whole prefill | Chunk size 256 |
|---|---:|---:|
| Old-request tokens generated during incoming prefill, each repeat | 0 | 7 |
| Median of each repeat's maximum old-request generated-token gap | 1742.88 ms | 518.34 ms |
| Median incoming-request TTFT | 1597.34 ms | 2266.68 ms |
| Median workload output throughput | 22.34 tok/s | 19.27 tok/s |
| Peak active MLX memory | 1.306 GB | 1.306 GB |

The maximum gap is measured over the old request's whole generated timestamp
sequence, then its per-repeat maximum is summarized by the median. Tokens during
prefill are old-request worker timestamps between the incoming request's worker
admission time and its first generated token. Seven tokens correspond to the
seven opportunities to decode between eight 256-position chunks.

Chunking reduced the observed interruption by about 70%, while increasing the
incoming request's wait and reducing interference-case throughput. It changes
scheduling responsiveness, not just total work. In `context-burst`, median
throughput was 14.41 -> 15.20 tok/s and peak active MLX memory 1.996 -> 1.839 GB.
These short runs do not establish a general speedup or a production tail bound.

## Correctness and numerical scope

All 50 unit/regression tests pass, including real tiny-Qwen KV/logit comparisons,
padding-only initial chunks, chunk sizes 1/2/3/7/10, complete output sequences,
merging, zero/one-token budgets, decode interleaving, middle-chunk failure recovery
and shutdown with an unfinished prefill and no decode rows. Type checking of the
changed runtime/test modules reports no errors.

Both real-model preflight suites pass all nine checks. Both performance backends
complete all 18 requests without errors. Full token sequences match for 15/18
paired outputs. The three differences are the old 128-output-token request in
the interference case, once per repeat, first diverging at output index 10.
Every arriving long prompt and all context-burst outputs match. Interleaving
changes the old request's batch composition over time; this experiment records
the differences without diagnosing their individual numerical cause or claiming
universal exact generation equivalence. Original activation precision is kept.

A separate prefill comparison uses the same real model and weights with prompt
lengths `[2048, 128, 512]`, padded to 2048, and independent whole/chunked caches.
The first seven chunks contain only padding for the 128-token row. Final-position
BF16 logits match exactly in this sample (maximum absolute difference 0), all
three argmax tokens match, offsets remain `[2048, 128, 512]`, and valid KV/logit
values are finite. This is a selected regression check, not a universal guarantee.

Artifacts:

- [Whole-prefill summary](whole/summary.md) and [raw observations](whole/results.json).
- [Chunk-256 summary](chunk-256/summary.md) and [raw observations](chunk-256/results.json).
- [Comparison and first token differences](comparison.json).
- [Mixed-length real-model logits check](logits.json).

Prior milestone reports and tags are unchanged.

## Reproduce

Run from the repository root with fresh output directories:

```bash
uv run python -m unittest discover -v
uv run python -m mlx_nano_llm.benchmarks.runner --batch-size 4 --case prefill-interference --case context-burst --repeats 3 --output benchmarks/prefill-whole
uv run python -m mlx_nano_llm.benchmarks.runner --batch-size 4 --prefill-chunk-size 256 --case prefill-interference --case context-burst --repeats 3 --output benchmarks/prefill-chunk-256
```

Code reading order: `BatchState.prefill_chunk()` -> `InferenceEngine.prefill_next()`
-> `InferenceWorker._run_continuous()` -> `tests/test_chunked_prefill.py` and
the chunked scheduling tests in `tests/test_client.py`.
