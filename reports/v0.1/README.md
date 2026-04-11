# v0.1 — Qwen inference milestone

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

**Milestone:** single-prompt Qwen3-0.6B 4-bit inference on MLX, with greedy
decoding, KV caching, and benchmark reporting.

Code and report are archived together under the annotated Git tag `v0.1`.
The Python project version is `0.1.0`.

## What works

- Local quantized weight loading through MLX-LM, with `AutoTokenizer` and the
  Qwen chat template. Thinking mode is disabled.
- Our own generation loop: full-prompt prefill, then one-token cached decode,
  stopping at EOS or the output budget.
- An abstract `Sampler` interface with a `GreedySampler` implementation.
- Optional full-prefix recomputation (`--no-cache`) and shape tracing (`--trace`).
- Prefill/decode timing, token counts, peak MLX memory, and JSON result export.

Qwen3 layers, quantized weight loading, and KV-cache storage come from MLX-LM.
This milestone covers one request; batching, scheduling, and our own model-layer
implementation are future work.

## Recorded result

Prompt: `Count from 1 to 100, separated by commas.`

| Metric | Result |
| --- | ---: |
| Prompt length, including chat template | 26 tokens |
| Generated length, excluding EOS | 128 tokens |
| Decode steps | 127 |
| Prefill time | 0.026060292 s |
| Decode time | 0.866839050 s |
| Prefill throughput | 997.686 tok/s |
| Decode throughput | 146.509 tok/s |
| Peak MLX memory | 0.571457844 GB |

Generation reached the 128-token budget. The first output token came from
prefill, followed by 127 decode steps. The complete numeric result and
environment are in [qwen3-0.6b-4bit.json](qwen3-0.6b-4bit.json).

This archives the existing `benchmarks/cached-128.json` measurement. Its model
path has been made repository-relative; all recorded metrics are unchanged.

## Environment and method

- Apple M4, 16 GiB unified memory; macOS, arm64.
- Python 3.12.12.
- Model: `mlx-community/Qwen3-0.6B-4bit`, revision
  `73e3e38d981303bc594367cd910ea6eb48349da8`.
- Batch size 1; greedy selection; KV cache enabled; 4-bit weights with group
  size 64. The KV cache is not quantized.
- One measured request after a two-token warmup using a separate KV cache.
- Each timed forward pass includes token selection, scalar evaluation, and GPU
  synchronization. Loading, tokenization, input-array assembly, trace printing,
  and output text decoding are outside the timed regions.
- Memory is the MLX-reported peak during the measured request, including resident
  weights and KV cache. It excludes loading/warmup peaks and is not process RSS.

This is a single local baseline, not an average or a hardware-wide performance
claim. Short-prompt prefill throughput and repeated runs can vary.

## Reproduce at tag v0.1

```bash
uv sync --locked
uv run hf download mlx-community/Qwen3-0.6B-4bit \
  --revision 73e3e38d981303bc594367cd910ea6eb48349da8 \
  --local-dir models/Qwen3-0.6B-4bit
uv run python infer.py "Count from 1 to 100, separated by commas." \
  --max-tokens 128 --benchmark benchmarks/v0.1-recheck.json
```

New runs go into the ignored `benchmarks/` directory; this tracked report retains
the original v0.1 measurement.

## Validation

Pyright passed with zero errors and warnings. Local GPU checks covered ordinary
greedy inference, replacement sampler behavior, measured/unmeasured output
agreement, token counts and throughput calculations, EOS during prefill/decode,
zero/one-token budgets, statistics reset, and full-prefix recomputation.
