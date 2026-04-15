# Serving benchmark

This suite measures the async client → worker path on the local Qwen model.
The default is **one active request**; `--batch-size N` admits groups of up to N.
Finished rows now compact during decode. No new rows join an active group yet,
and there is no cross-request prefix reuse or paged KV cache.
Workloads stay separate from scheduling so the same requests compare each stage.

```bash
# All main scenarios, three repeats each.
uv run python bench.py --output benchmarks/fifo-baseline

# Same workloads, groups of at most four with compaction. Collection wait: 2 ms.
uv run python bench.py --batch-size 4 --output benchmarks/fixed-batch-4

# Add adjacent-length cases for allocation/kernel changes.
uv run python bench.py --profile full --repeats 3 \
  --max-concurrency 4 --max-prompt-tokens 2048 \
  --output benchmarks/fifo-full

# Select a smaller comparison, or increase the bounded sweep.
uv run python bench.py --case burst-1 --case burst-4
uv run python bench.py --max-concurrency 8 --max-prompt-tokens 4096
uv run python bench.py --case single --no-cache

uv run python -m unittest -v test_engine test_client test_bench test_batch_infer
```

Output directories contain `summary.md` and `results.json`. Existing results
are protected from accidental overwrite. `benchmarks/` is ignored by Git;
explicit milestone reports belong in `reports/`. Completed cases are saved as
the suite proceeds. A failed correctness check or request makes the command
exit unsuccessfully, and errors remain in the JSON.

## What the measurements mean

Every timestamp uses `time.perf_counter()` in the same process. The worker emits
a token event after `.item()` has evaluated the MLX graph. The async client
delivers that event on its event loop. This measures local token notification,
including event-loop delays; there is no HTTP transport or incremental text
detokenization. The final response still contains the complete decoded text.

| Metric | Formula / scope |
|---|---|
| Client TTFT | First token notification received − request submitted. Includes queueing, tokenization, prefill, sampling, and notification delay. |
| ITL | Time between consecutive token notifications. Pool token gaps for p50/p95/p99, so longer outputs contribute more samples. |
| TPOT | (Last token received − first token received) / (output tokens − 1). One sample per request. |
| Queue time | Worker starts handling request − worker accepts it into its queue. |
| Service TTFT | First token generated − worker starts handling request. Includes tokenization; **not pure prefill compute time**. |
| Worker TTFT / ITL | The same observations at generation time, before client callback delivery. |
| Delivery delay | Client receives token notification − worker generates token. |
| E2E latency | Final response received − submitted; includes final text decoding. |
| Output throughput | Total output tokens from successful requests / (last completion − first submission), per case/repeat. Includes idle arrival gaps and failed-request time. |
| Request throughput | Successful requests / that same whole workload window. |
| Peak memory | MLX peak active allocation, including resident weights, KV, and intermediates. Reset once after warmup and read after all case requests finish. |

Zero-token outputs have no TTFT. Zero/one-token outputs have no TPOT or ITL;
these values are `null`, never fabricated zeros. The JSON records sample counts,
actual prompt/output lengths, planned and actual arrivals, every token timestamp,
output IDs, errors, and the number of client requests in flight.

Summary tables pool request/token latency samples across repeats, report the
median of **per-repeat** throughput, and the maximum observed peak memory.
Per-repeat values remain available to see variation. Three repeats and short
bursts are useful for development; their tail percentiles are descriptive,
not a production SLO or a steady-state capacity measurement.

The report records both client concurrency and **admitted group sizes**.
Four in-flight requests can form smaller batches if their arrivals are staggered.
`--batch-wait-ms` bounds the collection wait (default 2 ms) and contributes to
queue time. Groups never refill during generation. Finished rows leave the active
tensors and KV cache, but their final response waits for the whole group; E2E
latency therefore includes this wait. `timing.batch_size` (and the legacy JSON
name `actual_batch_sizes`) describes the admitted group, not each decode step's
live row count. Trace output shows current rows. Metadata records compaction so
new results are distinguishable from the archived fixed-batch v0.3 measurements.

MLX's [peak-memory counter](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)
excludes unused buffers retained in the allocator cache. We record allocator
cache bytes separately at case boundaries. These are not process RSS or total
physical-memory usage. MLX counters are process-wide: run one backend at a time
and keep the older per-request `collect_stats` disabled during this suite.

The previous `infer.py --benchmark` still measures prefill and decode compute
regions. Its throughput excludes serving overhead and uses a different scope;
compare equivalent commands/workloads when assessing regressions.

## Scenarios along the roadmap

| Stage | Cases | What to compare |
|---|---|---|
| Single-request correctness | Cached vs full-prefix recompute on two prompts; A/B/A isolation; zero/one output budget | Exact greedy token IDs and request state isolation. These are regression checks, not an independent model-quality oracle. |
| Fixed batches | `single`, `burst-1/2/4/...` with equal 128-token prompts and 64-token outputs | Total throughput vs TTFT/ITL and active memory at increasing client concurrency. |
| Continuous batching | `mixed-burst`, `staggered` | Long request first, short requests behind it; new requests arrive every 50 ms during generation. Compare queue time, TTFT tails, throughput and per-request fairness. |
| Prefix reuse | `prefix-shared`, `prefix-control` | Four sequential requests: first cold, then related requests. Same lengths; control changes a token near the beginning. Inspect first request separately from follow-ups. |
| Paging / context memory | `context-*`, `context-burst` | Context-length sweep and simultaneous long-context requests. Compare active memory and latency at identical input/output work. |
| Allocation / kernel optimization | `boundary-127/128/129`, `255/256/257`, `511/512/513` in `full` | Nearby lengths and decode growth across boundaries. These are probes, not a claim about a future engine's page size; add the actual page/kernel boundaries when known. |

With fixed batching enabled, preflight checks additionally compare a mixed group
against serial greedy inference, reverse request order to test row mapping and
cache isolation, and verify an underfilled batch. Different kernel shapes can
produce floating-point differences; a token mismatch is reported as a failed
check for investigation, never silently accepted as correct.

Passing these selected checks does not establish bitwise equality on all prompts.
In BF16, serial and batched kernels can round nearby logits differently; the
synthetic performance prompts have exhibited greedy token differences. Fixed
output budgets still control work, but token-by-token equivalence must be
reported separately from performance. Investigate first-divergence logits on
identical histories before attributing a difference to state handling or rounding.

Prompt lengths include the Qwen chat template and are checked against the real
tokenizer. Synthetic prompts use repeated filler; they control token counts
but do not represent the diversity of production text. Shared prefix length is
measured on **token IDs**, not estimated from characters. The first shared-prefix
request and the first control request are identical; later control requests
change an early token, while shared requests only change a suffix.

Performance cases use greedy decoding with `ignore_eos=True` so each request
performs its entire stated output budget; EOS predictions count as generated
tokens in these cases. Correctness checks use normal EOS behavior. Model load
is excluded. Each case/repeat gets a fresh client, then an unrelated 128-token,
4-output-token warmup before measurement. This warms the common inference path,
not every possible sequence shape. After warmup, unused MLX allocator buffers
are released before resetting the peak and starting the case. Live model/KV
allocations remain. This prevents a previous large batch's unused buffers from
dominating later small cases. Prefix state starts cold; compiled kernels can
survive between cases. `metadata.allocator_policy` records this measurement
change; rerun the serial baseline with the same policy for timing comparisons.

Burst/staggered cases submit according to the schedule **without waiting for
earlier responses**. The prefix cases intentionally wait for each response so
a future prefix cache is populated before the next request. Planned vs actual
submission lag exposes event-loop stalls; don't silently treat delayed arrivals
as the intended offered load.

The concurrency sweep is client in-flight requests, **not active GPU batch size**.
Queued requests have no KV caches. Active batch rows each occupy KV storage,
including left padding. Compaction filters finished rows before the next forward
pass; the allocator may retain freed buffers and filtering itself can allocate. The
serial path's queue memory cannot demonstrate this simultaneous active-context cost.
The largest successful tested configuration is only a tested bound, not the
machine's limit. Increase `--max-concurrency` and `--max-prompt-tokens` deliberately;
this suite does not search until OOM. Throughput is a useful comparison, but
GPU utilization is not directly measured and is recorded as `null`.

## Code reading order and future backends

1. `bench_workloads.py`: `Case` / `RequestSpec` and deterministic arrivals.
2. `bench.py`: `run_case()` sends requests and records observations.
3. `bench_metrics.py`: formulas and percentile calculations.
4. `telemetry.py`, then `client.py` / `worker.py` / `infer.py`: timestamp path.

`BenchmarkBackend` needs `generate(..., on_token=..., ignore_eos=...)` and
`memory_snapshot(...)`. Future batching engines can implement this protocol
and use the same `run_case()`. They must emit ordered token notifications on the
client event loop before completing the response. Omit unavailable worker timing
instead of assigning made-up queue times. Keep memory measurement at case scope.

For each later milestone, preserve the model/tokenizer hashes, sampler, prompt
IDs, output budgets, arrivals, warmup policy, and environment. Compare each
request's stored output IDs with this baseline before interpreting speedups;
the current cache/full-recompute check alone cannot validate a new batch engine.
Admitted group sizes are recorded. Add genuine prefix hits/recomputed
tokens and page occupancy once those implementations exist. Raw source hashes
record the code even before a commit.

A useful reading exercise: if `burst-4` has nearly unchanged tok/s but much
higher TTFT than `burst-1`, which part of `queue → prefill → decode` grew?
