# Serial paged KV serving and pool reuse

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

This step adds `paged_kv`, `kv_page_size` and `kv_num_pages` to the engine, worker
and async client. The engine creates one persistent adapter per model layer,
with lazily initialized pools. `run_next()` passes those adapters to
`generate_tokens()`, which checks that every layer starts empty and has enough
free capacity before any K/V write. Its existing `finally` block returns the
request's pages, retaining the pool for later requests.

Serving is deliberately serial: `max_batch_size=1`, caching enabled, no chunked
prefill. Batched/chunked options are rejected rather than silently using the
contiguous backend. Multiple async callers may enqueue work, but requests execute
one at a time. This does not add prefix caching or direct paged attention;
attention still gathers valid history before existing SDPA.

## Real-model validation

The actual async client and worker load local Qwen3-0.6B 4-bit weights with original
activation precision. One engine processes A -> B -> A, each with a fixed
32-token output budget (EOS ignored). A has 63 formatted prompt tokens and B
has 65. Every layer has 16 pages of 16 tokens, capacity 256 tokens.

| Request | Prompt tokens | Final offset | Layer 0 physical page table | Complete output vs contiguous |
|---|---:|---:|---|---|
| A | 63 | 94 | `[0,1,2,3,4,5]` | Match |
| B | 65 | 96 | `[6,7,8,9,10,11]` | Match |
| A again | 63 | 94 | `[12,13,14,15,0,1]` | Match |

All 28 layer pools keep the same object identity across requests. Each request
starts at offset 0; after returning, all layer offsets are 0 and every pool has
all 16 pages free. The final request crosses the end of the free-page queue and
reuses physical pages non-contiguously without changing logical token order.
The two A outputs match each other as well as contiguous inference.

A preceding 257-token prompt requesting one output token fails capacity admission
without consuming a request ID or stopping the worker. The valid requests retain
FIFO IDs `[0,1,2]`. Closing the client stops the worker after draining its work.

[Raw validation](validation.json) records every layer's offsets and free counts,
pool-identity checks and complete token IDs. This was a correctness
run, not an isolated timing or memory-efficiency benchmark. Reserved pools can
use more memory than a short request needs; no throughput or zero-copy claim is
made.

## Regression checks

76 tests pass. New tests cover A/B/A pool reuse, non-contiguous page reuse,
capacity rejection before engine queueing, zero-output requests, EOS, callback failure,
partial-layer model failure and recovery, FIFO service, pre-admission and active
caller cancellation, and shutdown. Shared-cache preflight checks also reject
insufficient capacity across layers before any write, and refuse an already
occupied cache without clearing its existing history.

Cancellation preserves the worker's existing policy: queued requests can be
skipped; admitted requests finish computing after their caller cancels, and pages
are returned at completion. Failed inference returns pages before the next request.
Pools belong to the engine, so free-page counts describe allocator availability,
not memory returned to the OS.

```bash
uv run python -m unittest discover -v
```

See `tests/test_paged_serving.py` for the serial engine/client examples. The
earlier storage and single-request validation remain archived unchanged. This
implementation builds on the storage and single-request reference path.
