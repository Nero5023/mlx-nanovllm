# Fixed paged batches

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

This adds a single cache adapter, `PagedBatchKVCache`, and reuses `BatchState`
for token generation, EOS/budgets, callbacks and finished-row compaction. Each
layer borrows its persistent engine pool. Rows own independent page tables and
logical lengths. Padding tokens never enter physical KV pages; gather temporarily
left-pads valid histories for existing attention. Row filtering frees finished
rows and retains surviving tables without relocating their physical pages.

`InferenceEngine.run_batch()` selects a FIFO prefix whose total worst-case page
requirements fit every layer's pool. It reserves capacity logically for complete
generation rather than overcommitting and stalling mid-decode. The paged worker
collects a fixed group, then calls this path until the group is drained. Requests
joining during generation wait; final futures are delivered after the collected
group completes. An error fails that group and discards its engine-waiting tail,
while later work in the worker queue can still run.

## Verification

- 83 tests pass, with no type-check errors in changed runtime/batch tests.
- Tiny-Qwen tests compare ragged prefill and decode logits against contiguous
  caches, including removal of the longest row and mask/padding realignment.
- Tests cover complete generation, zero budgets, EOS, immediate row-page release,
  stable surviving tables, callback failure cleanup, shared pool reuse, capacity
  splitting and recovery after a failed group leaves accepted work waiting.
- The actual Qwen3-0.6B 4-bit model runs two batches through the async worker,
  with row orders `[0,1,2]` and `[2,1,0]`. Prompt lengths are `[63,47,65]`, output
  budgets `[8,16,4]`; EOS is ignored. Page size is 16, capacity 16 pages per layer.
- All six complete outputs match contiguous fixed-batch execution with the same
  row order. All layer pools keep their identities and return all 16 pages after
  each group. Dropping rows releases five pages in each recorded layer-0 event;
  surviving page tables remain unchanged. The worker stops normally on close.

[Raw real-model checks](validation.json) retain complete token IDs, release
events and free-page counts. This is a correctness sample, not
universal bitwise equivalence or an isolated performance benchmark. Physical
storage uses ordinary MLX indexing; attention still gathers contiguous arrays.

## Use

```python
async with InferenceClient(paged_kv=True, max_batch_size=4,
                           kv_page_size=16, kv_num_pages=256) as client:
    outputs = await asyncio.gather(
        client.generate("What is the capital of France?", 32),
        client.generate("Count from 1 to 10.", 64),
    )
```

`collect_stats=False` and `use_cache=True` are required for batches. Paged
chunked prefill and dynamic row admission are not enabled in this step. The
changes, including the preceding serial pool integration, remain uncommitted.
