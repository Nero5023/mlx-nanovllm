# Single-request paged KV with gather + existing attention

> Dependency version metadata is omitted from these archived measurements.
> They have not been rerun with this checkout's dependency lock. Compatibility
> checks do not update the archived performance results.

This step connects paged storage to `generate_tokens()` and the direct inference
CLI. `PagedKVAdapter` implements the cache interface used by Qwen3: `offset`,
`make_mask()` and `update_and_fetch()`. New K/V are appended to pages, valid
history is gathered into `[1, KV heads, tokens, head dim]`, then the existing
attention implementation runs. RoPE uses the sequence's logical token count.

Each request creates private pools, one per layer, initialized on first write
with the real activation dtype. Capacity is rounded to whole pages for
`prompt_length + max_tokens - 1` inputs. No pool is allocated for zero output
budget. A `finally` block returns pages on success, EOS, callback errors and model
errors. This does not introduce a shared serving pool or change batch scheduling.

## Real-model validation

The local Qwen3-0.6B 4-bit model retains its original BF16 activation precision
and last-position vocabulary projection. Page size is 16 tokens. Tests use
formatted prompt lengths on both sides of a page boundary and fixed 32-token
output budgets (EOS ignored).

| Prompt tokens | Output tokens | Max logits difference over 32 same-history forwards | Argmax matches | Complete output match | Every layer's K/V match |
|---:|---:|---:|---:|---|---|
| 63 | 32 | 0 | 32/32 | Yes | Yes |
| 64 | 32 | 0 | 32/32 | Yes | Yes |
| 65 | 32 | 0 | 32/32 | Yes | Yes |

Full generation runs independently with contiguous and paged caches. The logits
comparison uses separate caches and teacher-forces the contiguous run's tokens
into both paths so every forward sees identical input history. It checks logits,
finite values, per-layer offsets and bitwise K/V contents after each step. Final
offsets are 94, 95 and 96; six pages per layer are used, then explicitly released.
These are selected regression cases, not universal numerical equivalence claims.

[Raw validation](validation.json) includes complete token IDs and per-case results. Earlier milestone artifacts remain unchanged.

## Tests and usage

The suite has 67 passing tests. New integration tests cover reference-cache
layout and masks, multi-token continuation with existing KV, real tiny-Qwen
logits/KV, complete generation against contiguous and no-cache paths, page sizes
1/2/4/16, EOS and error cleanup, zero-budget requests, statistics, batch rejection
and capacity exhaustion. Storage tests also exercise physically non-contiguous
page mappings and lazy read lifetimes across page reuse.

```bash
uv run python -m unittest discover -v
uv run python -m mlx_nano_llm.infer "What is the capital of France?" --paged-kv --kv-page-size 16 --max-tokens 32 --trace
```

`--paged-kv` conflicts with `--no-cache`; the default remains contiguous caching.
`--benchmark` records `kv_cache_backend=paged-gather` and the page size, including
the same backend in warmup. Actual CLI smoke testing with page size 4 also ran
prefill and seven decode steps and produced the expected answer about Paris.

This path materializes contiguous K/V at every attention call and uses ordinary
MLX indexing for page writes. It is for validation, with no zero-copy or speedup
claim. Fixed reserved pool memory is distinct from used-page counts. The serving
engine's shared lifecycle, batched/chunked cache adapters and the Metal kernel
that reads physical pages directly are outside this step.
