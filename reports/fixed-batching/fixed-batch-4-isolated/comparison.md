# Fixed batch vs serial baseline

Both backends use the same Qwen3-0.6B 4-bit weights, BF16 activations, tokenizer, exact prompt IDs, output budgets, and arrival schedules. Each runs 12 cases × 3 repeats. Unused allocator buffers are cleared after warmup before every case. Throughput is the median of three workload-window rates; TTFT pools request samples; peak memory is the maximum MLX active allocation across repeats.

| Case | Actual batch sizes | Serial tok/s | Batch tok/s | Speedup | TTFT p50 ms, serial / batch | Peak GB, serial / batch |
|---|---|---:|---:|---:|---:|---:|
| single | [1] | 77.33 | 81.17 | 1.05× | 108.35 / 102.49 | 0.667 / 0.671 |
| burst-1 | [1] | 79.56 | 81.76 | 1.03× | 102.54 / 104.37 | 0.667 / 0.671 |
| burst-2 | [2] | 81.09 | 139.60 | 1.72× | 497.20 / 172.24 | 0.667 / 0.780 |
| burst-4 | [4] | 81.58 | 211.27 | 2.59× | 1252.97 / 317.14 | 0.667 / 1.028 |
| mixed-burst | [4] | 57.41 | 33.53 | 0.58× | 2510.69 / 2905.20 | 1.427 / 2.582 |
| staggered | [1, 3] | 78.03 | 92.19 | 1.18× | 1620.08 / 1592.82 | 0.667 / 0.853 |
| prefix-shared | [1] | 48.12 | 46.79 | 0.97× | 296.83 / 318.37 | 0.980 / 0.980 |
| prefix-control | [1] | 47.92 | 46.15 | 0.96× | 296.48 / 316.66 | 0.980 / 0.980 |
| context-128 | [1] | 71.85 | 71.49 | 0.99× | 102.09 / 105.34 | 0.667 / 0.671 |
| context-512 | [1] | 45.65 | 46.15 | 1.01× | 308.02 / 320.84 | 0.980 / 0.980 |
| context-2048 | [1] | 16.47 | 14.31 | 0.87× | 1430.30 / 1748.14 | 1.839 / 1.801 |
| context-burst | [4] | 17.03 | 16.71 | 0.98× | 4327.01 / 6525.40 | 1.839 / 4.435 |

Interpretation:

- Four short equal-length requests share forward passes and improve throughput. Long/short mixtures waste work in padding and finished rows; they can be slower than serial execution. This is fixed batching, with no slot replacement.
- Staggered arrivals form batches of 1 then 3: new requests wait for the active group. Final response futures complete together at batch end, even when a row stopped emitting earlier.
- Larger active memory is expected with multiple live KV rows and padded intermediate tensors. GPU utilization and machine capacity limits are not measured.

Correctness and numerical scope:

- 32 unit/regression tests pass, including real tiny-Qwen prefill/decode logit comparisons with different prompt lengths. The real-model preflight checks for mixed budgets, reordered rows, partial batches, cache/full recomputation and isolation also pass.
- Exact comparison of all performance outputs: 75/93 request outputs match serial; 18 differ. Differences are retained in comparison.json. Therefore these timings do not imply bit-for-bit output equivalence.
- A controlled 2-row, 128-token experiment compared 16 steps on identical teacher-forced histories. Two BF16 argmax differences came from nearby scores rounding to ties: 22.25/22.125 became 22.25/22.25, and 15.4375/15.375 became 15.5/15.5. FP32 reproduced neither divergence, with maximum absolute logit difference 0.000114. This diagnoses those examples, not every possible mismatch. Details: ../batch-numerics.json.
- Default serving precision remains unchanged. Performance uses ignore_eos=True so all output budgets execute despite token differences. Fixed lengths control tensor work; they are not a model-quality evaluation.

The earlier benchmarks/fixed-batch-4 run was interrupted after unused allocator buffers accumulated to about 11.7 GB and timings became unstable. It is preliminary, not the comparison baseline. The isolated runs above both use the updated allocator policy.

Reproduce:

```bash
uv run python bench.py --batch-size 1 --profile quick --repeats 3 --output benchmarks/new-serial
uv run python bench.py --batch-size 4 --profile quick --repeats 3 --output benchmarks/new-fixed
```
