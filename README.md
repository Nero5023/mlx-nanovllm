# MLX Nano LLM

Learning LLM inference on Apple Silicon, using `../nano-vllm` as a reference.
The first milestone handles one prompt with greedy decoding and a KV cache.

The [v0.1 milestone report](reports/v0.1/README.md) archives the Qwen3-0.6B
inference baseline, environment, and benchmark result. Code and report are tagged
`v0.1` together.

The `v0.2` tag records the async FIFO worker and serving benchmark baseline.
The engine now also supports fixed batches: several requests share each model
forward pass, and later arrivals wait until the current batch finishes.

## Fixed batches

The [fixed-batching benchmark archive](reports/fixed-batching/README.md) records
the serial comparison, raw observations, and BF16 numerical differences.

```bash
uv run python batch_example.py
uv run python bench.py --batch-size 4 --repeats 3
```

```python
async with InferenceClient(max_batch_size=4, batch_wait_ms=2.0) as client:
    outputs = await asyncio.gather(
        client.generate("What is the capital of France?", max_tokens=24),
        client.generate("Count from 1 to 20.", max_tokens=64),
    )
```

The existing `submit()` / `generate()` API is unchanged. Set `max_batch_size`
to enable batching; its default of 1 retains the serial path. Before starting
a batch, the worker collects up to that many FIFO requests, waiting at most
`batch_wait_ms` for more arrivals. A partial batch runs when the wait expires;
there is no requirement to fill every slot. Set the wait to 0 to collect only
already queued work. Collection wait is included in queue time and TTFT.

Start reading [batch_infer.py](batch_infer.py). `generate_batch_tokens()`:

1. Left-pads the prompts to `[batch, longest_prompt]`. Each row's final position
   is a real prompt token, so `logits[:, -1, :]` selects the correct next scores.
2. Uses MLX-LM's `BatchKVCache` to mask padding and keep each row's RoPE offset.
   Padding must not count as real context; for lengths `[3, 1]`, padding is
   `[0, 2]` and initial offsets are `[0, -2]`. After prefill offsets are `[3, 1]`.
3. Runs one forward pass per step. Greedy sampling selects one token per row;
   `.tolist()` evaluates the batch once. Decode input is `[batch, 1]`.
4. Tracks each row's EOS, output budget, sampler and token callback independently.
   Completed rows retain their KV and feed dummy tokens, without emitting more
   output. Batch membership stays fixed until all rows finish.

`engine.run_batch(N)` removes up to N already queued requests and calls this
loop. `worker.py` forms groups from async submissions; it never adds a new row
while a batch is generating. Token callbacks arrive during generation, but final
request futures all complete at the end of their batch. This deliberately makes
the long/short-request cost visible before we implement continuous batching.

`RequestOutput.timing` records actual `batch_size` and a `batch_id` (the first
request ID in that group). These show whether simultaneous clients actually
formed a model batch. Invalid requests fail individually before model execution;
a shared inference failure fails the selected batch, and later batches continue.
Cancellation skips a request before admission; after admission its row remains
until batch completion. Memory probes and shutdown are FIFO barriers.

This stage requires KV caching and `collect_stats=False` when batching is
enabled. Per-request compute/memory stats cannot attribute shared batch work;
use `bench.py` for client latency and case-level memory. Padded rows consume
memory and compute, so similar prompt/output lengths are the simplest first
comparison. No active-row removal, slot refill, or prefix sharing is implemented.

```bash
uv run python -m unittest -v test_engine test_client test_bench test_batch_infer
```

The tests cover fixed membership, partial batches, failure/cancellation/shutdown,
individual EOS/budgets, padding, and real tiny-Qwen prefill/decode logits compared
with unpadded serial inference. The benchmark also compares real model output
token IDs for mixed lengths/budgets, reordered batches, and a partial batch.

Batch and serial execution can use different numerical kernels. With 4-bit
weights and BF16 activations, close logits can round to a tie and change greedy
token IDs even when padding and KV handling are correct. The implementation
preserves the model's original precision; exact output equivalence for every
prompt is not guaranteed. Keep strict regression checks on their test cases,
and inspect logits at the first divergence when comparing other workloads.

## Run

```bash
uv sync --locked
uv run python infer.py "What is the capital of France?" --max-tokens 32
uv run python infer.py "What is the capital of France?" --max-tokens 8 --trace
```

Python 3.12 is pinned in `.python-version`. Model files are already downloaded
to `models/Qwen3-0.6B-4bit/` and ignored by Git. To download them again:

```bash
uv run hf download mlx-community/Qwen3-0.6B-4bit \
  --revision 73e3e38d981303bc594367cd910ea6eb48349da8 \
  --local-dir models/Qwen3-0.6B-4bit
```

Inference loads local files and runs on MLX. It does not need a network connection.
The weights use 4-bit quantization; the KV cache uses the model's activation dtype.

## Async clients and one worker

```bash
uv run python async_example.py
```

```python
import asyncio
from client import InferenceClient

async def main() -> None:
    async with InferenceClient() as client:
        first = client.submit("What is the capital of France?", max_tokens=32)
        second = client.submit("What is 2 + 2?", max_tokens=16)
        for output in await asyncio.gather(first, second):
            print(output.request_id, output.text)

asyncio.run(main())
```

`client.py` runs on the asyncio event loop. `submit()` enqueues raw text and
immediately returns an awaitable future; `await client.generate(...)` is a
convenience for submitting and awaiting one result. Multiple coroutines share
one client. They can submit while an earlier request is generating.

With the default batch size of 1, `worker.py` owns a thread-safe FIFO queue and one dedicated thread. That thread
loads the model/tokenizer once, tokenizes each request, and calls the synchronous
engine's `add_request()` and `run_next()`. Every model operation stays on this
thread. Results and exceptions cross back through `concurrent.futures.Future`
and `asyncio.wrap_future`; the worker never modifies an asyncio future directly.

```text
async clients -> thread-safe queue -> one model-owning worker thread
      ^                                  |
      +---------- per-request future ----+
```

`async_example.py` submits B and C shortly after A. Its timestamps show them
arriving while A runs, then completing in order. Displayed latency includes
queueing and execution; this is an in-process example with whole-response output.

Use `async with` to manage the worker. Exiting stops submissions, drains accepted
requests, and joins the thread without blocking the event loop. Each request's
exception goes to its own waiter, and later requests continue. Cancelling a
future can skip work that has not started; it cannot interrupt active inference.
After active inference finishes, its cancelled waiter discards the result.
Each client belongs to one event loop and is closed permanently on exit.

The client accepts the same `use_cache`, `trace`, and `collect_stats` options as
the engine, and each submission can specify a sampler. Pass a separate sampler
instance per request when it keeps mutable state. There is no automatic warmup;
the existing benchmark command provides warmed measurements.

Pass `on_token=callback` to observe each generated token on the client's event
loop. The callback receives a `TokenEvent(token_id, generated_at)`; keep it quick.
Awaiting the returned future still gives the complete `RequestOutput`, now with
prompt length and worker queue/start/finish timestamps. `ignore_eos=True` is an
opt-in fixed-output-budget mode for benchmarks; ordinary requests stop at EOS.

Run the queue and concurrency checks with:

```bash
uv run python -m unittest -v test_engine test_client
```

These tests use a stand-in engine with real threads/futures, including blocked
inference, FIFO delivery, cancellation, failures, and shutdown. The async example
runs the real Qwen model.

## Synchronous engine

```bash
uv run python example.py
```

```python
from engine import InferenceEngine

engine = InferenceEngine()  # Load model and tokenizer once.
first_id = engine.add_request("What is the capital of France?", max_tokens=32)
second_id = engine.add_request("What is 2 + 2?", max_tokens=16)

for output in engine.run():
    print(output.request_id, output.text)
```

`add_request()` tokenizes, validates, and appends a request to the queue.
`run_next()` removes the oldest request and runs its entire generation before
returning a `RequestOutput`; it returns `None` when the queue is empty. `run()`
drains the queue and returns outputs in submission order. You can add more
requests after a run and reuse the same engine; IDs increase within that engine.
Use `run_batch(N)` instead to finish up to N queued requests together.

Each request has its own token budget and sampler (`sampler=MySampler()`). A
fresh greedy sampler is used by default. Model weights and tokenizer are shared;
the KV cache and output tokens are created anew for every request or fixed batch.
This is a synchronous Python interface: `run_next()` runs one request and
`run_batch(N)` runs a fixed group, without admitting new work during execution.
Invalid requests are rejected before enqueueing. If inference raises an error,
it propagates; the failed request is removed and later requests remain queued.

Read `request.py` for the request/output data, `engine.py` for queue management,
and `infer.py` for the generation loop. `pending_requests` gives the queue length.
The engine also accepts `use_cache=False`, `trace=True`, and `collect_stats=True`.
With statistics enabled, each output has its own `BenchmarkStats`; these include
the first request's cold-start costs because the engine does not run a warmup.
Use the benchmark command below for a warmed measurement.

Run queue behavior checks with `uv run python -m unittest -v test_engine`.
These use stand-ins for model loading and generation; `example.py` runs the real model.

## Small benchmark

For multi-request TTFT, inter-token latency, total throughput, and peak memory,
use the [serving benchmark suite](BENCHMARKS.md):

```bash
uv run python bench.py --profile full --repeats 3
```

It saves raw observations and a report under `benchmarks/`, with workloads for
the upcoming fixed-batch, continuous-batch, prefix-reuse, and paging milestones.
The smaller command below remains useful for inspecting one inference loop.

```bash
uv run python infer.py "Count from 1 to 100, separated by commas." \
  --max-tokens 128 --benchmark
```

This prints metrics and saves them to `benchmarks/latest.json` (overwritten each
run, ignored by Git). Supply a path to retain separate measurements, for example
`--benchmark benchmarks/cached.json`. Add `--no-cache` to compare full-prefix
recomputation; the JSON records whether caching was enabled.

- `prompt_tokens`: tokenized input length, including the chat template.
- `generated_tokens`: actual returned output length, excluding EOS.
- Prefill tok/s: prompt length divided by the first forward pass and token
  selection time. This pass also produces the first output token.
- Decode tok/s: subsequent forward-pass count (`decode_steps`) divided by their
  total compute and selection time. An EOS prediction counts as a step even
  though it is excluded from the returned output. Without EOS, generating N
  tokens takes N-1 decode steps. With `--no-cache`, each step recomputes the prefix.
- `peak_memory_gb`: MLX's peak used memory during the measured request, including
  resident model weights and KV cache, in decimal GB. This is not process RSS or
  total system memory. Model-loading and warmup peaks are excluded.

The benchmark first runs up to two unmeasured tokens (prefill and one decode)
using a separate cache. It then measures one request. Timers include `.item()`
and GPU synchronization so lazy computation finishes before timing ends. Model
loading, tokenization, input-array assembly, trace printing, and text decoding
are outside the timed regions. A phase with no steps reports `null` throughput.
This is a small local measurement; repeated runs can vary.

## Read the loop

Start with `encode_prompt()` and `generate_tokens()` in `infer.py`:

1. `AutoTokenizer` applies Qwen's chat template and encodes the text, following
   nano-vllm's tokenizer flow. Thinking is disabled for this greedy exercise.
2. **Prefill:** the model processes all prompt tokens and fills a cache per layer.
3. Take logits at the final input position, then `argmax` selects the next token.
4. **Decode:** pass just that new token, reusing the cache. Stop at EOS or the
   token budget. Decode the collected output IDs into text.

`sampler.py` defines the `Sampler` abstract base class and `GreedySampler`.
Its `sample()` method receives `[batch, vocabulary]` logits and returns an
integer MLX array of shape `[batch]` containing the selected token IDs.
`generate_tokens()` creates a fresh `GreedySampler` by default. To change token
selection, subclass `Sampler`, implement `sample()`, and pass an instance such
as `sampler=MySampler()`; the generation loop and cache handling stay the same.

`--trace` shows `[batch, input length]`, `[batch, input length, vocabulary]`,
the number of tokens cached per layer, and the selected token ID. The selected
token enters the cache on the *following* forward pass.

Try `--no-cache --trace`: every pass receives the full growing prefix instead.
Cache reuse is mathematically equivalent to recomputing the prefix; floating-point
kernel differences can slightly change logits and occasionally a close argmax.

Our code owns the generation loop. MLX-LM supplies the Qwen3 layers, quantized
weight loader, and KV-cache implementation. Those are useful reference code to
study and replace incrementally; this is not yet a from-scratch model or scheduler.

## Sources

- [Qwen's original model and tokenizer](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Community MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit)
- [Qwen3 layers in MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [KV cache in MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
