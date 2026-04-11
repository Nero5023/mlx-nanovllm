# MLX Nano LLM

Learning LLM inference on Apple Silicon, using `../nano-vllm` as a reference.
The first milestone handles one prompt with greedy decoding and a KV cache.

The [v0.1 milestone report](reports/v0.1/README.md) archives the Qwen3-0.6B
inference baseline, environment, and benchmark result. Code and report are tagged
`v0.1` together.

The next step adds a synchronous FIFO engine that reuses one loaded model and
finishes each request before starting the next.

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

## Multiple requests, one at a time

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

Each request has its own token budget and sampler (`sampler=MySampler()`). A
fresh greedy sampler is used by default. Model weights and tokenizer are shared;
the KV cache and output tokens are created anew for every request. This is a
single-threaded Python interface with one active request and no interleaving.
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
