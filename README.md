# MLX Nano LLM

Learning LLM inference on Apple Silicon, using `../nano-vllm` as a reference.
The first milestone handles one prompt with greedy decoding and a KV cache.

The [v0.1 milestone report](reports/v0.1/README.md) archives the Qwen3-0.6B
inference baseline, environment, and benchmark result. Code and report are tagged
`v0.1` together.

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
