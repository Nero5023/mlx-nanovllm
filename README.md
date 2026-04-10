# MLX Nano LLM

Learning LLM inference on Apple Silicon, using `../nano-vllm` as a reference.
The first milestone handles one prompt with greedy decoding and a KV cache.

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
