"""One prompt, greedy decoding, and an optional KV cache on MLX."""

import argparse
from collections.abc import Sequence
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.qwen3 import Model as Qwen3Model
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mlx_nano_llm.benchmark import BenchmarkStats
from mlx_nano_llm.model import load_model
from mlx_nano_llm.paged_kv import PagedKVAdapter
from mlx_nano_llm.sampler import GreedySampler, Sampler
from mlx_nano_llm.telemetry import TokenCallback, TokenEvent


DEFAULT_MODEL: Path = Path(__file__).resolve().parents[1] / "models" / "Qwen3-0.6B-4bit"


def encode_prompt(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    # Like nano-vllm/example.py: format the chat before encoding it.
    prompt: str = cast(
        str,  # A single conversation with tokenize=False returns text.
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


def generate_tokens(
    model: Qwen3Model,
    prompt_token_ids: list[int],
    eos_token_ids: set[int],
    max_tokens: int = 128,
    *,
    sampler: Sampler | None = None,
    use_cache: bool = True,
    trace: bool = False,
    stats: BenchmarkStats | None = None,
    on_token: TokenCallback | None = None,
    paged_kv: bool = False,
    kv_page_size: int = 16,
    paged_cache: Sequence[PagedKVAdapter] | None = None,
) -> list[int]:
    """Return new token IDs, excluding the prompt and the stop token.

    Sampler.sample receives [batch, vocabulary] logits and returns [batch] integer
    token IDs. Defaults to a fresh GreedySampler. The loop handles batch=1.
    paged_kv enables paged storage with gather + existing attention. Optional
    paged_cache reuses empty engine-owned layer caches; pages are returned after
    this request, while the supplied pools remain available for later requests.
    """
    if not prompt_token_ids:
        raise ValueError("The prompt must contain at least one token.")
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative.")

    if paged_kv and not use_cache:
        raise ValueError('paged_kv requires use_cache=True.')
    if not isinstance(kv_page_size, int) or isinstance(kv_page_size, bool) or kv_page_size < 1:
        raise ValueError('kv_page_size must be a positive integer.')
    if paged_cache is not None:
        if not paged_kv:
            raise ValueError('paged_cache requires paged_kv=True.')
        if len(paged_cache) != len(model.layers):
            raise ValueError('paged_cache must contain one cache per model layer.')
        if any(cache.offset != 0 for cache in paged_cache):
            raise ValueError('Each request must start with empty paged caches.')
        if any(cache.page_size != kv_page_size for cache in paged_cache):
            raise ValueError('Shared caches must use kv_page_size.')

    if sampler is None:
        sampler = GreedySampler()

    if stats is not None:
        # Finish earlier GPU work before resetting this request's measurements.
        mx.synchronize()
        mx.reset_peak_memory()
        stats.reset(len(prompt_token_ids))

    # Each request gets a fresh cache, with one K/V pair per transformer layer.
    cache: list[KVCache] | list[PagedKVAdapter] | None = None
    paged_caches: list[PagedKVAdapter] = []
    if use_cache:
        if paged_kv and max_tokens > 0:
            # The final sampled token never enters KV: prefill + at most N-1 decode inputs.
            capacity_tokens = len(prompt_token_ids) + max_tokens - 1
            num_pages = (capacity_tokens + kv_page_size - 1) // kv_page_size
            if paged_cache is None:
                paged_caches = [PagedKVAdapter(num_pages=num_pages, page_size=kv_page_size)
                                for _ in model.layers]
            else:
                # Check every layer before writing any KV for the new request.
                for layer_cache in paged_cache:
                    available = (layer_cache.storage.pool.free_page_count
                                 if layer_cache.storage is not None else layer_cache.num_pages)
                    if num_pages > available:
                        raise MemoryError(f'Request needs {num_pages} KV pages per layer; only {available} free.')
                paged_caches = list(paged_cache)
            cache = paged_caches
        elif not paged_kv:
            cache = make_prompt_cache(model)
    inputs: mx.array = mx.array([prompt_token_ids], dtype=mx.int32)  # [batch=1, sequence]
    generated: list[int] = []

    try:
        for step in range(max_tokens):
            step_start: float = perf_counter() if stats is not None else 0.0
            logits: mx.array = model(inputs, cache=cache)  # LastTokenQwen3 returns [1, 1, vocabulary].
            next_token: mx.array = sampler.sample(logits[:, -1, :])
            # Reading the selected token ID evaluates MLX's lazy graph.
            token_id: int = cast(int, next_token.item())

            if stats is not None:
                mx.synchronize()
                elapsed: float = perf_counter() - step_start
                if step == 0:
                    stats.prefill_seconds = elapsed
                else:
                    stats.decode_steps += 1
                    stats.decode_seconds += elapsed

            if trace:
                phase: str = "prefill" if step == 0 else "decode" if use_cache else "full"
                cached_tokens: int = cache[0].offset if cache is not None else 0
                print(
                    f"{phase:7} input={inputs.shape} logits={logits.shape} "
                    f"cached={cached_tokens} next={token_id}",
                    file=sys.stderr,
                )

            if token_id in eos_token_ids:
                break
            generated.append(token_id)
            if on_token is not None:
                on_token(TokenEvent(token_id, perf_counter()))

            # After prefill, the cache already holds the earlier tokens' K and V.
            if use_cache:
                inputs = mx.array([[token_id]], dtype=mx.int32)
            else:
                inputs = mx.array([prompt_token_ids + generated], dtype=mx.int32)
    finally:
        # Return pages on normal completion, EOS, callback errors, or model errors.
        for layer_cache in paged_caches:
            layer_cache.clear()

    if stats is not None:
        stats.generated_tokens = len(generated)
        # Include resident model memory even when max_tokens=0 allocates nothing.
        stats.peak_memory_bytes = max(mx.get_peak_memory(), mx.get_active_memory())

    return generated


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="What is the capital of France?")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-cache", action="store_true", help="Recompute the full prefix.")
    parser.add_argument("--paged-kv", action="store_true", help="Single-request paged KV storage with gather + existing attention.")
    parser.add_argument("--kv-page-size", type=int, default=16, help="Tokens per KV page (default: 16).")
    parser.add_argument("--trace", action="store_true", help="Print shapes and cache length.")
    parser.add_argument(
        "--benchmark", nargs="?", type=Path, const=Path("benchmarks/latest.json"),
        help="Warm up, measure inference, and save JSON (default: benchmarks/latest.json).",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.max_tokens < 0:
        parser.error("--max-tokens must be nonnegative")
    if args.kv_page_size < 1:
        parser.error("kv-page-size must be >= 1")
    if args.paged_kv and args.no_cache:
        parser.error("--paged-kv requires KV caching; remove --no-cache")
    if not (args.model / "config.json").is_file():
        parser.error(f"Model not found at {args.model}. See README.md for the download command.")

    # MLX-LM provides the Qwen3 architecture and quantized weight loading.
    # Transformers provides only the tokenizer; all model computation uses MLX.
    loaded_model, config = load_model(args.model)
    model: Qwen3Model = cast(Qwen3Model, loaded_model)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    prompt_token_ids: list[int] = encode_prompt(tokenizer, args.prompt)
    if len(prompt_token_ids) + args.max_tokens > config["max_position_embeddings"]:
        parser.error("Prompt plus requested output exceeds the model's context length.")

    eos: int | list[int] = config["eos_token_id"]
    eos_token_ids: set[int] = set(eos if isinstance(eos, list) else [eos])
    tokenizer_eos: int | None = cast(int | None, tokenizer.eos_token_id)
    if tokenizer_eos is not None:
        eos_token_ids.add(tokenizer_eos)

    stats: BenchmarkStats | None = None
    warmup_tokens: int = 0
    if args.benchmark is not None:
        # Warm up prefill and one decode step using a separate, disposable KV cache.
        warmup_tokens = min(args.max_tokens, 2)
        generate_tokens(
            model, prompt_token_ids, set(), warmup_tokens,
            use_cache=not args.no_cache,
            paged_kv=args.paged_kv, kv_page_size=args.kv_page_size,
        )
        stats = BenchmarkStats()

    generated: list[int] = generate_tokens(
        model,
        prompt_token_ids,
        eos_token_ids,
        args.max_tokens,
        use_cache=not args.no_cache,
        trace=args.trace,
        stats=stats,
        paged_kv=args.paged_kv, kv_page_size=args.kv_page_size,
    )
    # Decode together so tokens containing partial UTF-8 characters join correctly.
    print(tokenizer.decode(generated, skip_special_tokens=True))

    if stats is not None:
        record: dict[str, object] = {
            "model": str(args.model),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "use_cache": not args.no_cache,
            "kv_cache_backend": "paged-gather" if args.paged_kv else "contiguous" if not args.no_cache else "none",
            "kv_page_size": args.kv_page_size if args.paged_kv else None,
            "sampler": "GreedySampler",
            "warmup_tokens": warmup_tokens,
            **stats.to_dict(),
        }
        result_json: str = json.dumps(record, indent=2, ensure_ascii=False)
        output_path: Path = args.benchmark
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")
        print(f"\nBenchmark:\n{result_json}\nSaved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
