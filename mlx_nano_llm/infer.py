"""One prompt, greedy decoding, and an optional KV cache on MLX."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.utils import load_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mlx_nano_llm.benchmark import BenchmarkStats
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
) -> list[int]:
    """Return new token IDs, excluding the prompt and the stop token.

    Sampler.sample receives [batch, vocabulary] logits and returns [batch] integer
    token IDs. Defaults to a fresh GreedySampler. The loop handles batch=1.
    """
    if not prompt_token_ids:
        raise ValueError("The prompt must contain at least one token.")
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative.")

    if sampler is None:
        sampler = GreedySampler()

    if stats is not None:
        # Finish earlier GPU work before resetting this request's measurements.
        mx.synchronize()
        mx.reset_peak_memory()
        stats.reset(len(prompt_token_ids))

    # Each request gets a fresh cache, with one K/V pair per transformer layer.
    cache: list[KVCache] | None = make_prompt_cache(model) if use_cache else None
    inputs: mx.array = mx.array([prompt_token_ids], dtype=mx.int32)  # [batch=1, sequence]
    generated: list[int] = []

    for step in range(max_tokens):
        step_start: float = perf_counter() if stats is not None else 0.0
        logits: mx.array = model(inputs, cache=cache)  # [1, sequence, vocabulary]
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
    parser.add_argument("--trace", action="store_true", help="Print shapes and cache length.")
    parser.add_argument(
        "--benchmark", nargs="?", type=Path, const=Path("benchmarks/latest.json"),
        help="Warm up, measure inference, and save JSON (default: benchmarks/latest.json).",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.max_tokens < 0:
        parser.error("--max-tokens must be nonnegative")
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
    )
    # Decode together so tokens containing partial UTF-8 characters join correctly.
    print(tokenizer.decode(generated, skip_special_tokens=True))

    if stats is not None:
        record: dict[str, object] = {
            "model": str(args.model),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "use_cache": not args.no_cache,
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
