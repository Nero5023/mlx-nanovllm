"""One prompt, greedy decoding, and an optional KV cache on MLX."""

import argparse
import sys
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.utils import load_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_MODEL: Path = Path(__file__).parent / "models" / "Qwen3-0.6B-4bit"


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
    prompt_ids: list[int],
    eos_token_ids: set[int],
    max_tokens: int = 128,
    *,
    use_cache: bool = True,
    trace: bool = False,
) -> list[int]:
    """Return new token IDs, excluding the prompt and the stop token."""
    if not prompt_ids:
        raise ValueError("The prompt must contain at least one token.")
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative.")

    # Each request gets a fresh cache, with one K/V pair per transformer layer.
    cache: list[KVCache] | None = make_prompt_cache(model) if use_cache else None
    inputs: mx.array = mx.array([prompt_ids], dtype=mx.int32)  # [batch=1, sequence]
    generated: list[int] = []

    for step in range(max_tokens):
        logits: mx.array = model(inputs, cache=cache)  # [1, sequence, vocabulary]
        next_token: mx.array = mx.argmax(logits[:, -1, :], axis=-1)
        # argmax returns an integer; reading the scalar evaluates MLX's lazy graph.
        token_id: int = cast(int, next_token.item())

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

        # After prefill, the cache already holds the earlier tokens' K and V.
        if use_cache:
            inputs = mx.array([[token_id]], dtype=mx.int32)
        else:
            inputs = mx.array([prompt_ids + generated], dtype=mx.int32)

    return generated


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="What is the capital of France?")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-cache", action="store_true", help="Recompute the full prefix.")
    parser.add_argument("--trace", action="store_true", help="Print shapes and cache length.")
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
    prompt_ids: list[int] = encode_prompt(tokenizer, args.prompt)
    if len(prompt_ids) + args.max_tokens > config["max_position_embeddings"]:
        parser.error("Prompt plus requested output exceeds the model's context length.")

    eos: int | list[int] = config["eos_token_id"]
    eos_token_ids: set[int] = set(eos if isinstance(eos, list) else [eos])
    tokenizer_eos: int | None = cast(int | None, tokenizer.eos_token_id)
    if tokenizer_eos is not None:
        eos_token_ids.add(tokenizer_eos)
    generated: list[int] = generate_tokens(
        model,
        prompt_ids,
        eos_token_ids,
        args.max_tokens,
        use_cache=not args.no_cache,
        trace=args.trace,
    )
    # Decode together so tokens containing partial UTF-8 characters join correctly.
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
