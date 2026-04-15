"""One admitted group: prefill together, then remove finished rows during decode."""

import sys
from time import perf_counter
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.models.qwen3 import Model as Qwen3Model

from request import Request
from sampler import GreedySampler
from telemetry import TokenEvent


def generate_batch_tokens(
    model: Qwen3Model,
    requests: list[Request],
    eos_token_ids: set[int],
    *,
    pad_token_id: int = 0,
    trace: bool = False,
) -> list[list[int]]:
    """Return each row's output IDs in request order, excluding stopping EOS.

    Only unfinished requests participate in each forward pass. active_indices
    maps current batch rows to positions in requests/generated. No new requests
    enter this loop; admitting more work during decode is a later milestone.
    """
    for request in requests:
        if not request.prompt_token_ids or request.max_tokens < 0:
            raise ValueError("Each request needs a nonempty prompt and nonnegative budget.")
    generated: list[list[int]] = [[] for _ in requests]
    active_indices: list[int] = [
        index for index, request in enumerate(requests) if request.max_tokens > 0
    ]
    if not active_indices:
        return generated

    width: int = max(len(requests[index].prompt_token_ids) for index in active_indices)
    left_padding: list[int] = [
        width - len(requests[index].prompt_token_ids) for index in active_indices
    ]
    inputs: mx.array = mx.array([
        [pad_token_id] * padding + requests[index].prompt_token_ids
        for padding, index in zip(left_padding, active_indices)
    ], dtype=mx.int32)  # [batch, padded prompt length]

    # Each layer has batched K/V tensors. The cache supplies a padding-aware
    # attention mask and a separate RoPE offset for each row, starting at -pad.
    cache: list[BatchKVCache] = [BatchKVCache(left_padding) for _ in model.layers]
    all_greedy: bool = all(type(requests[index].sampler) is GreedySampler for index in active_indices)
    greedy: GreedySampler = GreedySampler()

    for step in range(max(request.max_tokens for request in requests)):
        logits: mx.array = model(inputs, cache=cache)  # [batch, sequence, vocabulary]
        last_logits: mx.array = logits[:, -1, :]  # Left padding aligns real last tokens.
        if all_greedy:
            next_tokens: mx.array = greedy.sample(last_logits)
        else:
            # Use the original request's sampler even after its row number changes.
            next_tokens = mx.concatenate([
                requests[index].sampler.sample(last_logits[row:row + 1])
                for row, index in enumerate(active_indices)
            ])
        token_ids: list[int] = cast(list[int], next_tokens.tolist())  # One GPU evaluation.
        generated_at: float = perf_counter()

        if trace:
            print(
                f"{'prefill' if step == 0 else 'decode':7} batch={len(active_indices)} "
                f"input={inputs.shape} logits={logits.shape} "
                f"offsets={cache[0].offset.tolist()} "
                f"request_ids={[requests[index].request_id for index in active_indices]}",
                file=sys.stderr,
            )

        keep_rows: list[int] = []  # Positions in this step's tensors, not original request IDs.
        for row, (index, token_id) in enumerate(zip(active_indices, token_ids, strict=True)):
            request: Request = requests[index]
            if token_id in eos_token_ids and not request.ignore_eos:
                continue
            generated[index].append(token_id)
            if request.on_token is not None:
                request.on_token(TokenEvent(token_id, generated_at))
            if len(generated[index]) < request.max_tokens:
                keep_rows.append(row)

        if not keep_rows:
            break
        if len(keep_rows) < len(active_indices):
            for layer_cache in cache:
                # Filters K, V, offsets and padding together. It also drops any
                # left padding shared by every surviving row.
                layer_cache.filter(keep_rows)
            active_indices = [active_indices[row] for row in keep_rows]
        inputs = mx.array([[token_ids[row]] for row in keep_rows], dtype=mx.int32)

    return generated
