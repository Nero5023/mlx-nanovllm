"""A fixed batch: one prefill, then one token per unfinished row per step."""

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

    Batch membership and the batch dimension stay fixed until every row finishes.
    Finished rows feed dummy tokens and retain their KV allocation. No new
    requests enter this loop; removing/refilling rows is a later milestone.
    """
    for request in requests:
        if not request.prompt_token_ids or request.max_tokens < 0:
            raise ValueError("Each request needs a nonempty prompt and nonnegative budget.")
    generated: list[list[int]] = [[] for _ in requests]
    finished: list[bool] = [request.max_tokens == 0 for request in requests]
    if all(finished):
        return generated

    width: int = max(len(request.prompt_token_ids) for request in requests)
    left_padding: list[int] = [width - len(request.prompt_token_ids) for request in requests]
    inputs: mx.array = mx.array([
        [pad_token_id] * padding + request.prompt_token_ids
        for padding, request in zip(left_padding, requests)
    ], dtype=mx.int32)  # [batch, padded prompt length]

    # Each layer has batched K/V tensors. The cache supplies a padding-aware
    # attention mask and a separate RoPE offset for each row, starting at -pad.
    cache: list[BatchKVCache] = [BatchKVCache(left_padding) for _ in model.layers]
    all_greedy: bool = all(type(request.sampler) is GreedySampler for request in requests)
    greedy: GreedySampler = GreedySampler()

    for step in range(max(request.max_tokens for request in requests)):
        logits: mx.array = model(inputs, cache=cache)  # [batch, sequence, vocabulary]
        last_logits: mx.array = logits[:, -1, :]  # Left padding aligns real last tokens.
        if all_greedy:
            next_tokens: mx.array = greedy.sample(last_logits)
        else:
            # Preserve each request's sampler/state; stopped rows no longer sample.
            next_tokens = mx.concatenate([
                mx.array([pad_token_id], dtype=mx.int32) if finished[row]
                else request.sampler.sample(last_logits[row:row + 1])
                for row, request in enumerate(requests)
            ])
        token_ids: list[int] = cast(list[int], next_tokens.tolist())  # One GPU evaluation.
        generated_at: float = perf_counter()

        if trace:
            print(
                f"{'prefill' if step == 0 else 'decode':7} batch={len(requests)} "
                f"input={inputs.shape} logits={logits.shape} "
                f"offsets={cache[0].offset.tolist()} finished={finished}",
                file=sys.stderr,
            )

        for row, (request, token_id) in enumerate(zip(requests, token_ids)):
            if finished[row]:
                continue
            if token_id in eos_token_ids and not request.ignore_eos:
                finished[row] = True
                continue
            generated[row].append(token_id)
            if request.on_token is not None:
                request.on_token(TokenEvent(token_id, generated_at))
            finished[row] = len(generated[row]) >= request.max_tokens

        if all(finished):
            break
        inputs = mx.array([
            [pad_token_id if finished[row] else token_id]
            for row, token_id in enumerate(token_ids)
        ], dtype=mx.int32)  # [batch, 1]; finished rows cannot affect other rows.

    return generated
