"""A resumable batch: prefill, decode one step, compact, and merge new rows."""

import sys
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.models.qwen3 import Model as Qwen3Model

from mlx_nano_llm.request import Request
from mlx_nano_llm.sampler import GreedySampler
from mlx_nano_llm.telemetry import TokenEvent


@dataclass
class ActiveRequest:
    request: Request
    token_ids: list[int] = field(default_factory=list)


class BatchState:
    """Keep row state between forward passes; each row owns its request/output.

    The first step prefills the prompts. Later steps consume the previous token.
    Only prefilled batches can merge: each row then has KV history plus one token
    waiting to enter the model. No request history needs to be recomputed.
    """

    def __init__(
        self, model: Qwen3Model, active: list[ActiveRequest], eos_token_ids: set[int],
        *, pad_token_id: int = 0, trace: bool = False,
    ) -> None:
        self.model: Qwen3Model = model
        self.active: list[ActiveRequest] = list(active)
        self.eos_token_ids: set[int] = eos_token_ids
        self.trace: bool = trace
        self._prefill: bool = True
        self._greedy: GreedySampler = GreedySampler()
        width: int = max((len(row.request.prompt_token_ids) for row in active), default=0)
        left_padding: list[int] = [width - len(row.request.prompt_token_ids) for row in active]
        self.inputs: mx.array = mx.array([
            [pad_token_id] * padding + row.request.prompt_token_ids
            for padding, row in zip(left_padding, active, strict=True)
        ], dtype=mx.int32)
        self.cache: list[BatchKVCache] = [BatchKVCache(left_padding) for _ in model.layers]

    @property
    def size(self) -> int:
        return len(self.active)

    def step(self) -> list[ActiveRequest]:
        """Run one forward pass and return the requests that just completed."""
        if not self.active:
            return []
        logits: mx.array = self.model(self.inputs, cache=self.cache)
        last_logits: mx.array = logits[:, -1, :]  # Left padding aligns real last tokens.
        if all(type(row.request.sampler) is GreedySampler for row in self.active):
            next_tokens: mx.array = self._greedy.sample(last_logits)
        else:
            next_tokens = mx.concatenate([
                state.request.sampler.sample(last_logits[row:row + 1])
                for row, state in enumerate(self.active)
            ])
        token_ids: list[int] = cast(list[int], next_tokens.tolist())  # One GPU evaluation.
        generated_at: float = perf_counter()

        if self.trace:
            print(
                f"{'prefill' if self._prefill else 'decode':7} batch={self.size} "
                f"input={self.inputs.shape} logits={logits.shape} "
                f"offsets={self.cache[0].offset.tolist()} "
                f"request_ids={[row.request.request_id for row in self.active]}",
                file=sys.stderr,
            )
        self._prefill = False
        finished: list[ActiveRequest] = []
        keep_rows: list[int] = []  # Positions in this step's tensors, not original request IDs.
        for row, (state, token_id) in enumerate(zip(self.active, token_ids, strict=True)):
            request: Request = state.request
            if token_id in self.eos_token_ids and not request.ignore_eos:
                finished.append(state)
                continue
            state.token_ids.append(token_id)
            if request.on_token is not None:
                request.on_token(TokenEvent(token_id, generated_at))
            if len(state.token_ids) < request.max_tokens:
                keep_rows.append(row)
            else:
                finished.append(state)

        if keep_rows and len(keep_rows) < self.size:
            for layer_cache in self.cache:
                # Filters K, V, offsets and padding together. It also drops any
                # left padding shared by every surviving row.
                layer_cache.filter(keep_rows)
        self.active = [self.active[row] for row in keep_rows]
        self.inputs = mx.array([[token_ids[row]] for row in keep_rows], dtype=mx.int32)
        if not self.active:
            self.cache = []  # Do not retain a finished group's KV allocation.
        return finished

    def extend(self, other: "BatchState") -> None:
        """Append prefilled rows; MLX aligns their KV lengths with left padding."""
        if not other.size:
            return
        if not self.size or self._prefill or other._prefill:
            raise ValueError("Only nonempty, prefilled batches can merge.")
        for cache, incoming in zip(self.cache, other.cache, strict=True):
            # incoming do the padding here
            cache.extend(incoming)
        self.inputs = mx.concatenate([self.inputs, other.inputs], axis=0)
        self.active.extend(other.active)
        if self.trace:
            print(f"merge   batch={self.size} request_ids="
                  f"{[row.request.request_id for row in self.active]}", file=sys.stderr)


def generate_batch_tokens(
    model: Qwen3Model,
    requests: list[Request],
    eos_token_ids: set[int],
    *,
    pad_token_id: int = 0,
    trace: bool = False,
) -> list[list[int]]:
    """Complete one group, returning output IDs in the original request order."""
    for request in requests:
        if not request.prompt_token_ids or request.max_tokens < 0:
            raise ValueError("Each request needs a nonempty prompt and nonnegative budget.")
    states: list[ActiveRequest] = [ActiveRequest(request) for request in requests]
    active: list[ActiveRequest] = [state for state in states if state.request.max_tokens > 0]
    if active:
        batch: BatchState = BatchState(model, active, eos_token_ids, pad_token_id=pad_token_id, trace=trace)
        while batch.size:
            batch.step()

    return [state.token_ids for state in states]
