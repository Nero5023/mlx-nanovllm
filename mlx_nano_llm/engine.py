"""Model ownership and request state; the worker decides when to admit or decode."""

from collections import deque
from pathlib import Path
from typing import cast

from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.utils import load_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mlx_nano_llm.benchmark import BenchmarkStats
from mlx_nano_llm.batch_infer import ActiveRequest, BatchState, generate_batch_tokens
from mlx_nano_llm.infer import DEFAULT_MODEL, encode_prompt, generate_tokens
from mlx_nano_llm.request import Request, RequestOutput
from mlx_nano_llm.sampler import GreedySampler, Sampler
from mlx_nano_llm.telemetry import TokenCallback


class InferenceEngine:
    """Load one model and tokenizer, then reuse them for queued requests.

    run_next()/run_batch() complete a request/group. Continuous serving uses
    prefill_next() to admit work and step() to advance the active batch once.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        *,
        use_cache: bool = True,
        trace: bool = False,
        collect_stats: bool = False,
    ) -> None:
        loaded_model, config = load_model(model_path)
        self.model: Qwen3Model = cast(Qwen3Model, loaded_model)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.max_context_tokens: int = config["max_position_embeddings"]
        eos: int | list[int] = config["eos_token_id"]
        self.eos_token_ids: set[int] = set(eos if isinstance(eos, list) else [eos])
        tokenizer_eos: int | None = cast(int | None, self.tokenizer.eos_token_id)
        if tokenizer_eos is not None:
            self.eos_token_ids.add(tokenizer_eos)

        self.use_cache: bool = use_cache
        self.trace: bool = trace
        self.collect_stats: bool = collect_stats
        self._waiting: deque[Request] = deque()
        self._next_request_id: int = 0
        self._batch: BatchState | None = None

    @property
    def pending_requests(self) -> int:
        return len(self._waiting)

    @property
    def active_requests(self) -> int:
        return self._batch.size if self._batch is not None else 0

    def add_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        *,
        sampler: Sampler | None = None,
        on_token: TokenCallback | None = None,
        ignore_eos: bool = False,
    ) -> int:
        """Validate and queue a prompt; return its ID without running inference."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be nonnegative.")
        prompt_token_ids: list[int] = encode_prompt(self.tokenizer, prompt)
        if not prompt_token_ids:
            raise ValueError("The prompt must contain at least one token.")
        if len(prompt_token_ids) + max_tokens > self.max_context_tokens:
            raise ValueError("Prompt plus requested output exceeds the model's context length.")

        request: Request = Request(
            request_id=self._next_request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            sampler=sampler if sampler is not None else GreedySampler(),
            on_token=on_token,
            ignore_eos=ignore_eos,
        )
        self._waiting.append(request)
        self._next_request_id += 1
        return request.request_id

    def run_next(self) -> RequestOutput | None:
        """Complete the oldest request, or return None if the queue is empty.

        Inference errors propagate to the caller. The failed request is removed,
        and subsequent requests remain queued for the next call.
        """
        if not self._waiting:
            return None
        request: Request = self._waiting.popleft()
        stats: BenchmarkStats | None = BenchmarkStats() if self.collect_stats else None
        # generate_tokens() creates a fresh KV cache local to this request.
        token_ids: list[int] = generate_tokens(
            self.model,
            request.prompt_token_ids,
            set() if request.ignore_eos else self.eos_token_ids,
            request.max_tokens,
            sampler=request.sampler,
            use_cache=self.use_cache,
            trace=self.trace,
            stats=stats,
            on_token=request.on_token,
        )
        text: str = cast(str, self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return RequestOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            text=text,
            token_ids=token_ids,
            stats=stats,
            prompt_tokens=len(request.prompt_token_ids),
        )

    def run(self) -> list[RequestOutput]:
        """Drain the queue in submission order and return the completed outputs."""
        outputs: list[RequestOutput] = []
        while (output := self.run_next()) is not None:
            outputs.append(output)
        return outputs

    def run_batch(self, max_batch_size: int) -> list[RequestOutput]:
        """Consume up to N oldest requests; remove completed rows during decode.

        Every request has its own budget, EOS rule, sampler and token callback.
        Selected requests are removed even if inference fails; later work stays
        queued. Per-request compute/memory statistics don't describe shared work,
        so use the serving benchmark's case-level measurements for batches.
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive.")
        if not self.use_cache or self.collect_stats:
            raise ValueError("Fixed batches require use_cache=True and collect_stats=False.")
        requests: list[Request] = [
            self._waiting.popleft() for _ in range(min(max_batch_size, len(self._waiting)))
        ]
        if not requests:
            return []
        pad_token_id: int | None = cast(int | None, self.tokenizer.pad_token_id)
        batches: list[list[int]] = generate_batch_tokens(
            self.model, requests, self.eos_token_ids,
            pad_token_id=pad_token_id if pad_token_id is not None else 0,
            trace=self.trace,
        )
        return [RequestOutput(
            request_id=request.request_id, prompt=request.prompt,
            text=cast(str, self.tokenizer.decode(token_ids, skip_special_tokens=True)),
            token_ids=token_ids, stats=None, prompt_tokens=len(request.prompt_token_ids),
        ) for request, token_ids in zip(requests, batches, strict=True)]

    def prefill_next(self, max_requests: int) -> list[RequestOutput]:
        """Prefill queued requests separately, then merge survivors into decode.

        The worker limits admission to available slots. A complete prefill blocks
        existing decode work; chunking that prefill is a later optimization.
        Zero-budget and one-token/EOS requests can finish at admission.
        """
        if max_requests < 1:
            raise ValueError("max_requests must be positive.")
        if not self.use_cache or self.collect_stats:
            raise ValueError("Batching requires use_cache=True and collect_stats=False.")
        count = min(max_requests, len(self._waiting))
        states: list[ActiveRequest] = []
        for _ in range(count):
            request = self._waiting.popleft()
            states.append(ActiveRequest(request))

        finished: list[ActiveRequest] = [state for state in states if state.request.max_tokens == 0]
        active: list[ActiveRequest] = [state for state in states if state.request.max_tokens > 0]

        if active:
            pad_token_id: int | None = cast(int | None, self.tokenizer.pad_token_id)
            incoming: BatchState = BatchState(
                self.model, active, self.eos_token_ids,
                pad_token_id=pad_token_id if pad_token_id is not None else 0, trace=self.trace,
            )
            finished.extend(incoming.step())
            if incoming.size:
                if self._batch is None:
                    self._batch = incoming
                else:
                    self._batch.extend(incoming)
        return [self._output(state) for state in finished]

    def step(self) -> list[RequestOutput]:
        """Decode one token per active row; return only newly finished requests."""
        if self._batch is None:
            return []
        finished: list[ActiveRequest] = self._batch.step()
        if not self._batch.size:
            self._batch = None
        return [self._output(state) for state in finished]

    def abort_batch(self) -> None:
        """Discard active state after a shared inference failure."""
        self._batch = None

    def _output(self, state: ActiveRequest) -> RequestOutput:
        return RequestOutput(
            request_id=state.request.request_id, prompt=state.request.prompt,
            text=cast(str, self.tokenizer.decode(state.token_ids, skip_special_tokens=True)),
            token_ids=state.token_ids, stats=None,
            prompt_tokens=len(state.request.prompt_token_ids),
        )
