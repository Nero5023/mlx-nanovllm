"""An asyncio interface to the dedicated inference worker."""

import asyncio
from pathlib import Path
from types import TracebackType
from typing import Self

from mlx_nano_llm.infer import DEFAULT_MODEL
from mlx_nano_llm.request import RequestOutput
from mlx_nano_llm.sampler import Sampler
from mlx_nano_llm.telemetry import MemorySnapshot, TokenCallback, TokenEvent
from mlx_nano_llm.worker import InferenceWorker


class InferenceClient:
    """Use one client from one event loop; concurrent coroutines may share it."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        *,
        use_cache: bool = True,
        trace: bool = False,
        collect_stats: bool = False,
        max_batch_size: int = 1,
        batch_wait_ms: float = 2.0,
    ) -> None:
        self._worker: InferenceWorker = InferenceWorker(
            model_path, use_cache=use_cache, trace=trace, collect_stats=collect_stats,
            max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started: bool = False
        self._closed: bool = False
        self._close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        if self._closed:
            raise RuntimeError("Client is closed.")
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("Use this client from its original event loop.")
        self._loop = loop
        try:
            await asyncio.wrap_future(self._worker.start())
        except BaseException:
            await self.close()
            raise
        if self._closed:
            raise RuntimeError("Client closed while starting.")
        self._started = True

    def submit(
        self, prompt: str, max_tokens: int = 128, *, sampler: Sampler | None = None,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> asyncio.Future[RequestOutput]:
        """Return immediately; optional token callbacks run on the client's event loop.

        Keep callbacks quick: slow callbacks delay this loop's observations.
        ignore_eos is useful for fixed-output-length benchmark workloads.
        """
        loop: asyncio.AbstractEventLoop = self._running_loop()

        def deliver(event: TokenEvent) -> None:
            if not result.cancelled() and on_token is not None:
                on_token(event)

        def from_worker(event: TokenEvent) -> None:
            loop.call_soon_threadsafe(deliver, event)

        # wrap_future delivers thread-safe worker results onto the asyncio loop.
        result: asyncio.Future[RequestOutput] = asyncio.wrap_future(
            self._worker.submit(
                prompt, max_tokens, sampler=sampler,
                on_token=from_worker if on_token is not None else None,
                ignore_eos=ignore_eos,
            ), loop=loop,
        )
        return result

    def _running_loop(self) -> asyncio.AbstractEventLoop:
        if self._closed or not self._started:
            raise RuntimeError("Start the client with 'async with' before submitting.")
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        if loop is not self._loop:
            raise RuntimeError("Use this client from its original event loop.")
        return loop

    async def generate(
        self, prompt: str, max_tokens: int = 128, *, sampler: Sampler | None = None,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> RequestOutput:
        return await self.submit(
            prompt, max_tokens, sampler=sampler, on_token=on_token, ignore_eos=ignore_eos
        )

    async def memory_snapshot(
        self, *, reset_peak: bool = False, clear_cache: bool = False,
    ) -> MemorySnapshot:
        loop: asyncio.AbstractEventLoop = self._running_loop()
        return await asyncio.wrap_future(
            self._worker.memory_snapshot(reset_peak=reset_peak, clear_cache=clear_cache), loop=loop
        )

    async def close(self) -> None:
        """Gracefully finish accepted work without blocking the event loop."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(asyncio.to_thread(self._worker.close))
        # A cancelled waiter must not cancel the ongoing worker shutdown.
        await asyncio.shield(self._close_task)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
