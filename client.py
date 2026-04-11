"""An asyncio interface to the dedicated inference worker."""

import asyncio
from pathlib import Path
from types import TracebackType
from typing import Self

from infer import DEFAULT_MODEL
from request import RequestOutput
from sampler import Sampler
from worker import InferenceWorker


class InferenceClient:
    """Use one client from one event loop; concurrent coroutines may share it."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        *,
        use_cache: bool = True,
        trace: bool = False,
        collect_stats: bool = False,
    ) -> None:
        self._worker: InferenceWorker = InferenceWorker(
            model_path, use_cache=use_cache, trace=trace, collect_stats=collect_stats
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
        self, prompt: str, max_tokens: int = 128, *, sampler: Sampler | None = None
    ) -> asyncio.Future[RequestOutput]:
        """Return immediately with an awaitable result for this request."""
        if self._closed or not self._started:
            raise RuntimeError("Start the client with 'async with' before submitting.")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Use this client from its original event loop.")
        # wrap_future delivers thread-safe worker results onto the asyncio loop.
        return asyncio.wrap_future(
            self._worker.submit(prompt, max_tokens, sampler=sampler), loop=self._loop
        )

    async def generate(
        self, prompt: str, max_tokens: int = 128, *, sampler: Sampler | None = None
    ) -> RequestOutput:
        return await self.submit(prompt, max_tokens, sampler=sampler)

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
