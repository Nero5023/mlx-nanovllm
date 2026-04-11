"""Exercise real threads and futures with a controllable stand-in engine."""

import asyncio
import threading
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

from client import InferenceClient
from request import RequestOutput
from sampler import Sampler


@dataclass
class EngineState:
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    loading: threading.Event = field(default_factory=threading.Event)
    loaded: threading.Event = field(default_factory=threading.Event)
    block_loading: bool = False
    calls: list[str] = field(default_factory=list)
    thread_ids: list[int] = field(default_factory=list)


class FakeEngine:
    def __init__(self, state: EngineState) -> None:
        self.state: EngineState = state
        self.state.thread_ids.append(threading.get_ident())
        self.state.loading.set()
        if state.block_loading and not state.loaded.wait(timeout=5):
            raise TimeoutError("Test did not release model loading.")
        self.next_id: int = 0
        self.pending: tuple[int, str] | None = None

    def add_request(
        self, prompt: str, max_tokens: int, *, sampler: Sampler | None
    ) -> int:
        self.state.thread_ids.append(threading.get_ident())
        self.pending = (self.next_id, prompt)
        self.next_id += 1
        return self.next_id - 1

    def run_next(self) -> RequestOutput:
        self.state.thread_ids.append(threading.get_ident())
        assert self.pending is not None
        request_id, prompt = self.pending
        self.pending = None
        self.state.calls.append(prompt)
        if prompt == "blocked":
            self.state.started.set()
            if not self.state.release.wait(timeout=5):
                raise TimeoutError("Test did not release inference.")
        if prompt == "error":
            raise ValueError("request failed")
        return RequestOutput(request_id, prompt, f"answer:{prompt}", [request_id], None)


class InferenceClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.state: EngineState = EngineState()

        def create_engine(
            model_path: Path, *, use_cache: bool, trace: bool, collect_stats: bool
        ) -> FakeEngine:
            return FakeEngine(self.state)

        self.factory: MagicMock = self.enterContext(
            patch("worker.InferenceEngine", side_effect=create_engine)
        )

    async def wait_for_event(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def test_submissions_while_busy_fifo_and_thread_ownership(self) -> None:
        async with InferenceClient() as client:
            first = client.submit("blocked")
            try:
                await self.wait_for_event(self.state.started)
                second = client.submit("second")
                third = client.submit("third")
                await asyncio.sleep(0)  # The event loop can keep scheduling work.
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                self.assertEqual(self.state.calls, ["blocked"])
            finally:
                self.state.release.set()
            outputs = await asyncio.wait_for(asyncio.gather(first, second, third), 2)
            self.assertEqual([out.request_id for out in outputs], [0, 1, 2])
            self.assertEqual([out.text for out in outputs], [
                "answer:blocked", "answer:second", "answer:third"
            ])
            self.factory.assert_called_once()
            self.assertEqual(len(set(self.state.thread_ids)), 1)
            self.assertNotEqual(self.state.thread_ids[0], threading.get_ident())

    async def test_request_error_does_not_stop_worker(self) -> None:
        async with InferenceClient() as client:
            failed = client.submit("error")
            succeeding = client.submit("success")
            with self.assertRaisesRegex(ValueError, "request failed"):
                await failed
            self.assertEqual((await succeeding).text, "answer:success")
            self.assertEqual((await client.generate("later")).text, "answer:later")

    async def test_cancel_queued_and_running_waiters(self) -> None:
        async with InferenceClient() as client:
            running = client.submit("blocked")
            try:
                await self.wait_for_event(self.state.started)
                queued = client.submit("cancelled before starting")
                queued.cancel()
                running.cancel()
                await asyncio.sleep(0)  # Deliver cancellations to worker futures.
                last = client.submit("last")
            finally:
                self.state.release.set()
            with self.assertRaises(asyncio.CancelledError):
                await running
            with self.assertRaises(asyncio.CancelledError):
                await queued
            self.assertEqual((await asyncio.wait_for(last, 2)).text, "answer:last")
            self.assertEqual(self.state.calls, ["blocked", "last"])

    async def test_close_drains_queue_and_rejects_new_requests(self) -> None:
        client: InferenceClient = InferenceClient()
        self.addAsyncCleanup(client.close)
        with self.assertRaises(RuntimeError):
            client.submit("before start")
        await client.start()
        first = client.submit("blocked")
        try:
            await self.wait_for_event(self.state.started)
            second = client.submit("second")
            closing = asyncio.create_task(client.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            with self.assertRaises(RuntimeError):
                client.submit("too late")
        finally:
            self.state.release.set()
        await asyncio.wait_for(closing, 2)
        self.assertEqual([out.prompt for out in await asyncio.gather(first, second)],
                         ["blocked", "second"])
        self.assertFalse(client._worker._thread.is_alive())
        await client.close()  # Closing again is safe.

    async def test_loading_is_async_and_startup_cancellation_cleans_up(self) -> None:
        self.state.block_loading = True
        client: InferenceClient = InferenceClient()
        self.addAsyncCleanup(client.close)
        starting = asyncio.create_task(client.start())
        try:
            await self.wait_for_event(self.state.loading)
            self.assertFalse(starting.done())
            starting.cancel()
            await asyncio.sleep(0)
        finally:
            self.state.loaded.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 2)
        self.assertFalse(client._worker._thread.is_alive())

    async def test_model_loading_error_reaches_caller(self) -> None:
        self.factory.side_effect = RuntimeError("model load failed")
        client: InferenceClient = InferenceClient()
        with self.assertRaisesRegex(RuntimeError, "model load failed"):
            async with client:
                self.fail("A client with a failed model must not become ready.")
        self.assertFalse(client._worker._thread.is_alive())
        with self.assertRaises(RuntimeError):
            client.submit("after failure")


if __name__ == "__main__":
    unittest.main()
