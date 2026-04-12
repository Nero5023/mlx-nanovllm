"""Exercise real threads and futures with a controllable stand-in engine."""

import asyncio
import threading
import unittest
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

from client import InferenceClient
from request import RequestOutput
from sampler import Sampler
from telemetry import TokenCallback, TokenEvent


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
        self.on_token: TokenCallback | None = None

    def add_request(
        self, prompt: str, max_tokens: int, *, sampler: Sampler | None,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> int:
        self.state.thread_ids.append(threading.get_ident())
        self.pending = (self.next_id, prompt)
        self.on_token = on_token
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
        if self.on_token is not None:
            self.on_token(TokenEvent(request_id, perf_counter()))
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

    async def test_tokens_arrive_on_client_loop_before_completion(self) -> None:
        events: list[TokenEvent] = []
        callback_threads: list[int] = []

        def receive(event: TokenEvent) -> None:
            events.append(event)
            callback_threads.append(threading.get_ident())

        async with InferenceClient() as client:
            output = await client.generate("stream", on_token=receive)
        self.assertEqual([event.token_id for event in events], output.token_ids)
        self.assertEqual(callback_threads, [threading.get_ident()])
        assert output.timing is not None
        self.assertLessEqual(output.timing.enqueued_at, output.timing.started_at)
        self.assertLessEqual(output.timing.started_at, events[0].generated_at)
        self.assertLessEqual(events[0].generated_at, output.timing.finished_at)

    async def test_memory_probe_runs_on_worker_and_resets_only_when_requested(self) -> None:
        probe_threads: list[int] = []

        def synchronize() -> None:
            probe_threads.append(threading.get_ident())

        with (
            patch("worker.mx.synchronize", side_effect=synchronize),
            patch("worker.mx.reset_peak_memory") as reset,
            patch("worker.mx.get_peak_memory", return_value=200),
            patch("worker.mx.get_active_memory", return_value=100),
            patch("worker.mx.get_cache_memory", return_value=50),
        ):
            async with InferenceClient() as client:
                await client.memory_snapshot(reset_peak=True)
                await client.generate("one")
                snapshot = await client.memory_snapshot()
            reset.assert_called_once()
        self.assertEqual(snapshot.peak_bytes, 200)
        self.assertEqual(snapshot.allocator_cache_bytes, 50)
        self.assertEqual(probe_threads, [self.state.thread_ids[0]] * 2)

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
