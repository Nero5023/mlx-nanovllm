"""Exercise real threads and futures with a controllable stand-in engine."""

import asyncio
import threading
import unittest
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_nano_llm.client import InferenceClient
from mlx_nano_llm.request import RequestOutput
from mlx_nano_llm.sampler import Sampler
from mlx_nano_llm.telemetry import TokenCallback, TokenEvent


@dataclass
class EngineState:
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    loading: threading.Event = field(default_factory=threading.Event)
    loaded: threading.Event = field(default_factory=threading.Event)
    block_loading: bool = False
    calls: list[str] = field(default_factory=list)
    batches: list[list[str]] = field(default_factory=list)
    decode_batches: list[list[str]] = field(default_factory=list)
    block_decode_at: int | None = None
    decoding: threading.Event = field(default_factory=threading.Event)
    decode_release: threading.Event = field(default_factory=threading.Event)
    thread_ids: list[int] = field(default_factory=list)


@dataclass
class FakeRequest:
    request_id: int
    prompt: str
    max_tokens: int
    on_token: TokenCallback | None
    token_ids: list[int] = field(default_factory=list)


class FakeEngine:
    def __init__(self, state: EngineState) -> None:
        self.state: EngineState = state
        self.state.thread_ids.append(threading.get_ident())
        self.state.loading.set()
        if state.block_loading and not state.loaded.wait(timeout=5):
            raise TimeoutError("Test did not release model loading.")
        self.next_id: int = 0
        self.waiting: list[FakeRequest] = []
        self.active: list[FakeRequest] = []

    def add_request(
        self, prompt: str, max_tokens: int, *, sampler: Sampler | None,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> int:
        self.state.thread_ids.append(threading.get_ident())
        if max_tokens < 0:
            raise ValueError("negative budget")
        self.waiting.append(FakeRequest(self.next_id, prompt, max_tokens, on_token))
        self.next_id += 1
        return self.next_id - 1

    def run_next(self) -> RequestOutput:
        self.state.thread_ids.append(threading.get_ident())
        return self._complete(self.waiting.pop(0))

    @property
    def active_requests(self) -> int:
        return len(self.active)

    @property
    def prefilling_requests(self) -> int:
        return 0

    def prefill_next(self, max_batch_size: int, *, chunk_size: int | None = None) -> list[RequestOutput]:
        self.state.thread_ids.append(threading.get_ident())
        selected = self.waiting[:max_batch_size]
        del self.waiting[:max_batch_size]
        self.state.batches.append([item.prompt for item in selected])
        for item in selected:
            self._start(item)
        finished, surviving = self._advance(selected)
        self.active.extend(surviving)
        return finished

    def step(self) -> list[RequestOutput]:
        self.state.thread_ids.append(threading.get_ident())
        self.state.decode_batches.append([item.prompt for item in self.active])
        if len(self.state.decode_batches) == self.state.block_decode_at:
            self.state.decoding.set()
            if not self.state.decode_release.wait(timeout=5):
                raise TimeoutError("Test did not release decode.")
        finished, self.active = self._advance(self.active)
        return finished

    def abort_batch(self) -> None:
        self.active.clear()

    def _advance(self, items: list[FakeRequest]) -> tuple[list[RequestOutput], list[FakeRequest]]:
        finished: list[RequestOutput] = []
        surviving: list[FakeRequest] = []
        for item in items:
            if len(item.token_ids) < item.max_tokens:
                item.token_ids.append(item.request_id)
                if item.on_token is not None:
                    item.on_token(TokenEvent(item.request_id, perf_counter()))
            if len(item.token_ids) == item.max_tokens:
                finished.append(RequestOutput(
                    item.request_id, item.prompt, f"answer:{item.prompt}", item.token_ids, None,
                ))
            else:
                surviving.append(item)
        return finished, surviving

    def _start(self, item: FakeRequest) -> None:
        self.state.calls.append(item.prompt)
        if item.prompt == "blocked":
            self.state.started.set()
            if not self.state.release.wait(timeout=5):
                raise TimeoutError("Test did not release inference.")
        if item.prompt == "error":
            raise ValueError("request failed")

    def _complete(self, item: FakeRequest) -> RequestOutput:
        self._start(item)
        if item.on_token is not None:
            item.on_token(TokenEvent(item.request_id, perf_counter()))
        return RequestOutput(item.request_id, item.prompt, f"answer:{item.prompt}", [item.request_id], None)


class ChunkedFakeEngine(FakeEngine):
    """Three resumable prefill chunks for incoming prompts; one for ordinary prompts."""

    def __init__(self, state: EngineState, events: list[str]) -> None:
        super().__init__(state)
        self.events = events
        self.incoming: list[FakeRequest] = []
        self.remaining = 0

    @property
    def prefilling_requests(self) -> int:
        return len(self.incoming)

    def prefill_next(self, max_batch_size: int, *, chunk_size: int | None = None) -> list[RequestOutput]:
        if not self.incoming:
            self.incoming = self.waiting[:max_batch_size]
            del self.waiting[:max_batch_size]
            self.remaining = 3 if any(r.prompt.startswith('incoming') for r in self.incoming) else 1
            for item in self.incoming:
                self._start(item)
        self.events.append(f'prefill:{self.incoming[0].prompt}:{self.remaining}')
        if self.remaining == 2 and self.incoming[0].prompt == 'incoming-error':
            raise RuntimeError('chunk failed')
        self.remaining -= 1
        if self.remaining:
            return []
        finished, surviving = self._advance(self.incoming)
        self.incoming = []
        self.active.extend(surviving)
        return finished

    def step(self) -> list[RequestOutput]:
        self.events.append('decode')
        return super().step()

    def abort_batch(self) -> None:
        super().abort_batch()
        self.incoming.clear()


class InferenceClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.state: EngineState = EngineState()

        def create_engine(
            model_path: Path, *, use_cache: bool, trace: bool, collect_stats: bool
        ) -> FakeEngine:
            return FakeEngine(self.state)

        self.factory: MagicMock = self.enterContext(
            patch("mlx_nano_llm.worker.InferenceEngine", side_effect=create_engine)
        )

    async def wait_for_event(self, event: threading.Event) -> None:
        self.assertTrue(await asyncio.to_thread(event.wait, 2))

    async def test_chunked_prefill_yields_to_decode_between_chunks(self) -> None:
        events: list[str] = []
        self.factory.side_effect = lambda *a, **kw: ChunkedFakeEngine(self.state, events)
        self.state.block_decode_at = 1
        async with InferenceClient(max_batch_size=2, prefill_chunk_size=2) as client:
            long = client.submit('long', 12)
            try:
                await self.wait_for_event(self.state.decoding)
                incoming = client.submit('incoming', 2)
            finally:
                self.state.decode_release.set()
            await asyncio.wait_for(asyncio.gather(long, incoming), 2)
        positions = [i for i, event in enumerate(events) if event.startswith('prefill:incoming:')]
        self.assertEqual(len(positions), 3)
        for a, b in zip(positions, positions[1:]):
            self.assertIn('decode', events[a + 1:b])
        self.assertTrue(all(len(rows) <= 2 for rows in self.state.decode_batches))

    async def test_middle_chunk_failure_fails_both_groups_and_recovers(self) -> None:
        events: list[str] = []
        self.factory.side_effect = lambda *a, **kw: ChunkedFakeEngine(self.state, events)
        self.state.block_decode_at = 1
        async with InferenceClient(max_batch_size=2, prefill_chunk_size=2) as client:
            long = client.submit('long', 12)
            try:
                await self.wait_for_event(self.state.decoding)
                incoming = client.submit('incoming-error', 2)
            finally:
                self.state.decode_release.set()
            for future in (long, incoming):
                with self.assertRaisesRegex(RuntimeError, 'chunk failed'):
                    await asyncio.wait_for(future, 2)
            recovered = await asyncio.wait_for(client.generate('recovered', 2), 2)
            self.assertEqual(recovered.prompt, 'recovered')

    async def test_shutdown_drains_chunked_prefill_even_without_decode_rows(self) -> None:
        events: list[str] = []
        self.factory.side_effect = lambda *a, **kw: ChunkedFakeEngine(self.state, events)
        client = InferenceClient(max_batch_size=1, prefill_chunk_size=2)
        await client.start()
        future = client.submit('incoming', 3)
        await asyncio.wait_for(client.close(), 2)
        self.assertEqual(len((await future).token_ids), 3)
        self.assertEqual(sum(e.startswith('prefill:incoming:') for e in events), 3)
        self.assertFalse(client._worker._thread.is_alive())

    async def test_chunked_prefill_options_are_validated_before_start(self) -> None:
        for chunk, cache, stats in ((0, True, False), (-1, True, False),
                                    (2, False, False), (2, True, True)):
            with self.subTest(chunk=chunk, cache=cache, stats=stats), self.assertRaises(ValueError):
                InferenceClient(prefill_chunk_size=chunk, use_cache=cache, collect_stats=stats)

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
            patch("mlx_nano_llm.worker.mx.synchronize", side_effect=synchronize),
            patch("mlx_nano_llm.worker.mx.reset_peak_memory") as reset,
            patch("mlx_nano_llm.worker.mx.clear_cache") as clear,
            patch("mlx_nano_llm.worker.mx.get_peak_memory", return_value=200),
            patch("mlx_nano_llm.worker.mx.get_active_memory", return_value=100),
            patch("mlx_nano_llm.worker.mx.get_cache_memory", return_value=50),
        ):
            async with InferenceClient() as client:
                await client.memory_snapshot(reset_peak=True, clear_cache=True)
                await client.generate("one")
                snapshot = await client.memory_snapshot()
            reset.assert_called_once()
            clear.assert_called_once()
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

    async def test_full_batch_waits_for_a_slot_and_preserves_fifo_admission(self) -> None:
        async with InferenceClient(max_batch_size=2, batch_wait_ms=100) as client:
            first = client.submit("blocked")
            second = client.submit("second")
            try:
                await self.wait_for_event(self.state.started)
                third = client.submit("third")
                fourth = client.submit("fourth")
                await asyncio.sleep(0)
                self.assertEqual(self.state.batches, [["blocked", "second"]])
                self.assertFalse(third.done())
            finally:
                self.state.release.set()
            outputs = await asyncio.wait_for(asyncio.gather(first, second, third, fourth), 2)
        self.assertEqual(self.state.batches, [["blocked", "second"], ["third", "fourth"]])
        self.assertEqual([out.request_id for out in outputs], [0, 1, 2, 3])
        self.assertEqual([out.timing.batch_size for out in outputs if out.timing], [2] * 4)
        self.assertEqual([out.timing.batch_id for out in outputs if out.timing], [0, 0, 2, 2])
        self.assertEqual(len(set(self.state.thread_ids)), 1)

    async def test_finished_result_returns_and_queued_request_replaces_it_before_long_request_finishes(self) -> None:
        self.state.block_decode_at = 2
        events: list[TokenEvent] = []
        async with InferenceClient(max_batch_size=2, batch_wait_ms=100) as client:
            long = client.submit("blocked", max_tokens=8)
            short = client.submit("short", max_tokens=2, on_token=events.append)
            try:
                await self.wait_for_event(self.state.started)
                replacement = client.submit("replacement", max_tokens=2)
                self.state.release.set()
                await self.wait_for_event(self.state.decoding)
                result = await asyncio.wait_for(asyncio.shield(short), 2)
                self.assertEqual([event.token_id for event in events], result.token_ids)
                self.assertFalse(long.done())
                self.assertEqual(self.state.batches, [["blocked", "short"], ["replacement"]])
                self.assertEqual(self.state.decode_batches[1], ["blocked", "replacement"])
            finally:
                self.state.release.set()
                self.state.decode_release.set()
            outputs = await asyncio.wait_for(asyncio.gather(long, replacement), 2)
        self.assertEqual([len(output.token_ids) for output in outputs], [8, 2])
        self.assertTrue(all(len(rows) <= 2 for rows in self.state.decode_batches))
        assert result.timing is not None and outputs[0].timing is not None
        self.assertLess(result.timing.finished_at, outputs[0].timing.finished_at)

    async def test_arrival_joins_an_underfilled_active_batch(self) -> None:
        self.state.block_decode_at = 2
        async with InferenceClient(max_batch_size=3, batch_wait_ms=0) as client:
            first = client.submit("blocked", max_tokens=6)
            try:
                await self.wait_for_event(self.state.started)
                newcomer = client.submit("newcomer", max_tokens=2)
                self.state.release.set()
                await self.wait_for_event(self.state.decoding)
                self.assertEqual(self.state.decode_batches[1], ["blocked", "newcomer"])
                self.assertFalse(first.done())
            finally:
                self.state.release.set()
                self.state.decode_release.set()
            outputs = await asyncio.wait_for(asyncio.gather(first, newcomer), 2)
        self.assertEqual([len(output.token_ids) for output in outputs], [6, 2])

    async def test_partial_batch_and_invalid_request_make_progress(self) -> None:
        async with InferenceClient(max_batch_size=4, batch_wait_ms=5) as client:
            bad = client.submit("invalid", max_tokens=-1)
            good = client.submit("good")
            with self.assertRaisesRegex(ValueError, "negative budget"):
                await bad
            result = await asyncio.wait_for(good, 2)
            self.assertEqual(result.prompt, "good")
            assert result.timing is not None
            self.assertEqual(result.timing.batch_size, 1)
            self.assertEqual((await asyncio.wait_for(client.generate("alone"), 2)).prompt, "alone")

    async def test_failed_batch_does_not_poison_next_batch(self) -> None:
        async with InferenceClient(max_batch_size=2, batch_wait_ms=100) as client:
            failed = client.submit("error")
            affected = client.submit("same batch")
            results = await asyncio.gather(failed, affected, return_exceptions=True)
            self.assertTrue(all(isinstance(result, ValueError) for result in results))
            self.assertEqual((await client.generate("later")).prompt, "later")

    async def test_incoming_prefill_failure_fails_active_requests_and_worker_recovers(self) -> None:
        async with InferenceClient(max_batch_size=2, batch_wait_ms=0) as client:
            active = client.submit("blocked", max_tokens=8)
            try:
                await self.wait_for_event(self.state.started)
                incoming = client.submit("error")
                later = client.submit("later", max_tokens=1)
            finally:
                self.state.release.set()
            results = await asyncio.wait_for(asyncio.gather(active, incoming, return_exceptions=True), 2)
            self.assertTrue(all(isinstance(result, ValueError) for result in results))
            self.assertEqual((await asyncio.wait_for(later, 2)).prompt, "later")

    async def test_memory_probe_is_a_batch_barrier(self) -> None:
        observed_calls: list[list[str]] = []
        with (
            patch("mlx_nano_llm.worker.mx.synchronize", side_effect=lambda: observed_calls.append(self.state.calls.copy())),
            patch("mlx_nano_llm.worker.mx.get_peak_memory", return_value=200),
            patch("mlx_nano_llm.worker.mx.get_active_memory", return_value=100),
            patch("mlx_nano_llm.worker.mx.get_cache_memory", return_value=50),
        ):
            async with InferenceClient(max_batch_size=4, batch_wait_ms=0) as client:
                first = client.submit("blocked")
                try:
                    await self.wait_for_event(self.state.started)
                    probe = asyncio.create_task(client.memory_snapshot())
                    await asyncio.sleep(0)  # Put the probe in the FIFO before the next request.
                    second = client.submit("after probe")
                    self.assertFalse(probe.done())
                finally:
                    self.state.release.set()
                await asyncio.wait_for(asyncio.gather(first, probe, second), 2)
        self.assertEqual(observed_calls, [["blocked"]])
        self.assertEqual(self.state.batches, [["blocked"], ["after probe"]])

    async def test_batch_cancellation_and_shutdown_drain(self) -> None:
        client: InferenceClient = InferenceClient(max_batch_size=2, batch_wait_ms=100)
        self.addAsyncCleanup(client.close)
        await client.start()
        first = client.submit("blocked")
        second = client.submit("second")
        try:
            await self.wait_for_event(self.state.started)
            cancelled = client.submit("cancelled")
            cancelled.cancel()
            first.cancel()
            await asyncio.sleep(0)
            last = client.submit("last")
            closing = asyncio.create_task(client.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
        finally:
            self.state.release.set()
        await asyncio.wait_for(closing, 2)
        self.assertTrue(first.cancelled())
        self.assertTrue(cancelled.cancelled())
        self.assertEqual([out.prompt for out in await asyncio.gather(second, last)], ["second", "last"])
        self.assertEqual(self.state.batches, [["blocked", "second"], ["last"]])
        self.assertFalse(client._worker._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
