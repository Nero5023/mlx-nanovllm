"""One model-owning thread: FIFO admission, continuous decode, individual results."""

from concurrent.futures import Future
from collections import deque
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from time import perf_counter

import mlx.core as mx

from engine import InferenceEngine
from infer import DEFAULT_MODEL
from request import RequestOutput
from sampler import Sampler
from telemetry import MemorySnapshot, RequestTiming, TokenCallback


@dataclass(frozen=True)
class _WorkItem:
    prompt: str
    max_tokens: int
    sampler: Sampler | None
    future: Future[RequestOutput]
    enqueued_at: float
    on_token: TokenCallback | None
    ignore_eos: bool


@dataclass(frozen=True)
class _MemoryProbe:
    reset_peak: bool
    future: Future[MemorySnapshot]
    clear_cache: bool = False


@dataclass(frozen=True)
class _RunningRequest:
    item: _WorkItem
    started_at: float
    batch_size: int  # Size of this request's prefill/admission group.
    batch_id: int


class InferenceWorker:
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
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive.")
        if not isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be finite and nonnegative.")
        if max_batch_size > 1 and (not use_cache or collect_stats):
            raise ValueError("Batching requires use_cache=True and collect_stats=False.")
        self._max_batch_size: int = max_batch_size
        self._batch_wait_seconds: float = batch_wait_ms / 1000
        self._queue: Queue[_WorkItem | _MemoryProbe | None] = Queue()
        # Accessed only by the worker thread after initialization.
        self._running: dict[int, _RunningRequest] = {}
        self._ready: Future[None] = Future()
        self._lock: Lock = Lock()
        self._started: bool = False
        self._closed: bool = False
        self._thread: Thread = Thread(
            target=self._run,
            args=(model_path, use_cache, trace, collect_stats),
            name="mlx-inference-worker",
        )

    def start(self) -> Future[None]:
        """Start model loading in the worker; the future signals readiness."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker is closed.")
            if not self._started:
                self._started = True
                # Loading cannot be interrupted by cancelling a readiness waiter.
                self._ready.set_running_or_notify_cancel()
                try:
                    self._thread.start()
                except BaseException as error:
                    self._started = False
                    self._closed = True
                    self._ready.set_exception(error)
                    raise
            return self._ready

    def submit(
        self, prompt: str, max_tokens: int = 128, *, sampler: Sampler | None = None,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> Future[RequestOutput]:
        """Enqueue raw text without running the tokenizer or model here."""
        with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("Worker must be started and open before submitting.")
            future: Future[RequestOutput] = Future()
            self._queue.put_nowait(_WorkItem(
                prompt, max_tokens, sampler, future, perf_counter(), on_token, ignore_eos
            ))
            return future

    def memory_snapshot(
        self, *, reset_peak: bool = False, clear_cache: bool = False,
    ) -> Future[MemorySnapshot]:
        """Measure on the model thread, after earlier queued requests complete."""
        with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("Worker must be started and open before measuring.")
            future: Future[MemorySnapshot] = Future()
            self._queue.put_nowait(_MemoryProbe(reset_peak, future, clear_cache))
            return future

    def close(self) -> None:
        """Stop accepting work, drain accepted requests, and join the thread.

        This blocks; the async client calls it outside the event loop.
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put_nowait(None)  # FIFO shutdown marker, after pending work.
            started: bool = self._started
        if started:
            self._thread.join()

    def _fail_startup(self, error: BaseException) -> None:
        with self._lock:
            self._closed = True
        # Any submissions accepted while loading also receive the startup error.
        while True:
            try:
                item: _WorkItem | _MemoryProbe | None = self._queue.get_nowait()
            except Empty:
                break
            if item is not None and item.future.set_running_or_notify_cancel():
                item.future.set_exception(error)

    def _run(
        self, model_path: Path, use_cache: bool, trace: bool, collect_stats: bool
    ) -> None:
        try:
            engine: InferenceEngine = InferenceEngine(
                model_path, use_cache=use_cache, trace=trace, collect_stats=collect_stats
            )
        except BaseException as error:
            self._fail_startup(error)
            self._ready.set_exception(error)
            return
        self._ready.set_result(None)

        if self._max_batch_size > 1:
            self._run_continuous(engine)
            return
        deferred: deque[_MemoryProbe | None] = deque()
        while (items := self._next_batch(deferred)) is not None:
            self._run_requests(engine, items)

    def _run_continuous(self, engine: InferenceEngine) -> None:
        deferred: deque[_MemoryProbe | None] = deque()
        while True:
            if not self._running:
                items: list[_WorkItem] | None = self._next_batch(deferred)
                if items is None:
                    return
            else:
                # Never wait to fill slots while existing requests need decode.
                items = self._take_ready(deferred, self._max_batch_size - len(self._running))
            admitted: int = self._admit_requests(engine, items)
            try:
                if admitted:
                    # Do prefil here, but not very ideal, other requests need to wait this to be refilled first.
                    self._deliver_results(engine.prefill_next(admitted))
                if engine.active_requests:
                    self._deliver_results(engine.step())
                if len(self._running) != engine.active_requests:
                    raise RuntimeError("Active rows and outstanding request results disagree.")
            except BaseException as error:
                # A prefill/decode/merge failure invalidates the shared active batch.
                engine.abort_batch()
                for state in self._running.values():
                    state.item.future.set_exception(error)
                self._running.clear()

    def _take_ready(
        self, deferred: deque[_MemoryProbe | None], slots: int,
    ) -> list[_WorkItem]:
        """Read ready requests without crossing a memory/shutdown barrier."""
        items: list[_WorkItem] = []
        while len(items) < slots and not deferred:
            try:
                item: _WorkItem | _MemoryProbe | None = self._queue.get_nowait()
            except Empty:
                break
            if item is not None and item.future.cancelled():
                continue
            if not isinstance(item, _WorkItem):
                deferred.append(item)  # Drain earlier active work before the control message.
                break
            items.append(item)
        return items

    def _admit_requests(
        self, engine: InferenceEngine, items: list[_WorkItem],
    ) -> int:
        started_at: float = perf_counter()
        accepted: list[tuple[int, _WorkItem]] = []
        for item in items:
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                request_id: int = engine.add_request(
                    item.prompt, item.max_tokens, sampler=item.sampler,
                    on_token=item.on_token, ignore_eos=item.ignore_eos,
                )
            except BaseException as error:
                item.future.set_exception(error)
            else:
                accepted.append((request_id, item))
        for request_id, item in accepted:
            self._running[request_id] = _RunningRequest(item, started_at, len(accepted), accepted[0][0])
        return len(accepted)

    def _deliver_results(self, outputs: list[RequestOutput]) -> None:
        """Deliver completed engine outputs to their callers with request timing."""
        for output in outputs:
            state: _RunningRequest = self._running.pop(output.request_id)
            state.item.future.set_result(replace(output, timing=RequestTiming(
                state.item.enqueued_at, state.started_at, perf_counter(),
                batch_size=state.batch_size, batch_id=state.batch_id,
            )))

    def _next_batch(
        self, deferred: deque[_MemoryProbe | None],
    ) -> list[_WorkItem] | None:
        """Collect the next batch, handling control messages in FIFO order.

        A control message found during collection is deferred until the caller
        finishes this batch. Return None when the shutdown marker is reached.
        """
        while True:
            item: _WorkItem | _MemoryProbe | None = (
                deferred.popleft() if deferred else self._queue.get()
            )
            if item is None:
                return None
            if item.future.cancelled():
                continue
            if isinstance(item, _MemoryProbe):
                self._handle_memory_probe(item)
                continue

            items: list[_WorkItem] = [item]
            deadline: float = perf_counter() + self._batch_wait_seconds
            while len(items) < self._max_batch_size:
                try:
                    following: _WorkItem | _MemoryProbe | None = self._queue.get(
                        timeout=max(0.0, deadline - perf_counter())
                    )
                except Empty:
                    break  # Partial batches must make progress even with one client.
                if not isinstance(following, _WorkItem):
                    deferred.append(following)
                    break
                if not following.future.cancelled():
                    items.append(following)
            return items

    def _handle_memory_probe(self, probe: _MemoryProbe) -> None:
        """Measure memory on the worker thread and deliver the result or error."""
        if not probe.future.set_running_or_notify_cancel():
            return
        try:
            mx.synchronize()
            if probe.clear_cache:
                mx.clear_cache()  # Release unused allocator buffers, not live KV/weights.
            if probe.reset_peak:
                mx.reset_peak_memory()
            probe.future.set_result(MemorySnapshot(
                peak_bytes=max(mx.get_peak_memory(), mx.get_active_memory()),
                active_bytes=mx.get_active_memory(),
                allocator_cache_bytes=mx.get_cache_memory(),
            ))
        except BaseException as error:
            probe.future.set_exception(error)

    def _run_requests(self, engine: InferenceEngine, items: list[_WorkItem]) -> None:
        running: list[_WorkItem] = []
        started_at: float = perf_counter()
        for item in items:
            # Cancel before admission: skip. Once admitted, cancellation only
            # discards its result; generation follows the original EOS/budget rules.
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                engine.add_request(
                    item.prompt, item.max_tokens, sampler=item.sampler,
                    on_token=item.on_token, ignore_eos=item.ignore_eos,
                )
            except BaseException as error:
                item.future.set_exception(error)
            else:
                running.append(item)
        if not running:
            return
        try:
            output: RequestOutput | None = engine.run_next()
            outputs: list[RequestOutput] = [output] if output is not None else []
            if len(outputs) != len(running):
                raise RuntimeError("The submitted requests produced an incomplete batch.")
        except BaseException as error:
            # A shared model failure affects this batch; later batches can still run.
            for item in running:
                item.future.set_exception(error)
        else:
            finished_at: float = perf_counter()
            for item, output in zip(running, outputs, strict=True):
                item.future.set_result(replace(output, timing=RequestTiming(
                    item.enqueued_at, started_at, finished_at,
                    batch_size=len(running), batch_id=outputs[0].request_id,
                )))
