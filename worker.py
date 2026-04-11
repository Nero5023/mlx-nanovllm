"""One dedicated model-owning thread, a FIFO queue, and thread-safe futures."""

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread

from engine import InferenceEngine
from infer import DEFAULT_MODEL
from request import RequestOutput
from sampler import Sampler


@dataclass(frozen=True)
class _WorkItem:
    prompt: str
    max_tokens: int
    sampler: Sampler | None
    future: Future[RequestOutput]


class InferenceWorker:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        *,
        use_cache: bool = True,
        trace: bool = False,
        collect_stats: bool = False,
    ) -> None:
        self._queue: Queue[_WorkItem | None] = Queue()
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
        self, prompt: str, max_tokens: int = 128, *, sampler: Sampler | None = None
    ) -> Future[RequestOutput]:
        """Enqueue raw text without running the tokenizer or model here."""
        with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("Worker must be started and open before submitting.")
            future: Future[RequestOutput] = Future()
            self._queue.put_nowait(_WorkItem(prompt, max_tokens, sampler, future))
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
                item: _WorkItem | None = self._queue.get_nowait()
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

        while (item := self._queue.get()) is not None:
            # Cancellation can skip queued work, but cannot interrupt a running model.
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                engine.add_request(item.prompt, item.max_tokens, sampler=item.sampler)
                output: RequestOutput | None = engine.run_next()
                if output is None:
                    raise RuntimeError("The submitted request produced no result.")
            except BaseException as error:
                item.future.set_exception(error)
            else:
                item.future.set_result(output)
