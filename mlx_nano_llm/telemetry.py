"""Small observations shared by inference and the benchmark client."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    generated_at: float  # time.perf_counter(), after MLX has evaluated this token.


type TokenCallback = Callable[[TokenEvent], None]


@dataclass(frozen=True)
class RequestTiming:
    enqueued_at: float
    started_at: float  # Before tokenization, after leaving the worker queue.
    finished_at: float
    batch_size: int = 1  # Newly admitted group size; active decode may combine several groups.
    batch_id: int | None = None


@dataclass(frozen=True)
class MemorySnapshot:
    peak_bytes: int
    active_bytes: int
    allocator_cache_bytes: int
