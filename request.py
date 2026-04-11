"""A queued inference request and its completed output."""

from dataclasses import dataclass

from benchmark import BenchmarkStats
from sampler import Sampler


@dataclass(frozen=True)
class Request:
    request_id: int
    prompt: str
    prompt_token_ids: list[int]
    max_tokens: int
    sampler: Sampler


@dataclass(frozen=True)
class RequestOutput:
    request_id: int
    prompt: str
    text: str
    token_ids: list[int]
    stats: BenchmarkStats | None
