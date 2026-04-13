"""Metric definitions; all raw timestamps use the same monotonic clock."""

from dataclasses import dataclass, field
from statistics import mean

from telemetry import RequestTiming


@dataclass
class Observation:
    name: str
    planned_at: float
    submitted_at: float
    completed_at: float = 0.0
    generated_at: list[float] = field(default_factory=list)
    received_at: list[float] = field(default_factory=list)
    event_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    timing: RequestTiming | None = None
    error: str | None = None

    def metrics(self) -> dict[str, float | int | None]:
        received: list[float] = self.received_at
        generated: list[float] = self.generated_at
        timing: RequestTiming | None = self.timing
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": len(self.output_token_ids),
            "submission_lag_ms": (self.submitted_at - self.planned_at) * 1000,
            "e2e_ms": (self.completed_at - self.submitted_at) * 1000,
            "ttft_ms": (received[0] - self.submitted_at) * 1000 if received else None,
            "tpot_ms": (received[-1] - received[0]) * 1000 / (len(received) - 1)
            if len(received) > 1 else None,
            "queue_ms": (timing.started_at - timing.enqueued_at) * 1000 if timing else None,
            "service_ttft_ms": (generated[0] - timing.started_at) * 1000
            if generated and timing else None,
            "worker_ttft_ms": (generated[0] - timing.enqueued_at) * 1000
            if generated and timing else None,
        }


def intervals_ms(timestamps: list[float]) -> list[float]:
    return [(end - start) * 1000 for start, end in zip(timestamps, timestamps[1:])]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """Linear interpolation; report n so tiny samples aren't mistaken for stable tails."""
    ordered: list[float] = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position: float = (len(ordered) - 1) * fraction
        lower: int = int(position)
        upper: int = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "n": len(values), "mean": mean(values) if values else None,
        "p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99),
        "max": max(values) if values else None,
    }


def summarize(observations: list[Observation]) -> dict[str, object]:
    successful: list[Observation] = [item for item in observations if item.error is None]
    duration: float = (
        max(item.completed_at for item in observations)
        - min(item.submitted_at for item in observations)
    ) if observations else 0.0
    output_tokens: int = sum(len(item.output_token_ids) for item in successful)
    request_metrics: list[dict[str, float | int | None]] = [
        item.metrics() for item in successful
    ]
    distributions: dict[str, object] = {}
    for key in ("ttft_ms", "tpot_ms", "queue_ms", "service_ttft_ms", "worker_ttft_ms",
                "e2e_ms", "submission_lag_ms"):
        values: list[float] = []
        for metrics in request_metrics:
            value: float | int | None = metrics[key]
            if value is not None:
                values.append(float(value))
        distributions[key] = distribution(values)

    events: list[tuple[float, int]] = []
    for item in observations:
        events.extend([(item.submitted_at, 1), (item.completed_at, -1)])
    inflight: int = 0
    max_inflight: int = 0
    for _, change in sorted(events):
        inflight += change
        max_inflight = max(max_inflight, inflight)
    return {
        "requests": len(observations), "succeeded": len(successful),
        "failed": len(observations) - len(successful),
        "duration_seconds": duration,
        "prompt_tokens": sum(item.prompt_tokens for item in successful),
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / duration if duration > 0 else None,
        "requests_per_second": len(successful) / duration if duration > 0 else None,
        "max_client_inflight": max_inflight,
        "actual_batch_sizes": sorted({item.timing.batch_size for item in successful if item.timing}),
        "completed_batches": len({item.timing.batch_id for item in successful
                                  if item.timing and item.timing.batch_id is not None}),
        **distributions,
        "itl_ms": distribution([gap for item in successful for gap in intervals_ms(item.received_at)]),
        "worker_itl_ms": distribution([
            gap for item in successful for gap in intervals_ms(item.generated_at)
        ]),
        "delivery_ms": distribution([
            (received - generated) * 1000 for item in successful
            for received, generated in zip(item.received_at, item.generated_at)
        ]),
    }
