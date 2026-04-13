"""Check metric arithmetic and load generation without real inference."""

import asyncio
import unittest
from time import perf_counter

from bench import run_case
from bench_metrics import Observation, distribution, intervals_ms, summarize
from bench_workloads import Case, RequestSpec, common_prefix_tokens
from request import RequestOutput
from telemetry import MemorySnapshot, RequestTiming, TokenCallback, TokenEvent


class MetricTests(unittest.TestCase):
    def test_queue_ttft_tpot_and_delivery_are_distinct(self) -> None:
        observation: Observation = Observation(
            "one", planned_at=1.0, submitted_at=1.0, completed_at=2.1,
            generated_at=[1.4, 1.7, 1.9], received_at=[1.5, 1.8, 2.0],
            event_token_ids=[1, 2, 3], output_token_ids=[1, 2, 3], prompt_tokens=4,
            timing=RequestTiming(1.0, 1.2, 2.05),
        )
        metrics = observation.metrics()
        expected: dict[str, float] = {
            "ttft_ms": 500.0, "tpot_ms": 250.0, "queue_ms": 200.0,
            "service_ttft_ms": 200.0, "worker_ttft_ms": 400.0, "e2e_ms": 1100.0,
        }
        for key, expected_value in expected.items():
            value: float | int | None = metrics[key]
            assert value is not None
            self.assertAlmostEqual(value, expected_value)
        gaps: list[float] = intervals_ms(observation.received_at)
        self.assertAlmostEqual(gaps[0], 300.0)
        self.assertAlmostEqual(gaps[1], 200.0)

    def test_throughput_uses_whole_window_and_counts_failures(self) -> None:
        observations: list[Observation] = [
            Observation("a", 10, 10, completed_at=12, output_token_ids=[1, 2, 3]),
            Observation("b", 10.5, 10.5, completed_at=11.5, output_token_ids=[4]),
            Observation("failed", 11, 11, completed_at=13, output_token_ids=[5], error="failed"),
        ]
        result = summarize(observations)
        self.assertEqual(result["duration_seconds"], 3)
        self.assertEqual(result["output_tokens_per_second"], 4 / 3)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["max_client_inflight"], 3)

    def test_zero_and_one_token_requests_have_no_decode_latency(self) -> None:
        zero: Observation = Observation("zero", 0, 0, completed_at=1)
        one: Observation = Observation("one", 0, 0, completed_at=1, received_at=[0.5])
        self.assertIsNone(zero.metrics()["ttft_ms"])
        self.assertIsNone(zero.metrics()["tpot_ms"])
        self.assertEqual(one.metrics()["ttft_ms"], 500)
        self.assertIsNone(one.metrics()["tpot_ms"])
        self.assertEqual(distribution([])["n"], 0)
        self.assertIsNone(distribution([])["p95"])
        self.assertEqual(distribution([1, 3])["p50"], 2)

    def test_shared_prefix_is_measured_in_tokens(self) -> None:
        requests: list[RequestSpec] = [
            RequestSpec("a", "a", [1, 2, 3], 2),
            RequestSpec("b", "b", [1, 2, 4], 2),
        ]
        self.assertEqual(common_prefix_tokens(requests), 2)


class FakeBackend:
    def __init__(self, *, wait_for_second: bool = False, drop_token: bool = False) -> None:
        self.wait_for_second: bool = wait_for_second
        self.drop_token: bool = drop_token
        self.second_arrived: asyncio.Event = asyncio.Event()
        self.started: list[str] = []
        self.completed: list[str] = []
        self.memory_resets: list[bool] = []

    async def memory_snapshot(
        self, *, reset_peak: bool = False, clear_cache: bool = False,
    ) -> MemorySnapshot:
        self.memory_resets.append(reset_peak)
        return MemorySnapshot(200, 100, 50)

    async def generate(
        self, prompt: str, max_tokens: int = 128, *,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> RequestOutput:
        self.started.append(prompt)
        if prompt == "second":
            self.second_arrived.set()
        if prompt == "first" and self.wait_for_second:
            await asyncio.wait_for(self.second_arrived.wait(), 2)
        if prompt == "error":
            raise ValueError("request failure")
        assert ignore_eos
        tokens: list[int] = list(range(max_tokens))
        for token in tokens:
            if on_token is not None and not self.drop_token:
                on_token(TokenEvent(token, perf_counter()))
            await asyncio.sleep(0)
        self.completed.append(prompt)
        return RequestOutput(0, prompt, "", tokens, None, prompt_tokens=2)


class LoadGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_staggered_arrival_does_not_wait_for_previous_response(self) -> None:
        backend: FakeBackend = FakeBackend(wait_for_second=True)
        case: Case = Case("arrivals", "test", [
            RequestSpec("a", "first", [1, 2], 2),
            RequestSpec("b", "second", [1, 2], 2, arrival_seconds=0.01),
        ])
        run = await run_case(backend, case, 1)
        self.assertEqual([item.error for item in run.observations], [None, None])
        self.assertEqual(backend.started, ["first", "second"])
        self.assertEqual(backend.memory_resets, [True, False])
        self.assertEqual(summarize(run.observations)["max_client_inflight"], 2)

    async def test_closed_loop_waits_and_failures_are_recorded(self) -> None:
        backend: FakeBackend = FakeBackend()
        case: Case = Case("closed", "test", [
            RequestSpec("a", "first", [1, 2], 2),
            RequestSpec("b", "error", [1, 2], 2),
            RequestSpec("c", "second", [1, 2], 2),
        ], sequential=True)
        run = await run_case(backend, case, 1)
        self.assertLessEqual(run.observations[0].completed_at, run.observations[1].submitted_at)
        self.assertEqual(summarize(run.observations)["failed"], 1)
        self.assertEqual(backend.completed, ["first", "second"])

    async def test_missing_token_events_fail_instead_of_inventing_ttft(self) -> None:
        backend: FakeBackend = FakeBackend(drop_token=True)
        case: Case = Case("broken", "test", [RequestSpec("a", "first", [1, 2], 2)])
        run = await run_case(backend, case, 1)
        self.assertIn("token notifications", run.observations[0].error or "")
        self.assertEqual(summarize(run.observations)["failed"], 1)


if __name__ == "__main__":
    unittest.main()
