"""Repeatable serving workloads for the current FIFO engine and future backends."""

import argparse
import asyncio
import hashlib
import json
from math import isfinite
import platform
import subprocess
from dataclasses import asdict, dataclass
from uuid import uuid4
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Protocol

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from bench_metrics import Observation, distribution, summarize
from bench_workloads import Case, RequestSpec, build_cases, common_prefix_tokens, make_request
from client import InferenceClient
from infer import DEFAULT_MODEL
from request import RequestOutput
from telemetry import MemorySnapshot, TokenCallback, TokenEvent


class BenchmarkBackend(Protocol):
    """A future batch engine can implement this without changing the workloads.

    Call on_token in order on the client event loop, before returning the output.
    Reset memory only between cases, never per request. Run one backend at a time:
    MLX's allocator counters are process-wide.
    """

    async def generate(
        self, prompt: str, max_tokens: int = 128, *,
        on_token: TokenCallback | None = None, ignore_eos: bool = False,
    ) -> RequestOutput: ...

    async def memory_snapshot(
        self, *, reset_peak: bool = False, clear_cache: bool = False,
    ) -> MemorySnapshot: ...


@dataclass
class CaseRun:
    case: str
    repeat: int
    observations: list[Observation]
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot

    def to_dict(self) -> dict[str, object]:
        return {
            "case": self.case, "repeat": self.repeat,
            "summary": summarize(self.observations),
            "memory_before": asdict(self.memory_before),
            "memory_after": asdict(self.memory_after),
            "requests": [{**asdict(item), "metrics": item.metrics()} for item in self.observations],
        }


async def run_case(backend: BenchmarkBackend, case: Case, repeat: int) -> CaseRun:
    memory_before: MemorySnapshot = await backend.memory_snapshot(reset_peak=True, clear_cache=True)
    origin: float = perf_counter()

    async def send(spec: RequestSpec, planned_at: float) -> Observation:
        await asyncio.sleep(max(0.0, planned_at - perf_counter()))
        observation: Observation = Observation(spec.name, planned_at, perf_counter())

        def receive(event: TokenEvent) -> None:
            observation.received_at.append(perf_counter())
            observation.generated_at.append(event.generated_at)
            observation.event_token_ids.append(event.token_id)

        try:
            output: RequestOutput = await backend.generate(
                spec.prompt, spec.max_tokens, on_token=receive, ignore_eos=True
            )
            observation.completed_at = perf_counter()
            observation.output_token_ids = output.token_ids
            observation.prompt_tokens = output.prompt_tokens
            observation.timing = output.timing
            if output.prompt_tokens != len(spec.prompt_token_ids):
                raise ValueError("Backend and workload prompt token counts differ.")
            if observation.event_token_ids != output.token_ids:
                raise ValueError("Missing, out-of-order, or incorrect token notifications.")
            if len(output.token_ids) != spec.max_tokens:
                raise ValueError("Backend did not honor the fixed output length (ignore_eos).")
        except Exception as error:
            observation.completed_at = perf_counter()
            observation.error = f"{type(error).__name__}: {error}"
        return observation

    observations: list[Observation]
    if case.sequential:
        observations = []
        for spec in case.requests:
            observations.append(await send(spec, perf_counter()))
    else:
        # Open-loop scheduling: a slow request does not delay later submissions.
        observations = await asyncio.gather(*(
            send(spec, origin + spec.arrival_seconds) for spec in case.requests
        ))
    memory_after: MemorySnapshot = await backend.memory_snapshot()
    return CaseRun(case.name, repeat, observations, memory_before, memory_after)


async def check_correctness(
    model_path: Path, max_batch_size: int = 1, batch_wait_ms: float = 2.0,
) -> dict[str, object]:
    """Greedy equivalence and request-cache isolation, outside performance timing."""
    prompts: list[str] = ["What is the capital of France?", "What is 2 + 2?"]
    cached: list[list[int]] = []
    async with InferenceClient(model_path) as client:
        for prompt in prompts:
            cached.append((await client.generate(prompt, 24)).token_ids)
        repeated: list[int] = (await client.generate(prompts[0], 24)).token_ids
        empty: list[int] = (await client.generate(prompts[0], 0)).token_ids
        one: list[int] = (await client.generate(prompts[0], 1)).token_ids
    uncached: list[list[int]] = []
    async with InferenceClient(model_path, use_cache=False) as client:
        for prompt in prompts:
            uncached.append((await client.generate(prompt, 24)).token_ids)
    checks: dict[str, bool] = {
        "cached_equals_full_recompute": cached == uncached,
        "a_b_a_cache_isolation": cached[0] == repeated,
        "zero_output_budget": empty == [],
        "one_output_budget": one == cached[0][:1],
    }
    batch_checks: dict[str, object] = {}
    refill_checks: dict[str, object] = {}
    if max_batch_size > 1:
        mixed_prompts: list[str] = prompts + ["Count from 1 to 20, separated by commas.", "Say hello."]
        budgets: list[int] = [24, 16, 32, 0]
        reference: list[list[int]] = []
        async with InferenceClient(model_path) as client:
            for prompt, budget in zip(mixed_prompts, budgets):
                reference.append((await client.generate(prompt, budget)).token_ids)
        async with InferenceClient(
            model_path, max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms,
        ) as client:
            batch_outputs: list[RequestOutput] = await asyncio.gather(*(
                client.generate(prompt, budget) for prompt, budget in zip(mixed_prompts, budgets)
            ))
            reversed_outputs: list[RequestOutput] = await asyncio.gather(*(
                client.generate(prompt, budget)
                for prompt, budget in reversed(list(zip(mixed_prompts, budgets)))
            ))
            partial: RequestOutput = await client.generate(prompts[0], 24)
            first_token: asyncio.Event = asyncio.Event()

            def started(event: TokenEvent) -> None:
                first_token.set()

            long_request: asyncio.Future[RequestOutput] = client.submit(
                mixed_prompts[2], 32, on_token=started,
            )
            # Also wake on failure/early completion instead of waiting forever for a token.
            long_request.add_done_callback(lambda _: first_token.set())
            await first_token.wait()
            newcomer: RequestOutput = await client.generate(prompts[0], 4)
            long_output: RequestOutput = await long_request
        checks["fixed_batch_equals_serial"] = [out.token_ids for out in batch_outputs] == reference
        checks["reordered_batch_equals_serial"] = [out.token_ids for out in reversed_outputs] == reference[::-1]
        checks["partial_batch_equals_serial"] = partial.token_ids == reference[0]
        checks["continuous_refill_equals_serial"] = (
            long_output.token_ids == reference[2] and newcomer.token_ids == reference[0][:4]
        )
        checks["newcomer_completes_before_long_request"] = (
            newcomer.timing is not None and long_output.timing is not None
            and newcomer.timing.finished_at < long_output.timing.finished_at
        )
        refill_checks = {
            "long_token_ids": long_output.token_ids, "newcomer_token_ids": newcomer.token_ids,
            "long_timing": asdict(long_output.timing) if long_output.timing else None,
            "newcomer_timing": asdict(newcomer.timing) if newcomer.timing else None,
        }
        batch_checks = {
            "prompts": mixed_prompts, "budgets": budgets, "serial_token_ids": reference,
            "batch_token_ids": [out.token_ids for out in batch_outputs],
            "reversed_batch_token_ids": [out.token_ids for out in reversed_outputs],
            "observed_batch_sizes": [out.timing.batch_size if out.timing else None for out in batch_outputs],
        }
    return {
        "passed": all(checks.values()), "checks": checks, "prompts": prompts,
        "cached_token_ids": cached, "uncached_token_ids": uncached,
        "scope": "Regression checks, not an independent model-quality oracle.",
        "fixed_batch": batch_checks,
        "continuous_refill": refill_checks,
    }


def command_output(arguments: list[str]) -> str | None:
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            arguments, capture_output=True, text=True, timeout=10, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def metadata(
    model_path: Path, use_cache: bool, max_batch_size: int = 1, batch_wait_ms: float = 2.0,
) -> dict[str, object]:
    source_hashes: dict[str, str] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": source_hashes,
        "platform": platform.platform(), "machine": platform.machine(),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "model": str(model_path.resolve()),
        "model_config": json.loads((model_path / "config.json").read_text()),
        "model_file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(model_path.iterdir())
            if path.suffix in {".json", ".safetensors"}
        },
        "backend": "continuous-batch-worker" if max_batch_size > 1 else "fifo-worker",
        "batch_compaction": max_batch_size > 1,
        "continuous_batching": max_batch_size > 1,
        "batch_size_scope": "Newly admitted group size; decode can merge survivors from several groups.",
        "prefill_policy": "Whole incoming group; active decode pauses during prefill; no chunking.",
        "max_active_requests": max_batch_size,
        "batch_wait_ms": batch_wait_ms if max_batch_size > 1 else 0.0,
        "use_kv_cache": use_cache, "prefix_reuse": False, "paged_kv": False,
        "sampler": "greedy", "ignore_eos_in_performance_cases": True,
        "warmup": "Unrelated 128-token prompt, 4 generated tokens; fresh client per case/repeat.",
        "allocator_policy": "Release unused allocator buffers after warmup, before each case's peak reset.",
        "memory": "MLX peak active allocation including resident weights; allocator cache separate; not RSS.",
        "timestamps": "time.perf_counter seconds, shared process clock; client callbacks, no network.",
        "gpu_utilization_percent": None,
    }


def format_number(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def write_report(
    directory: Path, info: dict[str, object], cases: list[Case],
    correctness: dict[str, object], runs: list[CaseRun],
) -> None:
    records: list[dict[str, object]] = [run.to_dict() for run in runs]
    payload: dict[str, object] = {
        "schema_version": 1, "metadata": info, "correctness": correctness,
        "workloads": [{**asdict(case), "common_prefix_tokens": common_prefix_tokens(case.requests)}
                      for case in cases],
        "runs": records,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines: list[str] = [
        f"# {info['backend']} benchmark", "",
        f"Correctness: **{'PASS' if correctness.get('passed') else 'FAIL'}**. "
        f"Chip: {info['chip']}. Max active requests: {info['max_active_requests']}, "
        f"batch wait: {info['batch_wait_ms']} ms, KV cache={info['use_kv_cache']}.", "",
        "TTFT includes queueing and client notification. ITL pools consecutive token gaps; "
        "TPOT averages gaps per request. Values are milliseconds. Throughput is the median "
        "of per-repeat output tokens / total workload wall time; memory is the maximum across repeats.", "",
        "| Case | Runs | OK / all | Admission group sizes | TTFT p50 / p95 | ITL p50 / p95 | TPOT p50 | Queue p50 | Output tok/s p50 | Peak GB |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        matching: list[CaseRun] = [run for run in runs if run.case == case.name]
        if not matching:
            continue
        observations: list[Observation] = [item for run in matching for item in run.observations]
        successful: list[Observation] = [item for item in observations if item.error is None]
        batch_sizes: list[int] = sorted({item.timing.batch_size for item in successful if item.timing})

        def metric(key: str, percentile: str = "p50") -> str:
            values: list[float] = []
            for item in successful:
                value: float | int | None = item.metrics()[key]
                if value is not None:
                    values.append(float(value))
            return format_number(distribution(values)[percentile])

        gaps: list[float] = [
            (end - start) * 1000 for item in successful
            for start, end in zip(item.received_at, item.received_at[1:])
        ]
        throughput: list[float] = []
        for run in matching:
            value: object = summarize(run.observations)["output_tokens_per_second"]
            if isinstance(value, (float, int)):
                throughput.append(float(value))
        lines.append(
            f"| {case.name} | {len(matching)} | {len(successful)} / {len(observations)} "
            f"| {', '.join(map(str, batch_sizes)) or '—'} "
            f"| {metric('ttft_ms')} / {metric('ttft_ms', 'p95')} "
            f"| {format_number(distribution(gaps)['p50'])} / {format_number(distribution(gaps)['p95'])} "
            f"| {metric('tpot_ms')} | {metric('queue_ms')} "
            f"| {format_number(distribution(throughput)['p50'])} "
            f"| {max(run.memory_after.peak_bytes for run in matching) / 1e9:.3f} |"
        )
    lines.extend([
        "", "Raw JSON includes p99 and sample counts, worker token timestamps, client arrival "
        "timestamps, submission lag, per-request input/output lengths, token IDs, errors, "
        "end-to-end latency and allocator cache memory.", "",
        "These are short synthetic workloads, not steady-state production load. Small-sample "
        "tail percentiles are descriptive. No GPU utilization or hardware capacity limit is measured. "
        "Queued requests have no active KV cache; active batch rows do. Concurrency is client "
        "in-flight work; admission group sizes do not track growing/shrinking decode tensors.", "",
        "The largest successful context/concurrency is only the largest tested configuration. "
        "Prefix scenarios measure the current baseline; this backend has no cross-request prefix cache.", "",
        "See BENCHMARKS.md for formulas, workload purposes and backend comparison rules.",
    ])
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    cases: list[Case] = build_cases(
        tokenizer, profile=args.profile,
        max_prompt_tokens=args.max_prompt_tokens, max_concurrency=args.max_concurrency,
    )
    if args.case:
        unknown: set[str] = set(args.case) - {case.name for case in cases}
        if unknown:
            raise ValueError(f"Unknown cases for this profile: {sorted(unknown)}")
        cases = [case for case in cases if case.name in args.case]
    info: dict[str, object] = metadata(args.model, not args.no_cache, args.batch_size, args.batch_wait_ms)
    info["profile"] = args.profile
    info["repeats"] = args.repeats
    print("Checking cache equivalence, request isolation, and selected backend...", flush=True)
    correctness: dict[str, object] = await check_correctness(args.model, args.batch_size, args.batch_wait_ms)
    runs: list[CaseRun] = []
    write_report(args.output, info, cases, correctness, runs)
    if not correctness["passed"]:
        raise RuntimeError(f"Correctness checks failed. Inspect {args.output / 'results.json'}")

    warmup: RequestSpec = make_request(tokenizer, 128, 4, index=99999)
    # Fresh backend state makes the first prefix request cold in every repeat.
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            async with InferenceClient(
                args.model, use_cache=not args.no_cache,
                max_batch_size=args.batch_size, batch_wait_ms=args.batch_wait_ms,
            ) as client:
                await client.generate(warmup.prompt, warmup.max_tokens, ignore_eos=True)
                run: CaseRun = await run_case(client, case, repeat)
            runs.append(run)
            summary: dict[str, object] = summarize(run.observations)
            print(
                f"[{repeat}/{args.repeats}] {case.name}: "
                f"{format_number(summary['output_tokens_per_second'])} tok/s, "
                f"peak {run.memory_after.peak_bytes / 1e9:.3f} GB, "
                f"failures {summary['failed']}", flush=True,
            )
            # Preserve completed results even if a later case is interrupted.
            write_report(args.output, info, cases, correctness, runs)
    print(f"Saved {args.output / 'summary.md'}", flush=True)
    if any(item.error is not None for run in runs for item in run.observations):
        raise RuntimeError("Some benchmark requests failed; inspect results.json.")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", action="append", help="Select a case; can be repeated.")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1, help="Max active rows with continuous refill. 1 keeps serial execution.")
    parser.add_argument("--batch-wait-ms", type=float, default=2.0, help="Max collection wait when idle; active decode never waits to fill slots.")
    parser.add_argument("--no-cache", action="store_true", help="Full-prefix recomputation ablation.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks") / f"suite-{uuid4().hex[:12]}")
    args: argparse.Namespace = parser.parse_args()
    if args.repeats < 1 or args.max_prompt_tokens < 128 or args.max_concurrency < 1:
        parser.error("repeats/concurrency must be >= 1; max-prompt-tokens must be >= 128")
    if args.batch_size < 1 or not isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        parser.error("batch-size must be >= 1; batch-wait-ms must be finite and >= 0")
    if args.batch_size > 1 and args.no_cache:
        parser.error("Batching requires KV caching; --no-cache only supports batch-size=1")
    if (args.output / "results.json").exists():
        parser.error("Output already contains results.json; choose a new directory.")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
