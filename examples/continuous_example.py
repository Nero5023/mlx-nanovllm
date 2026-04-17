"""Stagger three arrivals into two slots; print each result when it finishes."""

import asyncio
from time import perf_counter

from mlx_nano_llm.client import InferenceClient
from mlx_nano_llm.request import RequestOutput


async def main() -> None:
    async with InferenceClient(max_batch_size=2, trace=True) as client:
        origin: float = perf_counter()

        async def send(delay: float, prompt: str, budget: int) -> RequestOutput:
            await asyncio.sleep(delay)
            print(f"{perf_counter() - origin:.3f}s submit: {prompt}", flush=True)
            # Fixed budgets make the long/short overlap easy to see in the trace.
            output: RequestOutput = await client.generate(prompt, budget, ignore_eos=True)
            print(f"{perf_counter() - origin:.3f}s done: request={output.request_id}, "
                  f"tokens={len(output.token_ids)}", flush=True)
            return output

        await asyncio.gather(
            send(0.0, "Count from 1 to 100, separated by commas.", 64),
            send(0.05, "What is 2 + 2?", 8),
            send(0.10, "What is the capital of France?", 16),
        )


if __name__ == "__main__":
    asyncio.run(main())
