"""Clients submit while a single worker generates an earlier response."""

import asyncio
from time import perf_counter

from mlx_nano_llm.client import InferenceClient
from mlx_nano_llm.request import RequestOutput


async def main() -> None:
    async with InferenceClient() as client:
        start: float = perf_counter()

        async def send(name: str, prompt: str, delay: float, max_tokens: int) -> None:
            await asyncio.sleep(delay)
            submitted: float = perf_counter()
            print(f"{submitted - start:.3f}s: client {name} submits", flush=True)
            output: RequestOutput = await client.generate(prompt, max_tokens)
            finished: float = perf_counter()
            print(
                f"{finished - start:.3f}s: client {name} receives request "
                f"{output.request_id} (wait + execution: {finished - submitted:.3f}s)",
                flush=True,
            )
            print(f"{name}: {output.text}\n", flush=True)

        await asyncio.gather(
            send("A", "Count from 1 to 100, separated by commas.", 0.0, 128),
            send("B", "What is the capital of France?", 0.05, 32),
            send("C", "What is 2 + 2? Give a short answer.", 0.1, 16),
        )


if __name__ == "__main__":
    asyncio.run(main())
