"""Submit four requests together; inspect one fixed batch's tensor shapes."""

import asyncio

from client import InferenceClient
from request import RequestOutput


async def main() -> None:
    async with InferenceClient(max_batch_size=4, trace=True) as client:
        requests: list[tuple[str, int]] = [
            ("What is the capital of France?", 24),
            ("What is 2 + 2?", 16),
            ("Count from 1 to 20, separated by commas.", 32),
            ("Say hello.", 8),
        ]
        outputs: list[RequestOutput] = await asyncio.gather(*(
            client.generate(prompt, max_tokens=budget) for prompt, budget in requests
        ))
        for output in outputs:
            print(f"\nRequest {output.request_id}: {output.prompt}")
            print(output.text)
            if output.timing is not None:
                print(f"batch={output.timing.batch_id}, batch_size={output.timing.batch_size}")


if __name__ == "__main__":
    asyncio.run(main())
