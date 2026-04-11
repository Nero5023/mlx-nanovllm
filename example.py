"""Queue several prompts and serve them one at a time with one loaded model."""

from engine import InferenceEngine


def main() -> None:
    engine: InferenceEngine = InferenceEngine()
    engine.add_request("What is the capital of France?", max_tokens=32)
    engine.add_request("What is 2 + 2? Give a short answer.", max_tokens=16)
    engine.add_request("Say hello in Chinese.", max_tokens=16)

    # Each call completes one request before the next one begins.
    while (output := engine.run_next()) is not None:
        print(f"Request {output.request_id}: {output.prompt}", flush=True)
        print(f"Response: {output.text}\n", flush=True)


if __name__ == "__main__":
    main()
