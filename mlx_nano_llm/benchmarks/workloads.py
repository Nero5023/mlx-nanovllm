"""Deterministic workloads, independent of the engine's scheduling strategy."""

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from mlx_nano_llm.infer import encode_prompt


@dataclass(frozen=True)
class RequestSpec:
    name: str
    prompt: str
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_seconds: float = 0.0


@dataclass(frozen=True)
class Case:
    name: str
    purpose: str
    requests: list[RequestSpec]
    sequential: bool = False


def make_request(
    tokenizer: PreTrainedTokenizerBase,
    prompt_tokens: int,
    output_tokens: int,
    index: int = 0,
    *,
    arrival_seconds: float = 0.0,
    shared_prefix: bool = False,
) -> RequestSpec:
    """Size the *formatted* prompt, including the chat template, in tokens."""
    padding: int = max(0, prompt_tokens - 40)
    marker: int = 0 if shared_prefix else index
    for _ in range(12):
        prompt: str = (
            f"Document {marker}:\n" + " data" * padding
            + f"\nRequest {index}: Continue the numbered list: 1, 2, 3,"
        )
        token_ids: list[int] = encode_prompt(tokenizer, prompt)
        difference: int = prompt_tokens - len(token_ids)
        if difference == 0:
            return RequestSpec(
                f"request-{index}", prompt, token_ids, output_tokens, arrival_seconds
            )
        padding += difference
        if padding < 0:
            break
    raise ValueError(f"Cannot construct a prompt of exactly {prompt_tokens} tokens.")


def build_cases(
    tokenizer: PreTrainedTokenizerBase,
    *,
    profile: str = "quick",
    max_prompt_tokens: int = 2048,
    max_concurrency: int = 4,
) -> list[Case]:
    if max_prompt_tokens < 128 or max_concurrency < 1:
        raise ValueError("Use max_prompt_tokens >= 128 and max_concurrency >= 1.")

    def request(
        length: int, output: int, index: int = 0, *, delay: float = 0.0,
        shared: bool = False,
    ) -> RequestSpec:
        return make_request(
            tokenizer, min(length, max_prompt_tokens), output, index,
            arrival_seconds=delay, shared_prefix=shared,
        )

    cases: list[Case] = [Case(
        "single", "Single-request latency and throughput baseline.", [request(128, 64)]
    )]
    concurrency_levels: list[int] = sorted({
        value for value in (1, 2, 4, 8, max_concurrency) if value <= max_concurrency
    })
    for count in concurrency_levels:
        cases.append(Case(
            f"burst-{count}", "Equal-length simultaneous arrivals; future fixed batching.",
            [request(128, 64, index) for index in range(count)],
        ))

    count: int = min(4, max_concurrency)
    cases.extend([
        Case("mixed-burst", "Long request first; short requests reveal head-of-line blocking.", [
            request(1024 if index == 0 else 128, 128 if index == 0 else 16, index)
            for index in range(count)
        ]),
        Case("staggered", "Open-loop arrivals during decode; future continuous batching.", [
            request(128, 128 if index == 0 else 16, index, delay=index * 0.05)
            for index in range(count)
        ]),
        Case("prefix-shared", "Closed-loop: cold first request, then a reusable prefix.", [
            request(512, 32, index, shared=True) for index in range(4)
        ], sequential=True),
        Case("prefix-control", "Same lengths, an early different token prevents prefix reuse.", [
            request(512, 32, index) for index in range(4)
        ], sequential=True),
    ])
    lengths: set[int] = {128, min(512, max_prompt_tokens), max_prompt_tokens}
    if profile == "full":
        lengths.update(value for value in (256, 1024, 4096, 8192) if value <= max_prompt_tokens)
    for length in sorted(lengths):
        cases.append(Case(
            f"context-{length}", "Prefill/context memory sweep.", [request(length, 32)]
        ))
    cases.append(Case(
        "context-burst", "Long contexts together; future active KV memory and paging.",
        [request(max_prompt_tokens, 32, index) for index in range(max_concurrency)],
    ))
    if profile == "full":
        for length in (127, 128, 129, 255, 256, 257, 511, 512, 513):
            if length <= max_prompt_tokens:
                cases.append(Case(
                    f"boundary-{length}", "Adjacent lengths expose allocation/kernel boundaries.",
                    [request(length, 33)],
                ))
    return cases


def common_prefix_tokens(requests: list[RequestSpec]) -> int:
    if not requests:
        return 0
    count: int = 0
    for tokens in zip(*(request.prompt_token_ids for request in requests)):
        if len(set(tokens)) != 1:
            break
        count += 1
    return count
