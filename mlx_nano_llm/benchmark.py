"""Measurements for one generation request."""

from dataclasses import asdict, dataclass


@dataclass
class BenchmarkStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    decode_steps: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def reset(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = 0
        self.decode_steps = 0
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.peak_memory_bytes = 0

    @property
    def prefill_tokens_per_second(self) -> float | None:
        if self.prefill_seconds == 0:
            return None
        return self.prompt_tokens / self.prefill_seconds

    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.decode_seconds == 0:
            return None
        return self.decode_steps / self.decode_seconds

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            **asdict(self),
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "peak_memory_gb": self.peak_memory_bytes / 1e9,
        }
