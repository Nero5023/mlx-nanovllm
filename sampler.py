"""Strategies for selecting the next token from model logits."""

from abc import ABC, abstractmethod

import mlx.core as mx


class Sampler(ABC):
    """Interface for selecting one token per batch row."""

    @abstractmethod
    def sample(self, logits: mx.array) -> mx.array:
        """Map [batch, vocabulary] logits to [batch] integer token IDs."""
        raise NotImplementedError


class GreedySampler(Sampler):
    """Select the token with the highest logit."""

    def sample(self, logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)
