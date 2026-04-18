"""Compare last-position projection against the full Qwen3 output head."""

import unittest

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.models.qwen3 import Model, ModelArgs

from mlx_nano_llm.batch_infer import generate_batch_tokens
from mlx_nano_llm.infer import generate_tokens
from mlx_nano_llm.model import LastTokenQwen3
from mlx_nano_llm.request import Request
from mlx_nano_llm.sampler import GreedySampler


class LastTokenProjectionTests(unittest.TestCase):
    def models(self, tied: bool) -> tuple[Model, LastTokenQwen3]:
        mx.random.seed(7)
        args = ModelArgs(
            model_type="qwen3", hidden_size=16, num_hidden_layers=2,
            intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6,
            vocab_size=32, num_key_value_heads=1, max_position_embeddings=128,
            head_dim=8, tie_word_embeddings=tied, rope_theta=10000.0,
        )
        reference = Model(args)
        optimized = LastTokenQwen3(args)
        optimized.update(reference.parameters())
        return reference, optimized

    def test_padded_prefill_and_decode_logits_and_cache_match_full_projection(self) -> None:
        for tied in (True, False):
            with self.subTest(tied=tied):
                reference, optimized = self.models(tied)
                caches = [[BatchKVCache([0, 2]) for _ in model.layers]
                          for model in (reference, optimized)]
                for inputs in (mx.array([[1, 2, 3], [0, 0, 4]]), mx.array([[5], [6]])):
                    expected = reference(inputs, cache=caches[0])[:, -1:, :]
                    actual = optimized(inputs, cache=caches[1])
                    self.assertEqual(actual.shape, (2, 1, 32))
                    self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
                    for a, b in zip(*caches, strict=True):
                        self.assertTrue(mx.array_equal(a.offset, b.offset).item())
                        self.assertTrue(mx.array_equal(a.left_padding, b.left_padding).item())
                        for x, y in zip(a.state[:2], b.state[:2], strict=True):
                            self.assertTrue(mx.allclose(x, y, atol=1e-5, rtol=1e-5).item())

    def test_full_single_request_generation_with_and_without_cache(self) -> None:
        for tied in (True, False):
            for use_cache in (True, False):
                with self.subTest(tied=tied, use_cache=use_cache):
                    reference, optimized = self.models(tied)
                    expected = generate_tokens(reference, [1, 2, 3], set(), 8, use_cache=use_cache)
                    actual = generate_tokens(optimized, [1, 2, 3], set(), 8, use_cache=use_cache)
                    self.assertEqual(actual, expected)

    def test_full_batch_generation_with_padding_and_compaction(self) -> None:
        for tied in (True, False):
            with self.subTest(tied=tied):
                reference, optimized = self.models(tied)
                requests = [Request(i, str(i), prompt, budget, GreedySampler())
                            for i, (prompt, budget) in enumerate([
                                ([1, 2, 3, 4], 8), ([5], 3), ([6, 7], 1),
                            ])]
                self.assertEqual(generate_batch_tokens(optimized, requests, set()),
                                 generate_batch_tokens(reference, requests, set()))


if __name__ == "__main__":
    unittest.main()
