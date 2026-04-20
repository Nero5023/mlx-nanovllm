"""Single-request gather + existing attention matches contiguous KV inference."""

import unittest
from typing import cast
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Model

from mlx_nano_llm.infer import generate_tokens
from mlx_nano_llm.benchmark import BenchmarkStats
from mlx_nano_llm.paged_kv import PagedKVAdapter
from tests.test_chunked_prefill import tiny_model


class PagedInferenceTests(unittest.TestCase):
    def test_adapter_matches_contiguous_layout_offsets_and_masks_across_pages(self) -> None:
        paged = PagedKVAdapter(num_pages=4, page_size=2)
        dense = KVCache()
        for length in (3, 2, 1):
            expected_mask = dense.make_mask(length, return_array=True, window_size=None)
            actual_mask = paged.make_mask(length, return_array=True)
            if isinstance(expected_mask, mx.array):
                assert isinstance(actual_mask, mx.array)
                self.assertTrue(mx.array_equal(expected_mask, actual_mask).item())
            else:
                self.assertEqual(actual_mask, expected_mask)
            keys = mx.arange(length * 2 * 4).reshape(1, 2, length, 4).astype(mx.float32)
            values = keys + 100
            expected = dense.update_and_fetch(keys, values)
            actual = paged.update_and_fetch(keys, values)
            self.assertEqual(paged.offset, dense.offset)
            for a, b in zip(actual, expected, strict=True):
                assert isinstance(b, mx.array)
                self.assertEqual(a.shape, b.shape)
                self.assertTrue(mx.array_equal(a, b).item())
        assert paged.storage is not None
        self.assertEqual(len(paged.storage.block_table), 3)
        paged.clear()
        self.assertEqual(paged.offset, 0)
        self.assertEqual(paged.storage.pool.free_page_count, 4)

    def test_real_tiny_qwen_logits_and_every_layer_kv_match_with_past_context(self) -> None:
        model = tiny_model()
        dense = [KVCache() for _ in model.layers]
        paged = [PagedKVAdapter(num_pages=8, page_size=2) for _ in model.layers]
        # A multi-token call after existing context also checks causal-mask alignment.
        for tokens in ([1, 2, 3], [4, 5], [6], [7]):
            inputs = mx.array([tokens])
            expected = model(inputs, cache=dense)
            actual = model(inputs, cache=paged)
            self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
            for a, b in zip(paged, dense, strict=True):
                assert a.storage is not None
                self.assertEqual(a.offset, b.offset)
                keys, values = a.storage.gather()
                for x, y in zip((keys.transpose(1, 0, 2)[None], values.transpose(1, 0, 2)[None]),
                                b.state[:2], strict=True):
                    assert isinstance(y, mx.array)
                    self.assertTrue(mx.allclose(x, y, atol=1e-5, rtol=1e-5).item())

    def test_complete_generation_matches_dense_and_full_recompute(self) -> None:
        model = tiny_model()
        for prompt in ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]):
            expected = generate_tokens(model, prompt, set(), 12)
            self.assertEqual(expected, generate_tokens(model, prompt, set(), 12, use_cache=False))
            for page_size in (1, 2, 4, 16):
                with self.subTest(prompt=prompt, page_size=page_size):
                    self.assertEqual(generate_tokens(model, prompt, set(), 12,
                                                     paged_kv=True, kv_page_size=page_size), expected)

    def test_request_pages_are_released_after_completion_eos_and_callback_error(self) -> None:
        model = tiny_model()
        first_token = generate_tokens(model, [1, 2, 3], set(), 1)[0]
        created: list[PagedKVAdapter] = []

        def create(**kwargs):
            cache = PagedKVAdapter(**kwargs)
            created.append(cache)
            return cache

        def fail(_):
            raise RuntimeError('callback failed')

        with patch('mlx_nano_llm.infer.PagedKVAdapter', side_effect=create):
            generate_tokens(model, [1, 2, 3], set(), 5, paged_kv=True, kv_page_size=2)
            self.assertEqual(generate_tokens(model, [1, 2, 3], {first_token}, 5,
                                             paged_kv=True, kv_page_size=2), [])
            with self.assertRaisesRegex(RuntimeError, 'callback failed'):
                generate_tokens(model, [1, 2, 3], set(), 5, paged_kv=True,
                                kv_page_size=2, on_token=fail)
        self.assertEqual(len(created), len(model.layers) * 3)
        for cache in created:
            assert cache.storage is not None
            self.assertEqual(cache.offset, 0)
            self.assertEqual(cache.storage.pool.free_page_count, cache.num_pages)

    def test_zero_budget_allocates_no_paged_cache_and_options_are_validated(self) -> None:
        model = tiny_model()
        with patch('mlx_nano_llm.infer.PagedKVAdapter') as factory:
            self.assertEqual(generate_tokens(model, [1], set(), 0, paged_kv=True), [])
            factory.assert_not_called()
        with self.assertRaises(ValueError):
            generate_tokens(model, [1], set(), 1, paged_kv=True, use_cache=False)
        with self.assertRaises(ValueError):
            generate_tokens(model, [1], set(), 1, paged_kv=True, kv_page_size=0)

    def test_statistics_cover_the_paged_generation_request(self) -> None:
        stats = BenchmarkStats()
        output = generate_tokens(tiny_model(), [1, 2, 3], set(), 5,
                                 paged_kv=True, kv_page_size=2, stats=stats)
        self.assertEqual(len(output), 5)
        self.assertEqual(stats.prompt_tokens, 3)
        self.assertEqual(stats.generated_tokens, 5)
        self.assertEqual(stats.decode_steps, 4)
        self.assertGreater(stats.prefill_seconds, 0)
        self.assertGreater(stats.peak_memory_bytes, 0)

    def test_model_error_releases_pages_when_only_one_layer_updated(self) -> None:
        caches = [PagedKVAdapter(num_pages=4, page_size=2) for _ in range(2)]

        class FailingModel:
            layers = [None, None]

            def __call__(self, inputs, cache):
                keys = mx.zeros((1, 1, inputs.shape[1], 4))
                cache[0].update_and_fetch(keys, keys)
                raise RuntimeError('model failed after one layer')

        with patch('mlx_nano_llm.infer.PagedKVAdapter', side_effect=caches):
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                generate_tokens(cast(Model, FailingModel()), [1, 2, 3], set(), 5,
                                paged_kv=True, kv_page_size=2)
        assert caches[0].storage is not None
        self.assertEqual(caches[0].storage.pool.used_page_count, 0)
        self.assertEqual(caches[0].offset, 0)
        self.assertIsNone(caches[1].storage)

    def test_adapter_rejects_batches_and_capacity_overflow_preserves_history(self) -> None:
        cache = PagedKVAdapter(num_pages=1, page_size=2)
        with self.assertRaises(ValueError):
            cache.update_and_fetch(mx.zeros((2, 1, 1, 4)), mx.zeros((2, 1, 1, 4)))
        self.assertIsNone(cache.storage)
        keys = mx.ones((1, 1, 2, 4))
        cache.update_and_fetch(keys, keys + 1)
        with self.assertRaises(MemoryError):
            cache.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
        self.assertEqual(cache.offset, 2)
        assert cache.storage is not None
        self.assertTrue(mx.array_equal(cache.storage.gather()[0], mx.ones((2, 1, 4))).item())

    def test_shared_cache_capacity_is_checked_for_all_layers_before_any_write(self) -> None:
        caches = (PagedKVAdapter(num_pages=4, page_size=2),
                  PagedKVAdapter(num_pages=1, page_size=2))
        with self.assertRaises(MemoryError):
            generate_tokens(tiny_model(), [1, 2, 3], set(), 3, paged_kv=True,
                            kv_page_size=2, paged_cache=caches)
        self.assertTrue(all(cache.storage is None for cache in caches))

    def test_shared_cache_with_existing_history_is_rejected_without_clearing_it(self) -> None:
        caches = [PagedKVAdapter(num_pages=4, page_size=2) for _ in range(2)]
        keys = mx.ones((1, 1, 1, 4))
        caches[0].update_and_fetch(keys, keys)
        with self.assertRaisesRegex(ValueError, 'empty paged caches'):
            generate_tokens(tiny_model(), [1, 2, 3], set(), 3, paged_kv=True,
                            kv_page_size=2, paged_cache=caches)
        self.assertEqual(caches[0].offset, 1)
        assert caches[0].storage is not None
        self.assertEqual(caches[0].storage.pool.used_page_count, 1)
        caches[0].clear()


if __name__ == '__main__':
    unittest.main()
