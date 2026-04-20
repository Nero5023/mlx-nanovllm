"""Page boundaries, non-contiguous mappings, reuse and contiguous KV equivalence."""

import unittest
from typing import Any

import mlx.core as mx

from mlx_nano_llm.paged_kv import KVPagePool, PagedKVCache


class PagedKVTests(unittest.TestCase):
    def pool(self, pages=4, page_size=2, dtype=mx.float32) -> KVPagePool:
        return KVPagePool(num_pages=pages, page_size=page_size, num_kv_heads=2,
                          head_dim=3, dtype=dtype)

    def kv(self, tokens: int, start=0, dtype=mx.float32) -> tuple[mx.array, mx.array]:
        keys = mx.arange(start, start + tokens * 6).reshape(tokens, 2, 3).astype(dtype)
        return keys, keys + mx.array(100, dtype=dtype)

    def assert_kv(self, cache: PagedKVCache, keys: mx.array, values: mx.array) -> None:
        actual_keys, actual_values = cache.gather()
        self.assertEqual(actual_keys.shape, keys.shape)
        self.assertEqual(actual_keys.dtype, keys.dtype)
        self.assertTrue(mx.array_equal(actual_keys, keys).item())
        self.assertTrue(mx.array_equal(actual_values, values).item())

    def test_interleaved_sequences_use_non_contiguous_pages_without_aliasing(self) -> None:
        pool = self.pool()
        a, b = PagedKVCache(pool), PagedKVCache(pool)
        first_k, first_v = self.kv(3)
        b_k, b_v = self.kv(2, start=40)
        next_k, next_v = self.kv(2, start=80)
        a.append(first_k, first_v)
        b.append(b_k, b_v)
        a.append(next_k, next_v)
        self.assertEqual(a.block_table, (0, 1, 3))
        self.assertEqual(b.block_table, (2,))
        self.assertEqual(a.slot_mapping().tolist(), [0, 1, 2, 3, 6])
        self.assertEqual(pool.used_page_count, 4)
        self.assertEqual(pool.free_page_count, 0)
        self.assert_kv(a, mx.concatenate([first_k, next_k]), mx.concatenate([first_v, next_v]))
        self.assert_kv(b, b_k, b_v)

    def test_tail_capacity_works_when_full_and_exhaustion_is_atomic(self) -> None:
        pool = self.pool(pages=2)
        cache = PagedKVCache(pool)
        keys, values = self.kv(3)
        cache.append(keys, values)
        tail_k, tail_v = self.kv(1, start=50)
        cache.append(tail_k, tail_v)  # No free page, but the final page has one free slot.
        table = cache.block_table
        with self.assertRaises(MemoryError):
            cache.append(*self.kv(3))
        self.assertEqual(cache.length, 4)
        self.assertEqual(cache.block_table, table)
        self.assertEqual(pool.used_page_count, 2)
        self.assert_kv(cache, mx.concatenate([keys, tail_k]), mx.concatenate([values, tail_v]))

    def test_free_reuse_does_not_expose_stale_tail_or_corrupt_other_sequence(self) -> None:
        pool = self.pool(pages=3)
        a, b, c = (PagedKVCache(pool) for _ in range(3))
        a.append(*self.kv(3))
        b_k, b_v = self.kv(2, start=40)
        b.append(b_k, b_v)
        snapshot = a.gather()  # Keep an unevaluated read alive across release/reuse.
        a.clear()
        a.clear()
        self.assertEqual(pool.free_page_count, 2)
        self.assertEqual(a.block_table, ())
        self.assertEqual(a.length, 0)
        c_k, c_v = self.kv(1, start=80)
        c.append(c_k, c_v)
        self.assertEqual(c.block_table, (0,))
        self.assert_kv(c, c_k, c_v)  # Unwritten slots from a are not part of c.
        self.assert_kv(b, b_k, b_v)
        old_k, old_v = self.kv(3)
        self.assertTrue(mx.array_equal(snapshot[0], old_k).item())
        self.assertTrue(mx.array_equal(snapshot[1], old_v).item())
        c.clear()
        a.append(*self.kv(1, start=120))
        self.assertEqual(a.length, 1)
        a.clear()
        b.clear()
        self.assertEqual(pool.free_page_count, 3)

    def test_exhaustion_does_not_consume_the_remaining_free_page(self) -> None:
        pool = self.pool(pages=3)
        cache = PagedKVCache(pool)
        keys, values = self.kv(3)
        cache.append(keys, values)
        table = cache.block_table
        with self.assertRaises(MemoryError):
            cache.append(*self.kv(4))  # Needs two new pages; only one is free.
        self.assertEqual(pool.free_page_count, 1)
        self.assertEqual(cache.block_table, table)
        self.assert_kv(cache, keys, values)
        extra_k, extra_v = self.kv(2, start=50)
        cache.append(extra_k, extra_v)  # The unconsumed free page remains usable.
        self.assertEqual(cache.length, 5)
        self.assertEqual(pool.free_page_count, 0)
        self.assert_kv(cache, mx.concatenate([keys, extra_k]), mx.concatenate([values, extra_v]))

    def test_page_boundaries_and_dtypes_match_contiguous_storage(self) -> None:
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for length in (15, 16, 17, 31, 32, 33):
                with self.subTest(dtype=dtype, length=length):
                    pool = self.pool(pages=4, page_size=16, dtype=dtype)
                    cache = PagedKVCache(pool)
                    keys, values = self.kv(length, dtype=dtype)
                    # Append as several chunks; allocation must depend on total length.
                    for start in range(0, length, 5):
                        cache.append(keys[start:start+5], values[start:start+5])
                    self.assertEqual(len(cache.block_table), (length + 15) // 16)
                    self.assertEqual(cache.length, length)
                    self.assert_kv(cache, keys, values)

    def test_invalid_append_does_not_allocate_or_modify_existing_kv(self) -> None:
        pool = self.pool()
        cache = PagedKVCache(pool)
        keys, values = self.kv(1)
        cache.append(keys, values)
        invalid = [
            (mx.zeros((2, 3)), mx.zeros((2, 3))),
            (mx.zeros((1, 2, 3)), mx.zeros((2, 2, 3))),
            (mx.zeros((1, 1, 3)), mx.zeros((1, 1, 3))),
            (keys.astype(mx.float16), values.astype(mx.float16)),
        ]
        for bad_k, bad_v in invalid:
            with self.assertRaises(ValueError):
                cache.append(bad_k, bad_v)
            self.assertEqual(cache.length, 1)
            self.assertEqual(pool.used_page_count, 1)
            self.assert_kv(cache, keys, values)

    def test_empty_append_and_gather_allocate_no_pages(self) -> None:
        pool = self.pool()
        cache = PagedKVCache(pool)
        keys, values = self.kv(0)
        cache.append(keys, values)
        self.assertEqual(cache.slot_mapping().shape, (0,))
        self.assertEqual(pool.used_page_count, 0)
        self.assert_kv(cache, keys, values)

    def test_invalid_pool_configuration(self) -> None:
        for name in ('num_pages', 'page_size', 'num_kv_heads', 'head_dim'):
            for bad in (0, -1, True, 1.5):
                # Invalid runtime inputs deliberately bypass the constructor's type hints.
                config: dict[str, Any] = dict(num_pages=4, page_size=2, num_kv_heads=2, head_dim=3)
                config[name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    KVPagePool(**config)
        with self.assertRaises(ValueError):
            self.pool(dtype=mx.int32)


if __name__ == '__main__':
    unittest.main()
