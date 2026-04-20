"""Fixed paged batches reuse BatchState, release rows and split by page capacity."""

import asyncio
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from mlx_nano_llm.batch_infer import ActiveRequest, BatchState, generate_batch_tokens
from mlx_nano_llm.client import InferenceClient
from mlx_nano_llm.engine import InferenceEngine
from mlx_nano_llm.infer import generate_tokens
from mlx_nano_llm.paged_kv import PagedBatchKVCache, PagedKVAdapter
from mlx_nano_llm.request import Request
from mlx_nano_llm.sampler import GreedySampler
from tests.test_chunked_prefill import tiny_model
from tests.test_paged_serving import EnginePatches, CONFIG


class PagedBatchTests(unittest.TestCase):
    def assert_owners_free(self, owners):
        for owner in owners:
            assert owner.storage is not None
            self.assertEqual(owner.storage.pool.used_page_count, 0)

    def test_ragged_prefill_and_decode_logits_match_contiguous_cache(self):
        model = tiny_model()
        owners = [PagedKVAdapter(num_pages=16, page_size=2) for _ in model.layers]
        paged = [PagedBatchKVCache(owner, [0, 4, 2]) for owner in owners]
        dense = [BatchKVCache([0, 4, 2]) for _ in model.layers]
        for tokens in ([[1,2,3,4,5],[0,0,0,0,6],[0,0,7,8,9]], [[10],[11],[12]]):
            inputs = mx.array(tokens)
            expected, actual = model(inputs, cache=dense), model(inputs, cache=paged)
            self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
            self.assertEqual(paged[0].offset.tolist(), dense[0].offset.tolist())
        for cache in paged:
            cache.filter([1, 2])  # Drop the longest row, reducing gathered padding.
        for cache in dense:
            cache.filter([1, 2])
        self.assertTrue(mx.allclose(model(mx.array([[13],[14]]), cache=paged),
                                    model(mx.array([[13],[14]]), cache=dense), atol=1e-5, rtol=1e-5).item())
        for cache in paged:
            cache.clear()
        self.assert_owners_free(owners)

    def test_complete_outputs_and_error_cleanup_with_shared_pools(self):
        model = tiny_model()
        owners = [PagedKVAdapter(num_pages=16, page_size=2) for _ in model.layers]
        requests = [Request(i, str(i), p, n, GreedySampler(), ignore_eos=True)
                    for i, (p, n) in enumerate([([1,2,3,4,5],1),([6],8),([7,8,9],3),([10],0)])]
        expected = [generate_tokens(model, r.prompt_token_ids, set(), r.max_tokens) for r in requests]
        self.assertEqual(generate_batch_tokens(model, requests, set(), paged_cache=owners), expected)
        self.assert_owners_free(owners)

        def fail(_):
            raise RuntimeError('token callback failed')

        bad = [Request(0, 'bad', [1,2,3], 3, GreedySampler(), on_token=fail)]
        with self.assertRaisesRegex(RuntimeError, 'token callback failed'):
            generate_batch_tokens(model, bad, set(), paged_cache=owners)
        self.assert_owners_free(owners)
        self.assertEqual(generate_batch_tokens(model, requests, set(), paged_cache=owners), expected)

    def test_finished_rows_release_pages_without_moving_surviving_tables(self):
        model = tiny_model()
        owners = [PagedKVAdapter(num_pages=8, page_size=2) for _ in model.layers]
        states = [ActiveRequest(Request(i, str(i), p, n, GreedySampler(), ignore_eos=True))
                  for i, (p,n) in enumerate([([1,2,3],2),([4],5)])]
        batch = BatchState(model, states, set(), paged_cache=owners)
        batch.prefill()
        cache = batch.cache[0]
        assert isinstance(cache, PagedBatchKVCache)
        surviving_pages = cache.rows[1].block_table
        storage = cache.owner.storage
        assert storage is not None
        before = storage.pool.used_page_count
        self.assertEqual(batch.step(), [states[0]])
        self.assertEqual(cache.rows[0].block_table, surviving_pages)
        self.assertLess(storage.pool.used_page_count, before)
        batch.close()
        self.assertEqual(storage.pool.used_page_count, 0)

    def test_zero_budgets_and_eos_release_only_the_finished_rows(self):
        model = tiny_model()
        owners = [PagedKVAdapter(num_pages=16, page_size=2) for _ in model.layers]
        zero = [Request(0, 'zero', [1], 0, GreedySampler())]
        self.assertEqual(generate_batch_tokens(model, zero, set(), paged_cache=owners), [[]])
        self.assertTrue(all(owner.storage is None for owner in owners))
        eos = generate_tokens(model, [1,2,3], set(), 1)[0]
        requests = [Request(0, 'eos', [1,2,3], 8, GreedySampler()),
                    Request(1, 'keep', [4], 4, GreedySampler(), ignore_eos=True)]
        outputs = generate_batch_tokens(model, requests, {eos}, paged_cache=owners)
        self.assertEqual(outputs[0], [])
        self.assertEqual(outputs[1], generate_tokens(model, [4], set(), 4))
        self.assert_owners_free(owners)


class PagedBatchEngineTests(EnginePatches):
    def setUp(self):
        self.model = tiny_model()
        self.patch_engine(lambda _: (self.model, CONFIG))

    def test_capacity_selects_fifo_prefix_and_keeps_remaining_requests_waiting(self):
        engine = InferenceEngine(paged_kv=True, kv_page_size=2, kv_num_pages=4)
        ids = [engine.add_request(p, n, ignore_eos=True)
               for p,n in [('1,2,3',2),('4',4),('5,6,7,8,9',2)]]
        first = engine.run_batch(3)
        self.assertEqual([o.request_id for o in first], ids[:2])
        self.assertEqual(engine.pending_requests, 1)
        self.assert_free(engine)
        self.assertEqual([o.request_id for o in engine.run_batch(3)], ids[2:])
        self.assert_free(engine)


class PagedBatchWorkerTests(EnginePatches, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patch_engine(lambda _: (tiny_model(), CONFIG))
        self.engines = []

        def create(*args, **kwargs):
            engine = InferenceEngine(*args, **kwargs)
            self.engines.append(engine)
            return engine

        self.enterContext(patch('mlx_nano_llm.worker.InferenceEngine', side_effect=create))

    async def test_fixed_group_and_capacity_split_preserve_results_and_pool_reuse(self):
        async with InferenceClient(paged_kv=True, max_batch_size=3, batch_wait_ms=100,
                                   kv_page_size=2, kv_num_pages=4) as client:
            specs = [('1,2,3',2),('4',4),('5,6,7,8,9',2)]
            outputs = await asyncio.gather(*(client.generate(p,n,ignore_eos=True) for p,n in specs))
            self.assertEqual([o.request_id for o in outputs], [0,1,2])
            expected = [generate_tokens(tiny_model(), [int(x) for x in p.split(',')], set(), n) for p,n in specs]
            self.assertEqual([o.token_ids for o in outputs], expected)
            self.assert_free(self.engines[0])
        self.assertFalse(client._worker._thread.is_alive())

    async def test_failed_group_discards_its_waiting_tail_before_later_requests(self):
        original = generate_batch_tokens
        fail = True

        def generate(*args, **kwargs):
            nonlocal fail
            if fail:
                fail = False
                raise RuntimeError('batch failed')
            return original(*args, **kwargs)

        with patch('mlx_nano_llm.engine.generate_batch_tokens', side_effect=generate):
            async with InferenceClient(paged_kv=True, max_batch_size=3, batch_wait_ms=100,
                                       kv_page_size=2, kv_num_pages=2) as client:
                futures = [client.submit('1,2,3',2,ignore_eos=True) for _ in range(3)]
                for future in futures:
                    with self.assertRaisesRegex(RuntimeError, 'batch failed'):
                        await future
                self.assertEqual(self.engines[0].pending_requests, 0)
                recovered = await client.generate('4',2,ignore_eos=True)
                self.assertEqual(recovered.request_id, 3)
                self.assert_free(self.engines[0])


if __name__ == '__main__':
    unittest.main()
