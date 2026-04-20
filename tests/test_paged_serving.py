"""Engine-owned pools survive serial requests; request pages never survive completion."""

import asyncio
import threading
import unittest
from typing import Any
from unittest.mock import patch

import mlx.core as mx

from mlx_nano_llm.client import InferenceClient
from mlx_nano_llm.engine import InferenceEngine
from mlx_nano_llm.infer import generate_tokens
from tests.test_chunked_prefill import tiny_model


class EnginePatches(unittest.TestCase):
    def patch_engine(self, loader):
        self.enterContext(patch('mlx_nano_llm.engine.load_model', side_effect=loader))
        tokenizer = self.enterContext(patch('mlx_nano_llm.engine.AutoTokenizer.from_pretrained'))
        tokenizer.return_value.eos_token_id = 31
        tokenizer.return_value.pad_token_id = 0
        tokenizer.return_value.decode.side_effect = lambda ids, **kw: str(ids)
        self.enterContext(patch('mlx_nano_llm.engine.encode_prompt',
                                side_effect=lambda _, text: [int(x) for x in text.split(',')]))

    def caches(self, engine: InferenceEngine):
        assert engine._paged_cache is not None
        return engine._paged_cache

    def pool_ids(self, engine: InferenceEngine):
        ids = []
        for cache in self.caches(engine):
            assert cache.storage is not None
            ids.append(id(cache.storage.pool))
        return tuple(ids)

    def assert_free(self, engine: InferenceEngine):
        for cache in self.caches(engine):
            self.assertEqual(cache.offset, 0)
            if cache.storage is not None:
                self.assertEqual(cache.storage.block_table, ())
                self.assertEqual(cache.storage.pool.free_page_count, cache.num_pages)


CONFIG = {'eos_token_id': 31, 'max_position_embeddings': 128}


class PagedEngineTests(EnginePatches):
    def setUp(self):
        self.model = tiny_model()
        self.patch_engine(lambda _: (self.model, CONFIG))

    def test_a_b_a_reuses_the_same_layer_pools_and_resets_positions(self):
        engine = InferenceEngine(paged_kv=True, kv_page_size=2, kv_num_pages=3)
        tables = []

        def record(_):
            storage = self.caches(engine)[0].storage
            assert storage is not None
            tables.append(storage.block_table)

        outputs = []
        pool_ids = []
        for prompt in ('1,2,3', '6,7,8', '1,2,3'):
            engine.add_request(prompt, 2, ignore_eos=True, on_token=record)
            output = engine.run_next()
            assert output is not None
            outputs.append(output.token_ids)
            pool_ids.append(self.pool_ids(engine))
            self.assert_free(engine)
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(outputs[0], generate_tokens(self.model, [1, 2, 3], set(), 2))
        self.assertEqual(outputs[1], generate_tokens(self.model, [6, 7, 8], set(), 2))
        self.assertEqual(pool_ids[0], pool_ids[1])
        self.assertEqual(pool_ids[1], pool_ids[2])
        self.assertEqual(tables[0], (0, 1))
        self.assertEqual(tables[2], (2, 0))  # Freed physical pages can be reused non-contiguously.

    def test_capacity_validation_zero_budget_and_unsupported_modes_preserve_queue(self):
        engine = InferenceEngine(paged_kv=True, kv_page_size=2, kv_num_pages=2)
        with self.assertRaises(MemoryError):
            engine.add_request('1,2,3', 3)  # Needs five history tokens; pool holds four.
        self.assertEqual(engine.pending_requests, 0)
        self.assertTrue(all(c.storage is None for c in self.caches(engine)))
        zero_id = engine.add_request('1,2,3,4,5', 0)  # No KV needed despite the long prompt.
        zero = engine.run_next()
        assert zero is not None
        self.assertEqual(zero.request_id, zero_id)
        self.assertEqual(zero.token_ids, [])
        self.assertTrue(all(c.storage is None for c in self.caches(engine)))
        engine.add_request('1', 2, ignore_eos=True)
        with self.assertRaises(ValueError):
            engine.prefill_next(1)
        self.assertEqual(engine.pending_requests, 1)
        self.assertEqual(len(engine.run_batch(1)), 1)
        self.assert_free(engine)

    def test_eos_and_callback_failure_return_pages_before_the_next_request(self):
        engine = InferenceEngine(paged_kv=True, kv_page_size=2, kv_num_pages=8)
        first = generate_tokens(self.model, [1, 2, 3], set(), 1)[0]
        engine.eos_token_ids = {first}
        engine.add_request('1,2,3', 5)
        output = engine.run_next()
        assert output is not None
        self.assertEqual(output.token_ids, [])
        self.assert_free(engine)

        def fail(_):
            raise RuntimeError('callback failed')

        engine.add_request('1,2,3', 5, ignore_eos=True, on_token=fail)
        with self.assertRaisesRegex(RuntimeError, 'callback failed'):
            engine.run_next()
        self.assert_free(engine)
        engine.add_request('1,2,3', 5, ignore_eos=True)
        output = engine.run_next()
        assert output is not None
        self.assertEqual(output.token_ids, generate_tokens(self.model, [1, 2, 3], set(), 5))
        self.assert_free(engine)


    def test_partial_model_failure_reuses_clean_pools_for_the_following_request(self):
        base = self.model

        class FailOnce:
            def __init__(self):
                self.layers = base.layers
                self.fail_next = False

            def __call__(self, inputs, cache):
                if self.fail_next:
                    self.fail_next = False
                    pool = cache[0].storage.pool
                    keys = mx.zeros((1, pool.num_kv_heads, inputs.shape[1], pool.head_dim), dtype=pool.dtype)
                    cache[0].update_and_fetch(keys, keys)
                    raise RuntimeError('model failed after one layer')
                return base(inputs, cache=cache)

        model = FailOnce()
        with patch('mlx_nano_llm.engine.load_model', return_value=(model, CONFIG)):
            engine = InferenceEngine(paged_kv=True, kv_page_size=2, kv_num_pages=8)
        engine.add_request('1,2,3', 4, ignore_eos=True)
        first = engine.run_next()
        pool_ids = self.pool_ids(engine)
        model.fail_next = True
        engine.add_request('6,7,8', 4, ignore_eos=True)
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            engine.run_next()
        self.assert_free(engine)
        engine.add_request('1,2,3', 4, ignore_eos=True)
        again = engine.run_next()
        assert first is not None and again is not None
        self.assertEqual(again.token_ids, first.token_ids)
        self.assertEqual(self.pool_ids(engine), pool_ids)
        self.assert_free(engine)



class PagedWorkerTests(EnginePatches, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.models = []

        def load(_):
            model = tiny_model()  # Model construction and execution stay on the worker.
            self.models.append(model)
            return model, CONFIG

        self.patch_engine(load)
        self.engines = []

        def create(*args, **kwargs):
            engine = InferenceEngine(*args, **kwargs)
            self.engines.append(engine)
            return engine

        self.enterContext(patch('mlx_nano_llm.worker.InferenceEngine', side_effect=create))

    async def test_fifo_and_capacity_failure_recover_on_one_shared_engine(self):
        async with InferenceClient(paged_kv=True, kv_page_size=2, kv_num_pages=4) as client:
            bad = client.submit('1,2,3', 10, ignore_eos=True)
            a = client.submit('1,2,3', 4, ignore_eos=True)
            b = client.submit('6,7', 3, ignore_eos=True)
            with self.assertRaises(MemoryError):
                await bad
            outputs = await asyncio.gather(a, b)
            pool_ids = self.pool_ids(self.engines[0])
            again = await client.generate('1,2,3', 4, ignore_eos=True)
            self.assertEqual(again.token_ids, outputs[0].token_ids)
            self.assertEqual(pool_ids, self.pool_ids(self.engines[0]))
            self.assertEqual([o.request_id for o in outputs], [0, 1])
            self.assert_free(self.engines[0])
        self.assertEqual(len(self.engines), 1)
        self.assertFalse(client._worker._thread.is_alive())

    async def test_cancellation_and_shutdown_drain_active_work_and_return_pages(self):
        started, release = threading.Event(), threading.Event()

        class GatedModel:
            def __init__(self):
                self.base = tiny_model()
                self.layers = self.base.layers
                self.first = True

            def __call__(self, inputs, cache):
                output = self.base(inputs, cache=cache)
                if self.first:
                    self.first = False
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError('test did not release model')
                return output

        with patch('mlx_nano_llm.engine.load_model', side_effect=lambda _: (GatedModel(), CONFIG)):
            client = InferenceClient(paged_kv=True, kv_page_size=2, kv_num_pages=8)
            await client.start()
            self.addAsyncCleanup(client.close)
            try:
                active = client.submit('1,2,3', 5, ignore_eos=True)
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                active.cancel()
                queued = client.submit('4,5', 2, ignore_eos=True)
                queued.cancel()
                following = client.submit('6,7', 2, ignore_eos=True)
                closing = asyncio.create_task(client.close())
                await asyncio.sleep(0)
            finally:
                release.set()
            await asyncio.wait_for(closing, 3)
            self.assertTrue(active.cancelled())
            self.assertTrue(queued.cancelled())
            self.assertEqual(len((await following).token_ids), 2)
            self.assert_free(self.engines[0])
            self.assertFalse(client._worker._thread.is_alive())

    async def test_invalid_serving_options_fail_before_model_loading(self):
        cases: list[dict[str, Any]] = [
            {'prefill_chunk_size': 2}, {'use_cache': False},
            {'kv_num_pages': 0}, {'kv_page_size': 0},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                InferenceClient(paged_kv=True, **options)
        self.assertEqual(self.models, [])


if __name__ == '__main__':
    unittest.main()
