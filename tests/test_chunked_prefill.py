"""Resumable prefill, padding-only chunks, KV merging and complete generation."""

import unittest
from typing import cast
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs

from mlx_nano_llm.batch_infer import ActiveRequest, BatchState, generate_batch_tokens
from mlx_nano_llm.engine import InferenceEngine
from mlx_nano_llm.infer import generate_tokens
from mlx_nano_llm.model import LastTokenQwen3
from mlx_nano_llm.request import Request
from mlx_nano_llm.sampler import GreedySampler


def tiny_model() -> LastTokenQwen3:
    mx.random.seed(7)
    return LastTokenQwen3(ModelArgs(
        model_type="qwen3", hidden_size=16, num_hidden_layers=2,
        intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6,
        vocab_size=32, num_key_value_heads=1, max_position_embeddings=128,
        head_dim=8, tie_word_embeddings=True, rope_theta=10000.0,
    ))


class RecordingModel:
    def __init__(self, base: LastTokenQwen3):
        self.base = base
        self.layers = base.layers
        self.calls: list[tuple[str, int]] = []

    def model(self, inputs, cache=None):
        self.calls.append(("body", inputs.shape[1]))
        return self.base.model(inputs, cache=cache)

    def __call__(self, inputs, cache=None):
        self.calls.append(("head", inputs.shape[1]))
        return self.base(inputs, cache=cache)


class ChunkedPrefillTests(unittest.TestCase):
    def test_padding_only_chunks_preserve_cache_and_complete_outputs(self) -> None:
        model = tiny_model()
        for chunk_size in (1, 2, 3, 7, 10):
            with self.subTest(chunk_size=chunk_size):
                events = []
                requests = [
                    Request(i, str(i), prompt, budget, GreedySampler(), on_token=events.append)
                    for i, (prompt, budget) in enumerate([
                        ([1, 2, 3, 4, 5, 6, 7], 8), ([8], 3), ([9, 10, 11], 2),
                    ])
                ]
                expected = generate_batch_tokens(model, requests, set())
                events.clear()
                whole = BatchState(model, [ActiveRequest(r) for r in requests], set())
                whole.step()
                events.clear()
                recorded = RecordingModel(model)
                states = [ActiveRequest(r) for r in requests]
                batch = BatchState(cast(Model, recorded), states, set())
                while batch.prefilling:
                    finished = batch.prefill_chunk(chunk_size)
                    self.assertEqual(finished, [])
                    if batch.prefilling:
                        self.assertEqual(events, [])
                self.assertEqual(len(events), 3)  # Only the final chunk samples/emits.
                self.assertTrue(all(width <= chunk_size for _, width in recorded.calls))
                self.assertEqual(sum(kind == "head" for kind, _ in recorded.calls), 1)
                for a, b in zip(batch.cache, whole.cache, strict=True):
                    self.assertEqual(a.offset.tolist(), [7, 1, 3])
                    self.assertEqual(a.left_padding.tolist(), b.left_padding.tolist())
                    for x, y in zip(a.state[:2], b.state[:2], strict=True):
                        for row, padding in enumerate(cast(list[int], a.left_padding.tolist())):
                            self.assertTrue(mx.allclose(x[row, :, padding:, :], y[row, :, padding:, :],
                                                        atol=1e-5, rtol=1e-5).item())
                while batch.size:
                    batch.step()
                self.assertEqual([s.token_ids for s in states], expected)

    def engine(self, model: LastTokenQwen3) -> InferenceEngine:
        self.enterContext(patch('mlx_nano_llm.engine.load_model',
                                return_value=(model, {'eos_token_id': 31, 'max_position_embeddings': 128})))
        loader = self.enterContext(patch('mlx_nano_llm.engine.AutoTokenizer.from_pretrained'))
        loader.return_value.eos_token_id = 31
        loader.return_value.pad_token_id = 0
        self.enterContext(patch('mlx_nano_llm.engine.encode_prompt',
                                side_effect=lambda _, text: [int(t) for t in text.split(',')]))
        return InferenceEngine()

    def test_engine_decodes_existing_rows_between_chunks_then_merges(self) -> None:
        model = tiny_model()
        engine = self.engine(model)
        a = engine.add_request('1,2', 10, ignore_eos=True)
        engine.prefill_next(1)
        b = engine.add_request('3,4,5,6,7,8,9', 3, ignore_eos=True)
        outputs = []
        for _ in range(3):
            self.assertEqual(engine.prefill_next(1, chunk_size=2), [])
            self.assertEqual(engine.prefilling_requests, 1)
            self.assertEqual(engine.active_requests, 1)
            outputs.extend(engine.step())
        engine.prefill_next(1, chunk_size=2)
        self.assertEqual(engine.prefilling_requests, 0)
        self.assertEqual(engine.active_requests, 2)
        while engine.active_requests:
            outputs.extend(engine.step())
        by_id = {o.request_id: o.token_ids for o in outputs}
        self.assertEqual(by_id[a], generate_tokens(model, [1, 2], set(), 10))
        self.assertEqual(by_id[b], generate_tokens(model, [3, 4, 5, 6, 7, 8, 9], set(), 3))

    def test_zero_budget_and_one_token_results_during_chunked_admission(self) -> None:
        engine = self.engine(tiny_model())
        zero = engine.add_request('1', 0)
        one = engine.add_request('2,3,4,5,6', 1, ignore_eos=True)
        outputs = engine.prefill_next(2, chunk_size=2)
        self.assertEqual([(o.request_id, o.token_ids) for o in outputs], [(zero, [])])
        self.assertEqual(engine.prefilling_requests, 1)
        self.assertEqual(engine.prefill_next(2, chunk_size=2), [])
        outputs = engine.prefill_next(2, chunk_size=2)
        self.assertEqual([(o.request_id, len(o.token_ids)) for o in outputs], [(one, 1)])
        self.assertEqual(engine.active_requests + engine.prefilling_requests, 0)

    def test_invalid_chunk_does_not_consume_requests_and_abort_discards_partial_kv(self) -> None:
        engine = self.engine(tiny_model())
        engine.add_request('1,2,3,4,5', 2, ignore_eos=True)
        with self.assertRaises(ValueError):
            engine.prefill_next(1, chunk_size=0)
        self.assertEqual(engine.pending_requests, 1)
        engine.prefill_next(1, chunk_size=2)
        self.assertEqual(engine.prefilling_requests, 1)
        engine.abort_batch()
        self.assertEqual(engine.active_requests + engine.prefilling_requests, 0)
        next_id = engine.add_request('6', 1, ignore_eos=True)
        outputs = engine.prefill_next(1, chunk_size=2)
        self.assertEqual([o.request_id for o in outputs], [next_id])


if __name__ == '__main__':
    unittest.main()
