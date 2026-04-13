"""Fixed-shape stopping behavior and real Qwen attention/cache equivalence."""

import unittest
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache
from mlx_lm.models.qwen3 import Model, ModelArgs

from batch_infer import generate_batch_tokens
from infer import generate_tokens
from request import Request
from sampler import GreedySampler, Sampler
from telemetry import TokenEvent


class ScriptedModel:
    def __init__(self) -> None:
        self.layers: list[None] = [None]
        self.inputs: list[list[list[int]]] = []
        self.caches: list[BatchKVCache] = []

    def __call__(self, inputs: mx.array, cache: list[BatchKVCache]) -> mx.array:
        self.inputs.append(cast(list[list[int]], inputs.tolist()))
        self.caches = cache
        tokens: list[int] = [[9, 3, 7], [1, 9, 6], [2, 8, 5]][len(self.inputs) - 1]
        return mx.array([[[float(index == token) for index in range(10)]] for token in tokens])


class CountingSampler(Sampler):
    def __init__(self) -> None:
        self.calls: int = 0

    def sample(self, logits: mx.array) -> mx.array:
        self.calls += 1
        return mx.argmax(logits, axis=-1)


class FixedBatchTests(unittest.TestCase):
    def test_padding_eos_budgets_callbacks_and_fixed_rows(self) -> None:
        events: list[list[TokenEvent]] = [[], [], []]
        model: ScriptedModel = ScriptedModel()
        requests: list[Request] = [
            Request(0, "a", [1, 2], 3, GreedySampler(), on_token=events[0].append),
            Request(1, "b", [3], 3, GreedySampler(), on_token=events[1].append, ignore_eos=True),
            Request(2, "c", [4, 5, 6], 1, GreedySampler(), on_token=events[2].append),
        ]
        outputs = generate_batch_tokens(cast(Model, model), requests, {9})
        self.assertEqual(outputs, [[], [3, 9, 8], [7]])
        self.assertEqual([[event.token_id for event in row] for row in events], outputs)
        self.assertEqual(model.inputs, [
            [[0, 1, 2], [0, 0, 3], [4, 5, 6]],
            [[0], [3], [0]], [[0], [9], [0]],
        ])
        self.assertEqual(model.caches[0].left_padding.tolist(), [1, 2, 0])

    def test_per_request_samplers_stop_when_the_row_finishes(self) -> None:
        samplers: list[CountingSampler] = [CountingSampler() for _ in range(3)]
        requests: list[Request] = [
            Request(index, str(index), [index + 1], budget, sampler, ignore_eos=True)
            for index, (budget, sampler) in enumerate(zip([0, 3, 1], samplers))
        ]
        outputs = generate_batch_tokens(cast(Model, ScriptedModel()), requests, {9})
        self.assertEqual(outputs, [[], [3, 9, 8], [7]])
        self.assertEqual([sampler.calls for sampler in samplers], [0, 3, 1])

    def test_empty_or_all_zero_batches_do_not_call_model(self) -> None:
        model: ScriptedModel = ScriptedModel()
        self.assertEqual(generate_batch_tokens(cast(Model, model), [], {9}), [])
        self.assertEqual(generate_batch_tokens(cast(Model, model), [
            Request(0, "zero", [1], 0, GreedySampler())
        ], {9}), [[]])
        self.assertEqual(model.inputs, [])
        with self.assertRaises(ValueError):
            generate_batch_tokens(cast(Model, model), [Request(0, "", [], 1, GreedySampler())], {9})

    def test_tiny_qwen_padded_prefill_and_decode_match_unpadded_logits(self) -> None:
        mx.random.seed(7)
        model: Model = Model(ModelArgs(
            model_type="qwen3", hidden_size=16, num_hidden_layers=2,
            intermediate_size=32, num_attention_heads=2, rms_norm_eps=1e-6,
            vocab_size=32, num_key_value_heads=1, max_position_embeddings=128,
            head_dim=8, tie_word_embeddings=True, rope_theta=10000.0,
        ))
        prompts: list[list[int]] = [[1, 2, 3, 4], [5], [6, 7]]
        padding: list[int] = [0, 3, 2]
        batch_cache: list[BatchKVCache] = [BatchKVCache(padding) for _ in model.layers]
        serial_caches: list[list[KVCache]] = [[KVCache() for _ in model.layers] for _ in prompts]
        padded: mx.array = mx.array([[0] * pad + prompt for pad, prompt in zip(padding, prompts)])
        batch_logits: mx.array = model(padded, cache=batch_cache)[:, -1, :]
        for row, (prompt, cache) in enumerate(zip(prompts, serial_caches)):
            serial_logits: mx.array = model(mx.array([prompt]), cache=cache)[:, -1, :]
            self.assertTrue(mx.allclose(batch_logits[row:row + 1], serial_logits, atol=1e-5, rtol=1e-5).item())
        self.assertEqual(batch_cache[0].offset.tolist(), [4, 1, 2])
        for step in range(2):
            tokens: mx.array = mx.array([[8 + step], [10 + step], [12 + step]])
            batch_logits = model(tokens, cache=batch_cache)[:, -1, :]
            for row, cache in enumerate(serial_caches):
                serial_logits = model(tokens[row:row + 1], cache=cache)[:, -1, :]
                self.assertTrue(mx.allclose(batch_logits[row:row + 1], serial_logits, atol=1e-5, rtol=1e-5).item())
        requests: list[Request] = [
            Request(row, str(row), prompt, row + 1, GreedySampler())
            for row, prompt in enumerate(prompts)
        ]
        expected: list[list[int]] = [generate_tokens(model, req.prompt_token_ids, set(), req.max_tokens)
                                    for req in requests]
        self.assertEqual(generate_batch_tokens(model, requests, set()), expected)


if __name__ == "__main__":
    unittest.main()
