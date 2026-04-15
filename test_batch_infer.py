"""Compaction, request mapping, and real Qwen attention/cache equivalence."""

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
    def __init__(self, steps: list[list[int]] | None = None) -> None:
        self.layers: list[None] = [None, None]
        self.steps: list[list[int]] = steps if steps is not None else []
        self.inputs: list[list[list[int]]] = []
        self.caches: list[BatchKVCache] = []
        self.cache_history: list[list[dict[str, object]]] = []

    def __call__(self, inputs: mx.array, cache: list[BatchKVCache]) -> mx.array:
        self.inputs.append(cast(list[list[int]], inputs.tolist()))
        self.caches = cache
        states: list[dict[str, object]] = []
        for layer in cache:
            states.append({
                "offsets": layer.offset.tolist(),
                "padding": layer.left_padding.tolist(),
                "keys": layer.state[:2][0][:, 0, :, 0].tolist() if layer.keys is not None else None,
                "values": layer.state[:2][1][:, 0, :, 0].tolist() if layer.values is not None else None,
            })
            # Store recognizable token histories in real caches, so an incorrect
            # row filter or padding shift is visible on the next model call.
            keys: mx.array = inputs[:, None, :, None].astype(mx.float32)
            layer.update_and_fetch(keys, keys + 100)
        self.cache_history.append(states)
        tokens: list[int] = self.steps[len(self.inputs) - 1]
        assert len(tokens) == inputs.shape[0], "Unexpected number of active rows."
        return mx.array([[[float(index == token) for index in range(10)]] for token in tokens])


class CountingSampler(Sampler):
    def __init__(self) -> None:
        self.calls: int = 0

    def sample(self, logits: mx.array) -> mx.array:
        self.calls += 1
        return mx.argmax(logits, axis=-1)


class BatchCompactionTests(unittest.TestCase):
    def test_padding_eos_budgets_callbacks_and_finished_rows_removed(self) -> None:
        events: list[list[TokenEvent]] = [[], [], []]
        model: ScriptedModel = ScriptedModel([[9, 3, 7], [9], [8]])
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
            [[3]], [[9]],
        ])
        self.assertEqual(model.caches[0].left_padding.tolist(), [0])

    def test_repeated_compaction_preserves_original_outputs_and_every_cache_layer(self) -> None:
        events: list[list[TokenEvent]] = [[], [], []]
        model: ScriptedModel = ScriptedModel([[4, 9, 6], [7, 8], [5]])
        requests: list[Request] = [
            Request(101, "a", [1, 2], 2, GreedySampler(), on_token=events[0].append),
            Request(303, "b", [3, 4, 5, 6], 3, GreedySampler(), on_token=events[1].append),
            Request(505, "c", [7], 3, GreedySampler(), on_token=events[2].append),
        ]
        outputs = generate_batch_tokens(cast(Model, model), requests, {9})
        self.assertEqual(outputs, [[4, 7], [], [6, 8, 5]])
        self.assertEqual([[event.token_id for event in row] for row in events], outputs)
        self.assertEqual(model.inputs, [
            [[0, 0, 1, 2], [3, 4, 5, 6], [0, 0, 0, 7]], [[4], [6]], [[8]],
        ])
        for layer_state in model.cache_history[1]:
            self.assertEqual(layer_state, {
                "keys": [[1, 2], [0, 7]], "values": [[101, 102], [100, 107]],
                "offsets": [2, 1], "padding": [0, 1],
            })
        for layer_state in model.cache_history[2]:
            self.assertEqual(layer_state, {
                "keys": [[7, 6]], "values": [[107, 106]],
                "offsets": [2], "padding": [0],
            })
        self.assertEqual(len(model.cache_history[1]), 2)

    def test_per_request_samplers_stop_when_the_row_finishes(self) -> None:
        samplers: list[CountingSampler] = [CountingSampler() for _ in range(3)]
        requests: list[Request] = [
            Request(index, str(index), [index + 1] * (10 if index == 0 else 1),
                    budget, sampler, ignore_eos=True)
            for index, (budget, sampler) in enumerate(zip([0, 3, 1], samplers))
        ]
        model: ScriptedModel = ScriptedModel([[3, 7], [9], [8]])
        outputs = generate_batch_tokens(cast(Model, model), requests, {9})
        self.assertEqual(outputs, [[], [3, 9, 8], [7]])
        self.assertEqual([sampler.calls for sampler in samplers], [0, 3, 1])
        self.assertEqual(model.inputs[0], [[2], [3]])  # Zero-budget row excluded before padding.

    def test_empty_or_all_zero_batches_do_not_call_model(self) -> None:
        model: ScriptedModel = ScriptedModel()
        self.assertEqual(generate_batch_tokens(cast(Model, model), [], {9}), [])
        self.assertEqual(generate_batch_tokens(cast(Model, model), [
            Request(0, "zero", [1], 0, GreedySampler())
        ], {9}), [[]])
        self.assertEqual(model.inputs, [])
        with self.assertRaises(ValueError):
            generate_batch_tokens(cast(Model, model), [Request(0, "", [], 1, GreedySampler())], {9})

    def test_all_rows_finish_on_the_same_step(self) -> None:
        model: ScriptedModel = ScriptedModel([[9, 5]])
        requests: list[Request] = [
            Request(0, "eos", [1], 3, GreedySampler()),
            Request(1, "budget", [2], 1, GreedySampler()),
        ]
        self.assertEqual(generate_batch_tokens(cast(Model, model), requests, {9}), [[], [5]])
        self.assertEqual(len(model.inputs), 1)  # Never call filter([]) or run an empty batch.

    def test_tiny_qwen_prefill_decode_and_compaction_match_unpadded_logits(self) -> None:
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

        # Remove the longest prompt, then remove a survivor. The serial caches
        # keep the full unpadded histories as an independent numerical reference.
        for keep, next_tokens in [([1, 2], [[14], [15]]), ([1], [[16]])]:
            for layer in batch_cache:
                layer.filter(keep)
            serial_caches = [serial_caches[row] for row in keep]
            tokens = mx.array(next_tokens)
            batch_logits = model(tokens, cache=batch_cache)[:, -1, :]
            for row, cache in enumerate(serial_caches):
                serial_logits = model(tokens[row:row + 1], cache=cache)[:, -1, :]
                self.assertTrue(mx.allclose(batch_logits[row:row + 1], serial_logits, atol=1e-5, rtol=1e-5).item())
            keys: mx.array | None = batch_cache[0].keys
            values: mx.array | None = batch_cache[0].values
            assert keys is not None and values is not None
            self.assertEqual(keys.shape[0], len(keep))
            self.assertEqual(values.shape[0], len(keep))
        requests: list[Request] = [
            Request(row, str(row), prompt, row + 1, GreedySampler())
            for row, prompt in enumerate(prompts)
        ]
        expected: list[list[int]] = [generate_tokens(model, req.prompt_token_ids, set(), req.max_tokens)
                                    for req in requests]
        self.assertEqual(generate_batch_tokens(model, requests, set()), expected)


if __name__ == "__main__":
    unittest.main()
