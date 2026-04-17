"""Queue behavior checks without loading model weights."""

import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from mlx_lm.models.qwen3 import Model

from mlx_nano_llm.engine import InferenceEngine
from mlx_nano_llm.sampler import GreedySampler
from tests.test_batch_infer import ScriptedModel


class InferenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader: MagicMock = self.enterContext(
            patch(
                "mlx_nano_llm.engine.load_model",
                return_value=(object(), {"eos_token_id": 9, "max_position_embeddings": 16}),
            )
        )
        self.tokenizer_loader: MagicMock = self.enterContext(
            patch("mlx_nano_llm.engine.AutoTokenizer.from_pretrained")
        )
        self.tokenizer_loader.return_value.eos_token_id = 9
        self.tokenizer_loader.return_value.decode.return_value = "decoded response"
        self.encoder: MagicMock = self.enterContext(
            patch("mlx_nano_llm.engine.encode_prompt", return_value=[1, 2])
        )
        self.generator: MagicMock = self.enterContext(
            patch("mlx_nano_llm.engine.generate_tokens", return_value=[3])
        )
        self.engine: InferenceEngine = InferenceEngine(collect_stats=True)

    def test_fifo_budgets_samplers_and_model_reuse(self) -> None:
        sampler: GreedySampler = GreedySampler()
        first_id: int = self.engine.add_request("first", max_tokens=2, sampler=sampler)
        second_id: int = self.engine.add_request("second", max_tokens=3)
        self.generator.assert_not_called()  # Enqueueing does not generate tokens.
        self.assertEqual(self.engine.pending_requests, 2)

        first = self.engine.run_next()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.request_id, first_id)
        self.assertEqual(first.prompt, "first")
        self.assertEqual(first.text, "decoded response")
        self.assertEqual(self.engine.pending_requests, 1)
        self.assertEqual(self.generator.call_args.args[3], 2)
        self.assertIs(self.generator.call_args.kwargs["sampler"], sampler)

        rest = self.engine.run()
        self.assertEqual([output.request_id for output in rest], [second_id])
        self.assertEqual(self.generator.call_args.args[3], 3)
        self.assertIsNot(first.stats, rest[0].stats)
        self.assertIsNot(
            self.generator.call_args_list[0].kwargs["sampler"],
            self.generator.call_args_list[1].kwargs["sampler"],
        )
        self.assertEqual(self.engine.pending_requests, 0)

        third_id: int = self.engine.add_request("third", max_tokens=1)
        self.assertGreater(third_id, second_id)
        self.assertEqual(self.engine.run()[0].request_id, third_id)
        self.loader.assert_called_once()
        self.tokenizer_loader.assert_called_once()

    def test_empty_queue_and_zero_token_request(self) -> None:
        self.assertIsNone(self.engine.run_next())
        self.assertEqual(self.engine.run(), [])
        self.generator.assert_not_called()
        self.generator.return_value = []
        request_id: int = self.engine.add_request("zero", max_tokens=0)
        outputs = self.engine.run()
        self.assertEqual(outputs[0].request_id, request_id)
        self.assertEqual(outputs[0].token_ids, [])
        self.assertEqual(self.generator.call_args.args[3], 0)

    def test_ignore_eos_is_opt_in(self) -> None:
        self.engine.add_request("normal", max_tokens=1)
        output = self.engine.run()[0]
        self.assertEqual(self.generator.call_args.args[2], {9})
        self.assertEqual(output.prompt_tokens, 2)
        callback: MagicMock = MagicMock()
        self.engine.add_request("fixed budget", max_tokens=1, ignore_eos=True, on_token=callback)
        self.engine.run()
        self.assertEqual(self.generator.call_args.args[2], set())
        self.assertIs(self.generator.call_args.kwargs["on_token"], callback)

    def test_invalid_requests_leave_queue_unchanged(self) -> None:
        first_id: int = self.engine.add_request("valid", max_tokens=1)
        with self.assertRaises(ValueError):
            self.engine.add_request("negative", max_tokens=-1)
        with self.assertRaises(ValueError):
            self.engine.add_request("too long", max_tokens=15)
        self.encoder.return_value = []
        with self.assertRaises(ValueError):
            self.engine.add_request("no tokens", max_tokens=1)
        self.assertEqual(self.engine.pending_requests, 1)
        self.assertEqual(self.engine.run()[0].request_id, first_id)

    def test_failed_request_does_not_discard_waiting_requests(self) -> None:
        self.engine.add_request("fails", max_tokens=1)
        second_id: int = self.engine.add_request("still waiting", max_tokens=1)
        self.generator.side_effect = [RuntimeError("inference failed"), [3]]
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.engine.run_next()
        self.assertEqual(self.engine.pending_requests, 1)
        self.assertEqual(self.engine.run()[0].request_id, second_id)

    def test_fixed_batch_consumes_only_selected_requests_in_order(self) -> None:
        self.engine.collect_stats = False
        self.tokenizer_loader.return_value.pad_token_id = 0
        for prompt, budget in [("first", 1), ("second", 2), ("third", 3)]:
            self.engine.add_request(prompt, budget)
        with patch("mlx_nano_llm.engine.generate_batch_tokens", return_value=[[4], [5, 6]]) as generate:
            outputs = self.engine.run_batch(2)
            self.assertEqual([out.prompt for out in outputs], ["first", "second"])
            self.assertEqual([out.token_ids for out in outputs], [[4], [5, 6]])
            self.assertEqual([request.max_tokens for request in generate.call_args.args[1]], [1, 2])
        self.assertEqual(self.engine.pending_requests, 1)
        self.assertEqual(self.engine.run()[0].prompt, "third")
        self.assertEqual(self.engine.run_batch(2), [])

    def test_fixed_batch_failure_preserves_later_requests(self) -> None:
        self.engine.collect_stats = False
        for prompt in ["a", "b", "c"]:
            self.engine.add_request(prompt, 1)
        with patch("mlx_nano_llm.engine.generate_batch_tokens", side_effect=RuntimeError("batch failed")):
            with self.assertRaisesRegex(RuntimeError, "batch failed"):
                self.engine.run_batch(2)
        self.assertEqual(self.engine.pending_requests, 1)
        self.assertEqual(self.engine.run()[0].prompt, "c")

    def test_unsupported_batch_options_do_not_consume_queue(self) -> None:
        self.engine.add_request("waiting", 1)
        with self.assertRaises(ValueError):
            self.engine.run_batch(0)
        with self.assertRaisesRegex(ValueError, "collect_stats=False"):
            self.engine.run_batch(2)
        self.engine.collect_stats = False
        self.engine.use_cache = False
        with self.assertRaisesRegex(ValueError, "use_cache=True"):
            self.engine.run_batch(2)
        self.assertEqual(self.engine.pending_requests, 1)

    def test_stepwise_prefill_refill_zero_budget_and_individual_outputs(self) -> None:
        self.engine.collect_stats = False
        self.tokenizer_loader.return_value.pad_token_id = 0
        self.encoder.side_effect = [[1, 2], [3], [4], [6, 7, 8, 9, 1]]
        model: ScriptedModel = ScriptedModel([[4, 5], [6, 7], [8], [1, 2], [3]])
        self.engine.model = cast(Model, model)
        long_id: int = self.engine.add_request("long", 4)
        short_id: int = self.engine.add_request("short", 2)
        self.assertEqual(self.engine.prefill_next(2), [])
        self.assertEqual(self.engine.active_requests, 2)
        zero_id: int = self.engine.add_request("zero", 0)
        zero = self.engine.prefill_next(1)
        self.assertEqual([(out.request_id, out.token_ids) for out in zero], [(zero_id, [])])
        self.assertEqual(len(model.inputs), 1)  # Zero budget does not run or disturb active KV.
        short = self.engine.step()
        self.assertEqual([(out.request_id, out.token_ids) for out in short], [(short_id, [5, 7])])
        self.assertEqual(self.engine.active_requests, 1)
        incoming_id: int = self.engine.add_request("incoming", 2)
        self.assertEqual(self.engine.prefill_next(1), [])
        self.assertEqual(self.engine.active_requests, 2)
        incoming = self.engine.step()
        self.assertEqual([(out.request_id, out.token_ids) for out in incoming], [(incoming_id, [8, 2])])
        long = self.engine.step()
        self.assertEqual([(out.request_id, out.token_ids) for out in long], [(long_id, [4, 6, 1, 3])])
        self.assertEqual(self.engine.active_requests, 0)
        self.assertEqual(self.engine.step(), [])
        self.assertEqual(self.engine.pending_requests, 0)


if __name__ == "__main__":
    unittest.main()
