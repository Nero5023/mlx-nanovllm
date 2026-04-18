"""Inference-only Qwen3: project just the final position to the vocabulary."""

from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx_lm.models.qwen3 import Model as Qwen3Model, ModelArgs
from mlx_lm.utils import load_model as load_mlx_model


class LastTokenQwen3(Qwen3Model):
    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None) -> mx.array:
        # All prompt positions still pass through the transformer and update KV.
        hidden = self.model(inputs, cache, input_embeddings)[:, -1:, :]
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)


def _model_classes(config: dict) -> tuple[type[LastTokenQwen3], type[ModelArgs]]:
    if config['model_type'] != 'qwen3':
        raise ValueError('Last-token projection currently supports Qwen3 models only.')
    return LastTokenQwen3, ModelArgs


def load_model(model_path: Path) -> tuple[LastTokenQwen3, dict]:
    model, config = load_mlx_model(model_path, get_model_classes=_model_classes)
    return cast(LastTokenQwen3, model), config
