"""Guards for the two claims the training speed-up depends on."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens.config import BASE_MODEL  # noqa: E402
from newslens.modeling import (  # noqa: E402
    build_model,
    freeze_lower_layers,
    frozen_forward,
    trainable_forward,
)
from transformers import AutoTokenizer  # noqa: E402

TEXTS = ["The central bank raised rates again.", "A short one.", "Storm floods coastal towns " * 8]


def _batch():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return tokenizer(TEXTS, padding=True, return_tensors="pt")


def test_cached_path_matches_normal_forward():
    model = build_model().eval()
    batch = _batch()
    ids, mask = batch["input_ids"], batch["attention_mask"]
    with torch.inference_mode():
        normal = model(input_ids=ids, attention_mask=mask).logits
        hidden = frozen_forward(model, ids, mask, 2)
        cached = trainable_forward(model, hidden, mask, 2)
    assert torch.allclose(normal, cached, atol=1e-4)


def test_half_precision_cache_barely_moves_predictions():
    model = build_model().eval()
    batch = _batch()
    ids, mask = batch["input_ids"], batch["attention_mask"]
    with torch.inference_mode():
        hidden = frozen_forward(model, ids, mask, 2)
        rounded = torch.from_numpy(hidden.numpy().astype(np.float16).astype(np.float32))
        original = trainable_forward(model, hidden, mask, 2)
        stored = trainable_forward(model, rounded, mask, 2)
    assert torch.allclose(original, stored, atol=1e-2)


def test_only_top_layers_train():
    model = build_model()
    freeze_lower_layers(model, 2)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(
        name.startswith(("bert.encoder.layer.10", "bert.encoder.layer.11", "bert.pooler", "classifier"))
        for name in names
    )
    assert any(name.startswith("classifier") for name in names)
