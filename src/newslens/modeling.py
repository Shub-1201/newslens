"""BERT with its top layers split out, so frozen layers run only once.

Frozen layers always stay in eval mode, so their output for a given input never
changes. Caching that output and training only the top layers is therefore the
same model as ordinary partial fine-tuning, at a fraction of the CPU time.
tests/test_model.py checks the two paths give the same predictions.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .config import BASE_MODEL
from .taxonomy import TOPIC_KEYS

HIDDEN_SIZE = 768


def build_model() -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(TOPIC_KEYS),
        id2label=dict(enumerate(TOPIC_KEYS)),
        label2id={key: i for i, key in enumerate(TOPIC_KEYS)},
        attn_implementation="eager",
    )


def freeze_lower_layers(model: nn.Module, trainable_layers: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [
        *model.bert.encoder.layer[-trainable_layers:],
        model.bert.pooler,
        model.classifier,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _attention_mask(mask: torch.Tensor) -> torch.Tensor:
    blocked = 1.0 - mask[:, None, None, :].float()
    return blocked * torch.finfo(torch.float32).min


@torch.inference_mode()
def frozen_forward(model, ids, mask, trainable_layers: int) -> torch.Tensor:
    """Output of the last frozen layer, which is what gets cached."""
    attention = _attention_mask(mask)
    hidden = model.bert.embeddings(input_ids=ids)
    for layer in model.bert.encoder.layer[:-trainable_layers]:
        hidden = layer(hidden, attention_mask=attention)
    return hidden


def trainable_forward(model, hidden, mask, trainable_layers: int) -> torch.Tensor:
    """From cached hidden states to topic logits."""
    attention = _attention_mask(mask)
    for layer in model.bert.encoder.layer[-trainable_layers:]:
        hidden = layer(hidden, attention_mask=attention)
    return model.classifier(model.dropout(model.bert.pooler(hidden)))
