"""Build the corpus, cache frozen layers, train, evaluate and save the model.

Usage: python scripts/train.py [--limit 400] [--epochs 1]
"""
import argparse
import copy
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens import data  # noqa: E402
from newslens.config import (  # noqa: E402
    BASE_MODEL,
    CACHE_DIR,
    CONFIG,
    MODEL_DIR,
    RESULTS_DIR,
    TrainConfig,
)
from newslens.modeling import (  # noqa: E402
    HIDDEN_SIZE,
    build_model,
    freeze_lower_layers,
    frozen_forward,
    trainable_forward,
)
from newslens.taxonomy import TOPIC_KEYS  # noqa: E402


class FeatureStore:
    """Frozen-layer output for every row, kept on disk because it outgrows RAM."""

    def __init__(self, name, frame, tokenizer, model, config):
        encoded = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.ids, self.mask = encoded["input_ids"], encoded["attention_mask"]
        self.labels = torch.tensor(frame["label"].values, dtype=torch.long)
        self.lengths = self.mask.sum(1).numpy()
        CACHE_DIR.mkdir(exist_ok=True)
        shape = (len(frame), config.max_length, HIDDEN_SIZE)
        self.hidden = np.memmap(
            CACHE_DIR / f"{name}.bin", dtype=np.float16, mode="w+", shape=shape
        )
        order = np.argsort(self.lengths)  # Similar lengths keep padding small.
        for start in range(0, len(order), 32):
            rows = order[start : start + 32]
            width = int(self.lengths[rows].max())
            ids = self.ids[rows, :width]
            mask = self.mask[rows, :width]
            output = frozen_forward(model, ids, mask, config.trainable_layers)
            self.hidden[rows, :width] = output.numpy().astype(np.float16)
            done = min(start + 32, len(order))
            if done % 3200 == 0 or done == len(order):
                print(f"  cached {name}: {done}/{len(order)}", flush=True)
        self.hidden.flush()

    def batch(self, rows):
        rows = np.sort(rows)
        width = int(self.lengths[rows].max())
        hidden = torch.from_numpy(np.array(self.hidden[rows, :width], dtype=np.float32))
        return hidden, self.mask[rows, :width], self.labels[rows]


def training_batches(store, batch_size, rng):
    """Shuffled batches of similar length, so each batch has little padding."""
    order = rng.permutation(len(store.labels))
    batches = []
    pool_size = batch_size * 50
    for start in range(0, len(order), pool_size):
        pool = order[start : start + pool_size]
        pool = pool[np.argsort(store.lengths[pool])]
        batches += [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.inference_mode()
def logits_for(model, store, config):
    model.eval()
    order = np.argsort(store.lengths)
    chunks = np.zeros((len(order), len(TOPIC_KEYS)), dtype=np.float32)
    for start in range(0, len(order), 64):
        rows = np.sort(order[start : start + 64])
        hidden, mask, _ = store.batch(rows)
        chunks[rows] = trainable_forward(
            model, hidden, mask, config.trainable_layers
        ).numpy()
    return chunks


def set_train_mode(model, config):
    model.eval()
    for layer in model.bert.encoder.layer[-config.trainable_layers :]:
        layer.train()
    model.bert.pooler.train()
    model.dropout.train()
    model.classifier.train()


def fit_temperature(logits, labels) -> float:
    """Temperature that makes stated confidence match real accuracy on val."""
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    grid = torch.arange(0.3, 4.01, 0.02)
    losses = [torch.nn.functional.cross_entropy(logits / t, labels) for t in grid]
    return float(grid[int(torch.stack(losses).argmin())])


def score(logits, labels):
    predictions = logits.argmax(1)
    return {
        "n": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        # Only topics this subset really has: BBC covers 5 of the 13, and scoring
        # the other 8 would count each stray prediction as a whole failed topic.
        "macro_f1": float(f1_score(labels, predictions, labels=np.unique(labels), average="macro")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Rows, for a quick smoke run")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    args = parser.parse_args()
    config: TrainConfig = replace(CONFIG, epochs=args.epochs)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    corpus, audit = data.build_corpus(config)
    if args.limit:
        cells = corpus.groupby(["topic", "source"], group_keys=False)
        per_cell = max(20, args.limit // cells.ngroups)
        corpus = cells.head(per_cell).reset_index(drop=True)
    splits = data.make_splits(corpus, config)
    data.assert_no_leakage(splits)
    train_frame = data.add_headline_views(splits["train"], config)
    test_titles = splits["test"][
        (splits["test"]["source"] == "mnds")
        & (splits["test"]["title"].fillna("").str.split().str.len() >= config.min_words)
    ].assign(text=lambda f: f["title"])
    print(json.dumps(audit), flush=True)
    print(f"train {len(train_frame)} val {len(splits['val'])} test {len(splits['test'])}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = build_model().eval()
    freeze_lower_layers(model, config.trainable_layers)
    stores = {
        "train": FeatureStore("train", train_frame, tokenizer, model, config),
        "val": FeatureStore("val", splits["val"], tokenizer, model, config),
        "test": FeatureStore("test", splits["test"], tokenizer, model, config),
        "titles": FeatureStore("titles", test_titles, tokenizer, model, config),
    }
    print(f"cache ready {time.perf_counter() - started:.0f}s", flush=True)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * (len(train_frame) // config.batch_size + 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    val_labels = stores["val"].labels.numpy()
    best_score, best_state, history = -1.0, None, []

    for epoch in range(1, config.epochs + 1):
        set_train_mode(model, config)
        running_loss, seen = 0.0, 0
        batches = training_batches(stores["train"], config.batch_size, rng)
        for step, rows in enumerate(batches, 1):
            hidden, mask, labels = stores["train"].batch(rows)
            optimizer.zero_grad()
            logits = trainable_forward(model, hidden, mask, config.trainable_layers)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f"  epoch {epoch} step {step}/{len(batches)} loss {running_loss / seen:.3f} {elapsed:.0f}s", flush=True)
        val_f1 = score(logits_for(model, stores["val"], config), val_labels)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "val_macro_f1": val_f1})
        print(f"epoch {epoch} val macro-F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_score:
            best_score = val_f1
            best_state = {
                name: copy.deepcopy(p.detach())
                for name, p in model.named_parameters()
                if p.requires_grad
            }

    model.load_state_dict(best_state, strict=False)

    val_logits = logits_for(model, stores["val"], config)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits = logits_for(model, stores["test"], config)
    title_logits = logits_for(model, stores["titles"], config)
    test_labels = stores["test"].labels.numpy()
    report = classification_report(
        test_labels, test_logits.argmax(1),
        labels=list(range(len(TOPIC_KEYS))), target_names=TOPIC_KEYS,
        output_dict=True, zero_division=0,
    )
    by_source = {}
    sources = splits["test"]["source"].values
    for source in np.unique(sources):
        keep = sources == source
        by_source[source] = score(test_logits[keep], test_labels[keep])

    probabilities = torch.softmax(torch.tensor(test_logits) / temperature, dim=-1).numpy()
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == test_labels
    thresholds = {}
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
        kept = confidence >= threshold
        thresholds[str(threshold)] = {
            "coverage": float(kept.mean()),
            "accuracy_when_answering": float(correct[kept].mean()) if kept.any() else None,
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    (MODEL_DIR / "calibration.json").write_text(
        json.dumps({"temperature": temperature, "max_length": config.max_length})
    )
    metrics = {
        "config": config.to_dict(),
        "audit": audit,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "train_rows_with_headline_views": len(train_frame),
        "history": history,
        "temperature": temperature,
        "test": score(test_logits, test_labels),
        "test_by_source": by_source,
        "test_headlines_only": score(title_logits, stores["titles"].labels.numpy()),
        "per_topic": {key: report[key] for key in TOPIC_KEYS},
        "confidence_thresholds": thresholds,
        "confusion_matrix": confusion_matrix(
            test_labels, test_logits.argmax(1), labels=list(range(len(TOPIC_KEYS)))
        ).tolist(),
        "train_seconds_cpu": time.perf_counter() - started,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("test", "test_by_source", "test_headlines_only", "temperature", "confidence_thresholds")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
