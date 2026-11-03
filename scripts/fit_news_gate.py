"""Teach NewsLens to notice text that isn't news. Run after train.py.

Fits a logistic regression on BERT's pooled output, using the training stories
as news and seven kinds of real everyday text as not-news. Half of each kind is
kept out for testing. A second pass holds out one kind entirely, which shows how
the gate copes with text it has never seen. The threshold comes from validation
stories and headlines, so test data plays no part in choosing it.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens import data  # noqa: E402
from newslens.config import CACHE_DIR, CONFIG, MODEL_DIR, RESULTS_DIR  # noqa: E402
from newslens.predictor import GATE_FILE, Predictor  # noqa: E402
from newslens.taxonomy import TOPIC_KEYS  # noqa: E402

FEATURES_FILE = CACHE_DIR / "gate_features.npz"
SEED = 7
ALLOWED_NEWS_LOSS = 0.02
MIN_WORDS = 5
LETTERS = "qwertyuiopasdfghjklzxcvbnm"

# name on the Hub, text field, rows
HUB_KINDS = {
    "text messages": ("ucirvine/sms_spam", "sms", 900),
    "quotes": ("Abirate/english_quotes", "quote", 700),
    "film reviews": ("stanfordnlp/imdb", "text", 700),
    "tweets": ("dair-ai/emotion", "text", 900),
    "emails": ("SetFit/enron_spam", "text", 900),
}


def from_hub(name, field, rows):
    stream = load_dataset(name, split="train", streaming=True).shuffle(seed=SEED, buffer_size=5000)
    found = []
    for row in stream:
        text = str(row[field]).strip()
        if len(text.split()) >= MIN_WORDS:
            found.append(text[:1500])
        if len(found) == rows:
            break
    return found


def code_snippets(rows, rng):
    """Real source code: slices of the Python packages already installed."""
    root = Path(np.__file__).resolve().parents[1]
    found = []
    for path in sorted(root.glob("*/*.py"))[:1500]:
        lines = [line for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        if len(lines) > 40:
            start = rng.randrange(len(lines) - 30)
            found.append("\n".join(lines[start : start + rng.randint(4, 25)]))
    rng.shuffle(found)
    return found[:rows]


def gibberish(rows, rng):
    word = lambda: "".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 8)))
    return [" ".join(word() for _ in range(rng.randint(5, 40))) for _ in range(rows)]


@torch.inference_mode()
def features(predictor, texts, batch_size=16):
    """Pooled output for each text, batched by length to save padding."""
    order = np.argsort([len(text) for text in texts])
    output = np.zeros((len(texts), predictor.model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        encoded = predictor._encode([texts[i] for i in chosen])
        output[chosen] = predictor.pooled(encoded).numpy()
    return output


def headlines(frame, rows):
    usable = frame[frame["title"].fillna("").str.split().str.len() >= MIN_WORDS]
    return usable.sample(min(rows, len(usable)), random_state=SEED)


def collect(predictor):
    """Everything the gate needs, cached because it takes a quarter of an hour."""
    rng = random.Random(SEED)
    corpus, _ = data.build_corpus(CONFIG)
    splits = data.make_splits(corpus, CONFIG)
    groups = {
        "news:fit": splits["train"].sample(3000, random_state=SEED)["text"].tolist()
        + headlines(splits["train"], 1000)["title"].tolist(),
        "news:val": splits["val"]["text"].tolist(),
        "news:val_headlines": headlines(splits["val"], 800)["title"].tolist(),
        "news:test": splits["test"]["text"].tolist(),
        "news:test_headlines": headlines(splits["test"], 800)["title"].tolist(),
    }
    for kind, (name, field, rows) in HUB_KINDS.items():
        groups[f"other:{kind}"] = from_hub(name, field, rows)
    groups["other:code"] = code_snippets(700, rng)
    groups["other:gibberish"] = gibberish(400, rng)

    saved = {}
    for name, texts in groups.items():
        saved[name] = features(predictor, texts)
        print(f"{name}: {len(texts)}", flush=True)
    saved["test_topics"] = splits["test"]["topic"].to_numpy(dtype=str)
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(FEATURES_FILE, **saved)


def fit(news, others):
    scaler = StandardScaler().fit(news)
    rows = np.vstack([news, *others])
    labels = np.r_[np.ones(len(news)), np.zeros(sum(len(o) for o in others))]
    model = LogisticRegression(C=0.05, max_iter=500, class_weight="balanced")
    return scaler, model.fit(scaler.transform(rows), labels)


def scores(scaler, model, rows):
    return model.predict_proba(scaler.transform(rows))[:, 1]


def main():
    predictor = Predictor()
    if not FEATURES_FILE.exists():
        collect(predictor)
    saved = np.load(FEATURES_FILE)
    news = saved["news:fit"]
    kinds = [name.split(":")[1] for name in saved.files if name.startswith("other:")]
    fit_part = {kind: saved[f"other:{kind}"][::2] for kind in kinds}
    test_part = {kind: saved[f"other:{kind}"][1::2] for kind in kinds}
    validation = np.vstack([saved["news:val"], saved["news:val_headlines"]])

    def threshold_for(scaler, model):
        return float(np.quantile(scores(scaler, model, validation), ALLOWED_NEWS_LOSS))

    unseen = {}
    for held in kinds:
        scaler, model = fit(news, [rows for kind, rows in fit_part.items() if kind != held])
        threshold = threshold_for(scaler, model)
        unseen[held] = float((scores(scaler, model, test_part[held]) < threshold).mean())

    scaler, model = fit(news, list(fit_part.values()))
    threshold = threshold_for(scaler, model)
    test_scores = scores(scaler, model, saved["news:test"])
    passed = test_scores >= threshold
    topics = saved["test_topics"]
    results = {
        "threshold": threshold,
        "test_stories_passed": float(passed.mean()),
        "test_headlines_passed": float(
            (scores(scaler, model, saved["news:test_headlines"]) >= threshold).mean()
        ),
        "stories_passed_by_topic": {k: float(passed[topics == k].mean()) for k in TOPIC_KEYS},
        "caught_when_trained_on_kind": {
            kind: float((scores(scaler, model, rows) < threshold).mean())
            for kind, rows in test_part.items()
        },
        "caught_when_kind_held_out": unseen,
    }
    (MODEL_DIR / GATE_FILE).write_text(
        json.dumps(
            {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "weights": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
                "threshold": threshold,
            }
        )
    )
    path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["news_gate"] = results
    path.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
