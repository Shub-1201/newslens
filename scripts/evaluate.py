"""Re-score the saved model, end to end, the way the app will run it.

Training ends by halving the weights to float16, so this confirms the saved
files reproduce the training run. It also adds what the training run doesn't:
whether confidence can be trusted, and what the news check costs on real news.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens import data  # noqa: E402
from newslens.config import CONFIG, RESULTS_DIR  # noqa: E402
from newslens.predictor import Predictor  # noqa: E402
from newslens.taxonomy import TOPIC_TO_ID  # noqa: E402

BINS = 10


def summarise(predictions, topics):
    guessed = np.array([TOPIC_TO_ID[p.topic] for p in predictions])
    labels = np.array([TOPIC_TO_ID[t] for t in topics])
    confidence = np.array([p.confidence for p in predictions])
    correct = guessed == labels
    news = np.array([p.looks_like_news for p in predictions])
    sure = news & (confidence >= 0.5)
    edges = np.minimum((confidence * BINS).astype(int), BINS - 1)
    gap = sum(
        (edges == b).mean() * abs(correct[edges == b].mean() - confidence[edges == b].mean())
        for b in range(BINS)
        if (edges == b).any()
    )
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(labels, guessed, labels=np.unique(labels), average="macro")),
        "calibration_error": float(gap),
        "mean_confidence": float(confidence.mean()),
        "given_a_desk": float(sure.mean()),
        "accuracy_when_given_a_desk": float(correct[sure].mean()),
    }


def main():
    corpus, _ = data.build_corpus(CONFIG)
    test = data.make_splits(corpus, CONFIG)["test"]
    titled = test[test["title"].fillna("").str.split().str.len() >= 1]
    predictor = Predictor()

    predictions = predictor.classify_many(test["text"].tolist())
    results = {"stories": summarise(predictions, test["topic"].tolist())}
    for source in sorted(test["source"].unique()):
        keep = (test["source"] == source).to_numpy()
        results[f"source:{source}"] = summarise(
            [p for p, k in zip(predictions, keep) if k], test["topic"][keep].tolist()
        )
    results["headlines"] = summarise(
        predictor.classify_many(titled["title"].tolist()), titled["topic"].tolist()
    )

    path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(path.read_text())
    drift = abs(results["stories"]["accuracy"] - metrics["test"]["accuracy"])
    assert drift < 0.002, f"saved model drifted from the training run by {drift:.4f}"
    metrics["saved_model"] = results
    path.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
