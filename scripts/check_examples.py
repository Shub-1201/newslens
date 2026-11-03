"""Run the sample stories through the saved model and record how it did.

These are hand-written, modern-flavoured stories. They are a smoke test of how
the app behaves, not a benchmark.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens.config import RESULTS_DIR  # noqa: E402
from newslens.examples import EXAMPLES  # noqa: E402
from newslens.predictor import Predictor  # noqa: E402

NOT_NEWS = [
    "asdf qwer zxcv uiop hjkl vbnm tyui ghjk",
    "Hi team, please remember to submit your timesheets by Friday afternoon.",
    "To reset your password, click the link below and enter your new password twice.",
    "def add(a, b): return a + b  # this function adds two numbers together",
    "Les enfants jouent dans le jardin pendant que leurs parents préparent le dîner.",
]

predictor = Predictor()
rows = []
for (title, expected, text), result in zip(EXAMPLES, predictor.classify_many([e[2] for e in EXAMPLES])):
    rows.append(
        {
            "title": title,
            "expected": expected,
            "predicted": result.topic,
            "confidence": round(result.confidence, 3),
            "unsure": result.is_unsure,
            "looks_like_news": result.looks_like_news,
            "second": result.ranked[1][0],
        }
    )
labelled = [row for row in rows if row["expected"]]
hits = [row for row in labelled if row["predicted"] == row["expected"]]
not_news = [
    {"text": text, "predicted": r.topic, "confidence": round(r.confidence, 3), "unsure": r.is_unsure,
     "looks_like_news": r.looks_like_news}
    for text, r in zip(NOT_NEWS, predictor.classify_many(NOT_NEWS))
]
summary = {
    "labelled": len(labelled),
    "correct": len(hits),
    "stories_set_aside": sum(not row["looks_like_news"] for row in labelled),
    "not_news_set_aside": sum(not row["looks_like_news"] for row in not_news),
    "not_news_total": len(not_news),
    "stories": rows,
    "not_news": not_news,
}
(RESULTS_DIR / "examples.json").write_text(json.dumps(summary, indent=2))
for row in rows:
    mark = "ok " if row["predicted"] == row["expected"] else ("?  " if not row["expected"] else "BAD")
    print(mark, " " if row["looks_like_news"] else "!", row["title"][:34].ljust(34), str(row["expected"]).ljust(17), row["predicted"].ljust(17), row["confidence"], row["second"])
print(f"{len(hits)}/{len(labelled)} correct, {summary['stories_set_aside']} set aside as not news")
for row in not_news:
    print("set aside" if not row["looks_like_news"] else "MISSED", row["predicted"], row["confidence"], row["text"][:50])
