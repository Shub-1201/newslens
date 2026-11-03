import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens import feedback, render  # noqa: E402
from newslens.examples import EXAMPLES  # noqa: E402
from newslens.predictor import Prediction  # noqa: E402
from newslens.taxonomy import TOPICS  # noqa: E402


def _prediction(confidence):
    return Prediction(
        ranked=[("politics", confidence), ("business", 1 - confidence)],
        word_count=40,
        was_cut_short=False,
    )


def test_user_text_cannot_inject_html():
    text = "<script>alert(1)</script> markets fell"
    spans = [{"start": 0, "end": 8, "score": 1.0}, {"start": 34, "end": 38, "score": 0.5}]
    html = render.marked_story(text, spans, "business", False)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_punctuation_is_never_highlighted():
    spans = [{"start": 0, "end": 3, "score": 1.0}, {"start": 3, "end": 4, "score": 1.0}]
    html = render.marked_story("war, again", spans, "conflict", False)
    assert html.count('<mark class="for">') == 2  # The word and the legend, not the comma.


def test_unsure_slip_does_not_name_a_desk_as_its_answer():
    html = render.slip(_prediction(0.3))
    assert "No clear desk" in html and "Needs an editor" in html


def test_confident_slip_names_the_desk():
    assert "Politics" in render.slip(_prediction(0.95))


def test_feedback_writes_a_header_once(tmp_path):
    path = tmp_path / "answers.csv"
    for _ in range(2):
        feedback.save("sport", 0.9, "correct", "sport", "some story", path=path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2 and rows[0]["verdict"] == "correct"


def test_samples_cover_every_desk():
    assert {desk for _, desk, _ in EXAMPLES if desk} == set(TOPICS)


def test_text_that_is_not_news_gets_no_desk():
    prediction = _prediction(0.95)
    prediction.looks_like_news = False
    html = render.slip(prediction)
    assert "read like news" in html and "Send to" not in html
    assert render.batch_label(prediction) == "Not news"


def test_short_text_carries_a_warning_but_long_text_does_not():
    short = _prediction(0.95)
    short.word_count = 8
    assert "Short text" in render.slip(short)
    assert "Short text" not in render.slip(_prediction(0.95))
