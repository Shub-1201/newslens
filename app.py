"""NewsLens: paste a news story and see which desk it belongs on.

Run locally with `python app.py`. On Hugging Face Spaces this is the entry point.
"""
import json
import sys
import tempfile
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from newslens import feedback, render  # noqa: E402
from newslens.config import MODEL_DIR, RESULTS_DIR  # noqa: E402
from newslens.examples import EXAMPLES  # noqa: E402
from newslens.predictor import MIN_WORDS, Predictor  # noqa: E402
from newslens.taxonomy import TOPICS  # noqa: E402
from newslens.theme import CSS, HEAD  # noqa: E402

MAX_BATCH_ROWS = 500
SAMPLE_TEXT = {title: text for title, _, text in EXAMPLES}
DESK_TO_KEY = {name: key for key, name in TOPICS.items()}

if not (MODEL_DIR / "calibration.json").exists():
    sys.exit(f"No trained model in {MODEL_DIR}. Run scripts/train.py first; the README has the steps.")

predictor = Predictor()


def route_story(text):
    text = (text or "").strip()
    hide = gr.update(visible=False)
    if len(text.split()) < MIN_WORDS:
        slip = render.slip_message(
            "Story too short", f"Paste at least {MIN_WORDS} words so there is something to read."
        )
        return slip, "", None, hide, ""
    prediction = predictor.classify(text)
    if not prediction.looks_like_news:
        return render.slip(prediction), "", None, hide, ""
    spans = predictor.explain(text, prediction.topic)
    story = render.marked_story(text, spans, prediction.topic, prediction.was_cut_short)
    state = {"text": text, "topic": prediction.topic, "confidence": prediction.confidence}
    return render.slip(prediction), story, state, gr.update(visible=True, value=None), ""


def load_sample(title):
    return SAMPLE_TEXT.get(title, "")


def show_desk_picker(verdict):
    return gr.update(visible=verdict == "No, it's wrong", value=None)


def save_verdict(state, verdict, correct_desk):
    if not state or not verdict:
        return "Choose an answer first."
    if verdict == "No, it's wrong" and not correct_desk:
        return "Choose the right desk first."
    feedback.save(
        predicted=state["topic"],
        confidence=state["confidence"],
        verdict="correct" if verdict.startswith("Yes") else "wrong",
        correct_desk=DESK_TO_KEY.get(correct_desk, state["topic"]),
        text=state["text"],
    )
    return "Saved. Thanks for checking."


def sort_many(pasted, upload):
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            raise gr.Error("This file can't be read as a CSV. Save it as UTF-8 CSV and upload it again.")
        column = next((c for c in frame.columns if c.strip().lower() == "text"), None)
        if column is None:
            raise gr.Error("Add a column named text to the CSV, with one story in each row.")
        stories = frame[column].fillna("").astype(str).str.strip().tolist()
    else:
        stories = [line.strip() for line in (pasted or "").splitlines() if line.strip()]
    total = len(stories)
    stories = [story for story in stories if len(story.split()) >= MIN_WORDS]
    if len(stories) < total:
        gr.Info(f"Skipped {total - len(stories)} under {MIN_WORDS} words.")
    if not stories:
        raise gr.Error(f"Add at least one story of {MIN_WORDS} words or more.")
    if len(stories) > MAX_BATCH_ROWS:
        raise gr.Error(f"Sort up to {MAX_BATCH_ROWS} stories at a time.")
    predictions = predictor.classify_many(stories)
    table = pd.DataFrame(
        {
            "Desk": [render.batch_label(p) for p in predictions],
            "Best guess": [render.desk(p.topic) for p in predictions],
            "Confidence": [f"{p.confidence:.0%}" for p in predictions],
            "Story": [story[:140] for story in stories],
        }
    )
    output = Path(tempfile.mkdtemp()) / "newslens-sorted.csv"
    pd.DataFrame(
        {
            "desk": table["Desk"],
            "best_guess": table["Best guess"],
            "confidence": [round(p.confidence, 4) for p in predictions],
            "text": stories,
        }
    ).to_csv(output, index=False)
    return table, gr.update(value=str(output), visible=True)


def about_html() -> str:
    path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else None
    sections = [
        "<h2>What it does</h2><p>NewsLens reads a news story and picks one of 13 "
        "newsroom desks. It reads about the first 100 words, in English only. "
        "Text that doesn't read like news is set aside rather than forced onto a desk.</p>"
    ]
    if metrics:
        test = metrics["test"]
        sections.append(
            f"<h2>How often it's right</h2><p>On {test['n']:,} stories it never saw "
            f"in training, the top desk matched the label {test['accuracy']:.0%} of the time. "
            f"With a headline alone, that fell to {metrics['test_headlines_only']['accuracy']:.0%}.</p>"
        )
        gate = metrics.get("news_gate")
        if gate:
            missed = 1 - gate["test_stories_passed"]
            sections.append(
                "<h2>When it won't pick a desk</h2><p>It sets text aside if it doesn't read "
                f"like news. That check wrongly set aside {missed:.1%} of real test stories, "
                "mostly health and lifestyle pieces. It knows emails, texts, tweets, reviews, "
                "quotes and code, and may still be fooled by other kinds of writing.</p>"
            )
        rows = "".join(
            f"<tr><td>{escape(TOPICS[key])}</td><td class='number'>{values['f1-score']:.2f}</td></tr>"
            for key, values in sorted(metrics["per_topic"].items(), key=lambda kv: -kv[1]["f1-score"])
        )
        sections.append(
            "<h2>Strongest and weakest desks</h2><p>F1 score: 1.00 would be perfect.</p>"
            f"<table><thead><tr><th>Desk</th><th class='number'>F1</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    sections.append(
        "<h2>What it gets wrong</h2><p>Each story gets one desk, even when it fits two. "
        "Its training stories run from 2004 to 2022, so very recent names and events may "
        "throw it off. It sorts by topic only: it doesn't check whether a story is true.</p>"
        "<h2>Your answers</h2><p>When you save an answer, this app stores the story text "
        "with it. Don't save anything private.</p>"
    )
    return f'<div class="about">{"".join(sections)}</div>'


with gr.Blocks(title="NewsLens") as demo:
    gr.HTML(
        '<header class="masthead"><h1>Which desk gets this story?</h1>'
        "<p>Paste a news story. NewsLens sorts it to one of 13 newsroom desks "
        "and marks the words that swayed it most.</p></header>"
    )
    state = gr.State(None)
    with gr.Tabs():
        with gr.Tab("One story"):
            with gr.Row(elem_classes="workbench"):
                with gr.Column(scale=3):
                    story = gr.Textbox(
                        label="Story",
                        show_label=False,
                        lines=9,
                        max_lines=18,
                        placeholder="Paste a headline and the opening paragraph",
                        elem_classes="story-box",
                    )
                    with gr.Row(elem_classes="controls"):
                        route = gr.Button("Sort this story", elem_classes="route", scale=1)
                        sample = gr.Dropdown(
                            choices=list(SAMPLE_TEXT), label="Or try a sample story",
                            value=None, scale=2,
                        )
                with gr.Column(scale=2):
                    slip = gr.HTML(render.slip_empty())
            with gr.Row(elem_classes="workbench"):
                with gr.Column(scale=3):
                    marked = gr.HTML()
                with gr.Column(scale=2):
                    with gr.Column(visible=False, elem_classes="verdict") as verdict_box:
                        verdict = gr.Radio(
                            ["Yes, that's right", "No, it's wrong"], label="Did it pick the right desk?"
                        )
                        correct = gr.Dropdown(
                            choices=list(TOPICS.values()), label="Which desk should it be?", visible=False
                        )
                        save = gr.Button("Save my answer", elem_classes="quiet")
                        status = gr.Markdown(elem_classes="verdict-status")

        with gr.Tab("Many stories"):
            pasted = gr.Textbox(
                label="One story or headline on each line", lines=8,
                placeholder="Central bank holds rates as inflation cools",
            )
            upload = gr.File(label="Or upload a CSV with a column named text", file_types=[".csv"], type="filepath", height=150)
            sort_button = gr.Button("Sort these stories", elem_classes="route")
            results = gr.Dataframe(
                label="Sorted stories", wrap=True, interactive=False,
                column_widths=["20%", "20%", "13%", "47%"],
            )
            download = gr.DownloadButton("Download sorted CSV", visible=False, elem_classes="quiet")

        with gr.Tab("About"):
            gr.HTML(about_html())

    route_outputs = [slip, marked, state, verdict_box, status]
    route.click(route_story, story, route_outputs)
    sample.change(load_sample, sample, story).then(route_story, story, route_outputs)
    verdict.change(show_desk_picker, verdict, correct)
    save.click(save_verdict, [state, verdict, correct], status)
    sort_button.click(sort_many, [pasted, upload], [results, download])

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        head=HEAD,
        theme=gr.themes.Base(),
        server_name="0.0.0.0",
    )
