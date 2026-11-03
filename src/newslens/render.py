"""HTML for the routing slip and the marked-up story.

All user text goes through html.escape, because it comes straight from a
text box and is shown back inside raw HTML.
"""
from html import escape

from .predictor import MIN_WORDS, Prediction, mark_spans
from .taxonomy import TOPICS

SHORT_TEXT_WORDS = 25
MIN_MARKED_WORDS = 4
MARKED_SHARE_OF_WORDS = 1 / 8
STOPWORDS = frozenset(
    """a about after against all also an and any are as at be been before but by can
    could did do does for from had has have he her his how i if in into is it its more
    most near new no not of off on one or our out over said she so some than that the
    their them then there these they this to up us was we were what when which who
    while will with within without would you your""".split()
)


def desk(key: str) -> str:
    return TOPICS[key]


def plain_confidence(value: float) -> str:
    if value >= 0.9:
        return "Very likely"
    if value >= 0.7:
        return "Likely"
    if value >= 0.5:
        return "More likely than not"
    return "Best guess only"


def _bar(name: str, value: float, strong: bool = False) -> str:
    percent = round(value * 100)
    width = max(value * 100, 1)
    state = " strong" if strong else ""
    return (
        f'<div class="odds{state}">'
        f'<span class="odds-name">{escape(name)}</span>'
        f'<span class="odds-value">{percent}%</span>'
        f'<span class="odds-track" aria-hidden="true">'
        f'<span class="odds-fill" style="width:{width:.1f}%"></span></span></div>'
    )


def batch_label(prediction: Prediction) -> str:
    if not prediction.looks_like_news:
        return "Not news"
    if prediction.is_unsure:
        return "Needs an editor"
    return desk(prediction.topic)


def slip_empty() -> str:
    return (
        '<aside class="slip slip-empty" aria-live="polite">'
        '<p class="slip-kicker">Send to</p>'
        '<p class="slip-desk">Waiting for a story</p>'
        f'<p class="slip-note">Paste at least {MIN_WORDS} words, or pick a sample story.</p>'
        "</aside>"
    )


def slip_message(title: str, note: str) -> str:
    return (
        '<aside class="slip slip-empty" aria-live="polite">'
        f'<p class="slip-kicker">Send to</p><p class="slip-desk">{escape(title)}</p>'
        f'<p class="slip-note">{escape(note)}</p></aside>'
    )


def slip(prediction: Prediction) -> str:
    top, confidence = prediction.ranked[0]
    if not prediction.looks_like_news:
        kicker = "Not sorted"
        headline = "This doesn't read like news"
        note = "NewsLens only sorts news stories. If this is one, these desks came closest."
        rows = "".join(_bar(desk(k), v) for k, v in prediction.ranked[:3])
        extra = f'<div class="odds-list">{rows}</div>'
        state = " slip-unsure"
    elif prediction.is_unsure:
        kicker = "Needs an editor"
        headline = "No clear desk"
        note = "This could fit more than one desk, or it may not be a news story."
        rows = "".join(_bar(desk(k), v) for k, v in prediction.ranked[:3])
        extra = f'<div class="odds-list">{rows}</div>'
        state = " slip-unsure"
    else:
        kicker = "Send to"
        headline = desk(top)
        note = plain_confidence(confidence)
        if prediction.word_count < SHORT_TEXT_WORDS:
            note += ". Short text, so treat this with care. Add the opening paragraph for a better read."
        rows = _bar(desk(top), confidence, strong=True)
        rows += "".join(_bar(desk(k), v) for k, v in prediction.ranked[1:3])
        extra = f'<div class="odds-list">{rows}</div>'
        state = ""
    return (
        f'<aside class="slip{state}" aria-live="polite">'
        f'<p class="slip-kicker">{kicker}</p>'
        f'<p class="slip-desk">{escape(headline)}</p>'
        f'<p class="slip-note">{escape(note)}</p>{extra}</aside>'
    )


def _words_to_mark(text: str, spans: list[dict]) -> list[dict]:
    """The few words worth a reader's eye: strong, and carrying meaning."""
    def counts(span):
        word = text[span["start"] : span["end"]]
        return any(c.isalnum() for c in word) and word.lower() not in STOPWORDS

    strongest = sorted(filter(counts, mark_spans(spans)), key=lambda s: -abs(s["score"]))
    room = max(MIN_MARKED_WORDS, round(len(spans) * MARKED_SHARE_OF_WORDS))
    return sorted(strongest[:room], key=lambda s: s["start"])


def marked_story(text: str, spans: list[dict], topic: str, was_cut_short: bool) -> str:
    kept = _words_to_mark(text, spans)
    read_until = spans[-1]["end"] if spans else 0
    parts, cursor = [], 0
    for span in kept:
        parts.append(escape(text[cursor : span["start"]]))
        word = escape(text[span["start"] : span["end"]])
        kind = "for" if span["score"] > 0 else "against"
        parts.append(f'<mark class="{kind}">{word}</mark>')
        cursor = span["end"]
    parts.append(escape(text[cursor:read_until]))
    unread = ""
    if was_cut_short:
        unread = (
            '<p class="unread-note">The model stopped reading here, about the first '
            "100 words. News stories usually state their topic early.</p>"
        )
    key = (
        '<p class="mark-key"><span><mark class="for">Highlighted</mark> '
        f"pointed toward {escape(desk(topic))}</span>"
        '<span><mark class="against">Underlined</mark> pointed away from it</span></p>'
    )
    return (
        f'<section class="story" aria-label="Your story, marked up">{key}'
        f'<div class="story-text">{"".join(parts)}</div>{unread}</section>'
    )
