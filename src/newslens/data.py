"""Combine three news datasets into one clean, leak-free corpus."""
import re

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import CONFIG, TrainConfig
from .taxonomy import (
    BBC_MAP,
    HUFFPOST_MAP,
    LONG_ARTICLE_ONLY_TOPICS,
    MNDS_MAP,
    TOPIC_TO_ID,
)


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _bbc() -> pd.DataFrame:
    dataset = load_dataset("SetFit/bbc-news")
    raw = pd.concat([dataset[name].to_pandas() for name in dataset])
    return pd.DataFrame(
        {"text": raw["text"], "topic": raw["label_text"].map(BBC_MAP), "source": "bbc"}
    )


def _mnds() -> pd.DataFrame:
    raw = load_dataset("textminr/mn-ds")["train"].to_pandas()
    text = raw["title"].fillna("") + ". " + raw["content"].fillna("")
    return pd.DataFrame(
        {
            "text": text,
            "topic": raw["category_level_1"].map(MNDS_MAP),
            "source": "mnds",
            "title": raw["title"].fillna(""),
        }
    )


def _huffpost(per_topic: int) -> pd.DataFrame:
    raw = load_dataset("heegyu/news-category-dataset")["train"].to_pandas()
    raw["topic"] = raw["category"].map(HUFFPOST_MAP)
    raw = raw.dropna(subset=["topic"]).sort_values("date", ascending=False)
    # Newest first, so later years outweigh the 2012-2017 bulk of this dataset.
    raw = raw.groupby("topic", group_keys=False).head(per_topic)
    text = raw["headline"].fillna("") + ". " + raw["short_description"].fillna("")
    return pd.DataFrame({"text": text, "topic": raw["topic"], "source": "huffpost"})


def build_corpus(config: TrainConfig = CONFIG) -> tuple[pd.DataFrame, dict]:
    combined = pd.concat(
        [_bbc(), _mnds(), _huffpost(config.huffpost_per_topic)], ignore_index=True
    )
    rows_in = len(combined)
    unmapped = int(combined["topic"].isna().sum())
    combined = combined.dropna(subset=["topic"])

    combined["word_count"] = combined["text"].str.split().str.len()
    combined = combined[combined["word_count"] >= config.min_words]

    # Two keys: the full text catches exact copies, and the opening catches
    # wire stories that outlets republish with a different ending.
    combined["key"] = combined["text"].map(normalise)
    combined["opening"] = combined["key"].str[:100]
    deduped = combined.drop_duplicates("key").drop_duplicates("opening")

    deduped = deduped.reset_index(drop=True)
    deduped["label"] = deduped["topic"].map(TOPIC_TO_ID).astype(int)
    audit = {
        "rows_in": rows_in,
        "dropped_unmapped_topic": unmapped,
        "dropped_duplicates": len(combined) - len(deduped),
        "rows_out": len(deduped),
        "by_source": deduped["source"].value_counts().to_dict(),
        "by_topic": deduped["topic"].value_counts().to_dict(),
    }
    return deduped, audit


def make_splits(
    corpus: pd.DataFrame, config: TrainConfig = CONFIG
) -> dict[str, pd.DataFrame]:
    strata = corpus["topic"] + "|" + corpus["source"]
    holdout_size = config.val_fraction + config.test_fraction
    train, holdout = train_test_split(
        corpus, test_size=holdout_size, stratify=strata, random_state=config.seed
    )
    val, test = train_test_split(
        holdout,
        test_size=config.test_fraction / holdout_size,
        stratify=holdout["topic"] + "|" + holdout["source"],
        random_state=config.seed,
    )
    return {
        name: frame.reset_index(drop=True)
        for name, frame in {"train": train, "val": val, "test": test}.items()
    }


def add_headline_views(train: pd.DataFrame, config: TrainConfig = CONFIG) -> pd.DataFrame:
    """Teach short-text forms of topics that only exist as long articles.

    Conflict and disaster stories come only from MN-DS, so without this the
    model never sees them as a headline. Runs on the training split only.
    """
    has_title = train["title"].fillna("").str.split().str.len() >= config.min_words
    rows = train[has_title & train["topic"].isin(LONG_ARTICLE_ONLY_TOPICS)].copy()
    rows["text"] = rows["title"]
    rows["source"] = "mnds_headline"
    return pd.concat([train, rows], ignore_index=True)


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for column in ("key", "opening"):
                shared = set(splits[left][column]) & set(splits[right][column])
                if shared:
                    raise ValueError(f"{len(shared)} {column} in {left} and {right}")
