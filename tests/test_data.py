import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newslens import data  # noqa: E402
from newslens.taxonomy import BBC_MAP, HUFFPOST_MAP, MNDS_MAP, TOPICS  # noqa: E402


@pytest.mark.parametrize("mapping", [BBC_MAP, MNDS_MAP, HUFFPOST_MAP])
def test_every_mapped_label_is_a_real_desk(mapping):
    assert set(mapping.values()) <= set(TOPICS)


def test_every_desk_has_data():
    assert set(BBC_MAP.values()) | set(MNDS_MAP.values()) | set(HUFFPOST_MAP.values()) == set(TOPICS)


def test_normalise_treats_punctuation_and_case_as_equal():
    assert data.normalise("Rates, RISE!") == data.normalise("rates rise")


def test_leak_check_catches_a_shared_article():
    left = pd.DataFrame({"key": ["a", "b"], "opening": ["a", "b"]})
    right = pd.DataFrame({"key": ["c", "a"], "opening": ["c", "x"]})
    with pytest.raises(ValueError):
        data.assert_no_leakage({"train": left, "test": right})


def test_leak_check_catches_a_shared_opening():
    left = pd.DataFrame({"key": ["a"], "opening": ["same start"]})
    right = pd.DataFrame({"key": ["b"], "opening": ["same start"]})
    with pytest.raises(ValueError):
        data.assert_no_leakage({"train": left, "test": right})


def test_headline_views_only_cover_topics_without_short_text():
    train = pd.DataFrame(
        {
            "text": ["long war story", "long sport story"],
            "title": ["troops seize town after long fight", "club wins cup after long match"],
            "topic": ["conflict", "sport"],
            "source": ["mnds", "mnds"],
        }
    )
    result = data.add_headline_views(train)
    added = result[result["source"] == "mnds_headline"]
    assert added["topic"].tolist() == ["conflict"]
    assert added["text"].tolist() == ["troops seize town after long fight"]
