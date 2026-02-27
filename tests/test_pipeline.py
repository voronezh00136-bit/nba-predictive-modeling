"""Tests for data/pipeline.py"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from data.pipeline import (
    clean_box_scores,
    aggregate_player_stats,
    build_game_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_box_scores(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "player": [f"Player_{i % 5}" for i in range(n)],
            "team": [f"Team_{i % 3}" for i in range(n)],
            "opponent": [f"Team_{(i + 1) % 3}" for i in range(n)],
            "game_date": dates,
            "points": rng.integers(5, 35, size=n).astype(float),
            "rebounds": rng.integers(1, 15, size=n).astype(float),
            "assists": rng.integers(0, 12, size=n).astype(float),
            "fga": rng.integers(5, 25, size=n).astype(float),
            "fta": rng.integers(0, 10, size=n).astype(float),
            "fg3a": rng.integers(0, 8, size=n).astype(float),
            "minutes": rng.integers(10, 40, size=n).astype(float),
        }
    )


def _make_schedule(n: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "team": [f"Team_{i % 3}" for i in range(n)],
            "opponent": [f"Team_{(i + 1) % 3}" for i in range(n)],
            "game_date": dates,
            "home": [i % 2 == 0 for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# clean_box_scores
# ---------------------------------------------------------------------------

def test_clean_box_scores_returns_dataframe():
    raw = _make_raw_box_scores()
    result = clean_box_scores(raw)
    assert isinstance(result, pd.DataFrame)


def test_clean_box_scores_adds_true_shooting():
    raw = _make_raw_box_scores()
    result = clean_box_scores(raw)
    assert "true_shooting_pct" in result.columns


def test_clean_box_scores_adds_pace():
    raw = _make_raw_box_scores()
    result = clean_box_scores(raw)
    assert "pace" in result.columns


def test_clean_box_scores_missing_columns_raises():
    bad = pd.DataFrame({"player": ["A"], "team": ["T"]})
    with pytest.raises(ValueError, match="missing columns"):
        clean_box_scores(bad)


def test_clean_box_scores_game_date_is_datetime():
    raw = _make_raw_box_scores()
    raw["game_date"] = raw["game_date"].astype(str)
    result = clean_box_scores(raw)
    assert pd.api.types.is_datetime64_any_dtype(result["game_date"])


# ---------------------------------------------------------------------------
# aggregate_player_stats
# ---------------------------------------------------------------------------

def test_aggregate_player_stats_adds_rolling_cols():
    raw = _make_raw_box_scores(30)
    cleaned = clean_box_scores(raw)
    agg = aggregate_player_stats(cleaned, window=5)
    assert "points_avg5" in agg.columns
    assert "points_std5" in agg.columns


def test_aggregate_player_stats_row_count_preserved():
    raw = _make_raw_box_scores(20)
    cleaned = clean_box_scores(raw)
    agg = aggregate_player_stats(cleaned)
    assert len(agg) == len(cleaned)


# ---------------------------------------------------------------------------
# build_game_features
# ---------------------------------------------------------------------------

def test_build_game_features_returns_dataframe():
    box = clean_box_scores(_make_raw_box_scores(20))
    sched = _make_schedule(6)
    result = build_game_features(box, sched)
    assert isinstance(result, pd.DataFrame)


def test_build_game_features_has_back_to_back():
    box = clean_box_scores(_make_raw_box_scores(20))
    sched = _make_schedule(6)
    result = build_game_features(box, sched)
    assert "back_to_back" in result.columns


def test_build_game_features_missing_schedule_cols_raises():
    box = clean_box_scores(_make_raw_box_scores())
    bad_sched = pd.DataFrame({"team": ["A"]})
    with pytest.raises(ValueError, match="missing columns"):
        build_game_features(box, bad_sched)
