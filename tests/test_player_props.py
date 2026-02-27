"""Tests for models/player_props.py"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from models.player_props import PlayerPropsModel, PROP_TARGETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 100):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "points_avg10": rng.normal(20, 5, n),
            "points_std10": rng.uniform(1, 5, n),
            "rebounds_avg10": rng.normal(6, 2, n),
            "rebounds_std10": rng.uniform(0.5, 2, n),
            "assists_avg10": rng.normal(4, 2, n),
            "assists_std10": rng.uniform(0.5, 2, n),
            "true_shooting_pct_avg10": rng.uniform(0.4, 0.7, n),
            "minutes": rng.uniform(20, 38, n),
            "back_to_back": rng.integers(0, 2, n).astype(float),
        }
    )
    targets = pd.DataFrame(
        {
            "points": rng.normal(20, 5, n).clip(0),
            "rebounds": rng.normal(6, 2, n).clip(0),
            "assists": rng.normal(4, 2, n).clip(0),
        }
    )
    return X, targets


# ---------------------------------------------------------------------------
# PlayerPropsModel
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X, targets = _make_dataset()
    model = PlayerPropsModel(cv_folds=2)
    result = model.fit(X, targets)
    assert result is model


def test_predict_shape():
    X, targets = _make_dataset()
    model = PlayerPropsModel(cv_folds=2)
    model.fit(X, targets)
    preds = model.predict(X)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (len(X), len(PROP_TARGETS))


def test_predict_before_fit_raises():
    model = PlayerPropsModel()
    X, _ = _make_dataset(10)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(X)


def test_cv_scores_populated_after_fit():
    X, targets = _make_dataset()
    model = PlayerPropsModel(cv_folds=2)
    model.fit(X, targets)
    for target in PROP_TARGETS:
        assert target in model.cv_scores_
        assert model.cv_scores_[target] >= 0


def test_find_inefficiencies_returns_dataframe():
    X, targets = _make_dataset(80)
    model = PlayerPropsModel(cv_folds=2)
    model.fit(X, targets)

    rng = np.random.default_rng(2)
    # Set very low market lines so the model finds over-edges
    market_lines = pd.DataFrame(
        {
            "points": np.full(len(X), 5.0),
            "rebounds": np.full(len(X), 1.0),
            "assists": np.full(len(X), 0.5),
        },
        index=X.index,
    )
    result = model.find_inefficiencies(X, market_lines, threshold=1.0)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "edge" in result.columns
        assert "direction" in result.columns


def test_find_inefficiencies_direction_values():
    X, targets = _make_dataset(80)
    model = PlayerPropsModel(cv_folds=2)
    model.fit(X, targets)

    market_lines = pd.DataFrame(
        {
            "points": np.full(len(X), 5.0),
            "rebounds": np.full(len(X), 1.0),
            "assists": np.full(len(X), 0.5),
        },
        index=X.index,
    )
    result = model.find_inefficiencies(X, market_lines, threshold=0.5)
    if not result.empty:
        assert set(result["direction"].unique()).issubset({"over", "under"})
