"""Tests for models/game_probability.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.game_probability import GameProbabilityModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game_dataset(n: int = 120):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "home": rng.integers(0, 2, n).astype(float),
            "back_to_back": rng.integers(0, 2, n).astype(float),
            "team_points": rng.normal(110, 10, n),
            "team_rebounds": rng.normal(45, 5, n),
            "team_assists": rng.normal(25, 4, n),
            "avg_true_shooting": rng.uniform(0.50, 0.65, n),
            "pace": rng.normal(100, 5, n),
            "opp_team_points": rng.normal(110, 10, n),
            "opp_team_rebounds": rng.normal(45, 5, n),
            "opp_avg_true_shooting": rng.uniform(0.50, 0.65, n),
            "opp_pace": rng.normal(100, 5, n),
            "opp_back_to_back": rng.integers(0, 2, n).astype(float),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="win")
    return X, y


# ---------------------------------------------------------------------------
# GameProbabilityModel
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    X, y = _make_game_dataset()
    model = GameProbabilityModel(cv_folds=2)
    result = model.fit(X, y)
    assert result is model


def test_predict_proba_shape():
    X, y = _make_game_dataset()
    model = GameProbabilityModel(cv_folds=2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_sums_to_one():
    X, y = _make_game_dataset()
    model = GameProbabilityModel(cv_folds=2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_win_prob_in_range():
    X, y = _make_game_dataset()
    model = GameProbabilityModel(cv_folds=2)
    model.fit(X, y)
    probs = model.predict_win_prob(X)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_before_fit_raises():
    model = GameProbabilityModel()
    X, _ = _make_game_dataset(10)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_proba(X)


def test_cv_roc_auc_after_fit():
    X, y = _make_game_dataset()
    model = GameProbabilityModel(cv_folds=2)
    model.fit(X, y)
    assert model.cv_roc_auc_ is not None
    assert 0.0 <= model.cv_roc_auc_ <= 1.0
