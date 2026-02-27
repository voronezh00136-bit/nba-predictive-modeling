"""
Tests for src/models/game_predictor.py and src/models/player_props.py
"""

import numpy as np
import pandas as pd
import pytest

from src.models.game_predictor import GamePredictor, FEATURE_COLS as GAME_COLS
from src.models.player_props import (
    PlayerPropsModel,
    american_odds_to_implied_prob,
    decimal_odds_to_implied_prob,
    detect_inefficiency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "home_rolling_pts": rng.normal(110, 5, n),
            "away_rolling_pts": rng.normal(108, 5, n),
            "home_back_to_back": rng.integers(0, 2, n),
            "away_back_to_back": rng.integers(0, 2, n),
            "home_net_rating": rng.normal(2, 3, n),
            "away_net_rating": rng.normal(0, 3, n),
            "home_key_player_out": rng.integers(0, 2, n),
            "away_key_player_out": rng.integers(0, 2, n),
            "home_win": rng.integers(0, 2, n),
        }
    )
    return df


def _make_player_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "rolling_points": rng.normal(22, 4, n),
            "rolling_rebounds": rng.normal(5, 2, n),
            "rolling_assists": rng.normal(4, 1.5, n),
            "true_shooting_pct": rng.uniform(0.48, 0.65, n),
            "minutes": rng.normal(33, 3, n),
            "back_to_back": rng.integers(0, 2, n),
            "home_game": rng.integers(0, 2, n),
            "points": rng.normal(22, 6, n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# GamePredictor tests
# ---------------------------------------------------------------------------

class TestGamePredictor:
    def test_fit_predict_proba_shape(self):
        df = _make_game_df(n=120)
        predictor = GamePredictor()
        predictor.fit(df)
        probs = predictor.predict_proba(df.head(10))
        assert probs.shape == (10,)

    def test_predict_proba_range(self):
        df = _make_game_df()
        predictor = GamePredictor()
        predictor.fit(df)
        probs = predictor.predict_proba(df)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_binary(self):
        df = _make_game_df()
        predictor = GamePredictor()
        predictor.fit(df)
        preds = predictor.predict(df)
        assert set(preds).issubset({0, 1})

    def test_raises_without_fit(self):
        predictor = GamePredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict_proba(_make_game_df().head(5))

    def test_raises_on_missing_target(self):
        df = _make_game_df().drop(columns=["home_win"])
        predictor = GamePredictor()
        with pytest.raises(ValueError, match="home_win"):
            predictor.fit(df)

    def test_raises_on_no_features(self):
        df = pd.DataFrame({"home_win": [0, 1, 0]})
        predictor = GamePredictor()
        with pytest.raises(ValueError, match="feature"):
            predictor.fit(df)


# ---------------------------------------------------------------------------
# PlayerPropsModel tests
# ---------------------------------------------------------------------------

class TestPlayerPropsModel:
    def test_create_over_target(self):
        df = _make_player_df()
        df2 = PlayerPropsModel.create_over_target(df, "points", line=22.0)
        assert "over_points" in df2.columns
        assert df2["over_points"].isin([0, 1]).all()

    def test_fit_predict_proba(self):
        df = _make_player_df()
        df = PlayerPropsModel.create_over_target(df, "points", line=22.0)
        model = PlayerPropsModel(stat="points")
        model.fit(df)
        probs = model.predict_proba(df.head(10))
        assert probs.shape == (10,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_raises_without_fit(self):
        model = PlayerPropsModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba(_make_player_df().head(5))

    def test_raises_missing_target(self):
        df = _make_player_df()
        model = PlayerPropsModel(stat="points")
        with pytest.raises(ValueError, match="over_points"):
            model.fit(df)

    def test_scan_for_value(self):
        df = _make_player_df()
        df = PlayerPropsModel.create_over_target(df, "points", line=22.0)
        model = PlayerPropsModel(stat="points")
        model.fit(df)

        lines = pd.Series([22.0] * 10)
        market_odds = pd.Series([-110.0] * 10)
        result = model.scan_for_value(df.head(10), lines, market_odds)
        for col in ("model_prob", "market_implied", "edge", "value"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# Odds conversion helpers
# ---------------------------------------------------------------------------

class TestOddsHelpers:
    def test_american_negative(self):
        prob = american_odds_to_implied_prob(-110)
        assert prob == pytest.approx(110 / 210, rel=1e-6)

    def test_american_positive(self):
        prob = american_odds_to_implied_prob(+200)
        assert prob == pytest.approx(100 / 300, rel=1e-6)

    def test_decimal(self):
        prob = decimal_odds_to_implied_prob(2.0)
        assert prob == pytest.approx(0.5, rel=1e-6)

    def test_detect_inefficiency_value(self):
        result = detect_inefficiency(0.60, -110, odds_format="american", min_edge=0.03)
        assert result["value"] is True
        assert result["edge"] > 0

    def test_detect_inefficiency_no_value(self):
        result = detect_inefficiency(0.52, -110, odds_format="american", min_edge=0.05)
        assert result["value"] is False
