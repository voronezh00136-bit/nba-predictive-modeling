"""
Tests for src/features/engineering.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_true_shooting,
    add_pace,
    add_offensive_rating,
    add_defensive_rating,
    add_net_rating,
    build_game_features,
)


def _player_row(**kwargs) -> pd.DataFrame:
    """Build a minimal single-row player DataFrame."""
    defaults = {
        "points": 20.0,
        "fg_attempted": 15.0,
        "ft_attempted": 5.0,
        "fg3_attempted": 3.0,
        "turnovers": 2.0,
        "off_reb": 1.0,
        "minutes": 35.0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# True Shooting %
# ---------------------------------------------------------------------------

class TestTrueShooting:
    def test_formula(self):
        df = _player_row(points=20, fg_attempted=15, ft_attempted=4)
        result = add_true_shooting(df)
        expected = 20 / (2 * (15 + 0.44 * 4))
        assert result["true_shooting_pct"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_zero_denominator_returns_nan(self):
        df = _player_row(points=0, fg_attempted=0, ft_attempted=0)
        result = add_true_shooting(df)
        assert np.isnan(result["true_shooting_pct"].iloc[0])

    def test_original_unchanged(self):
        df = _player_row()
        original_cols = set(df.columns)
        add_true_shooting(df)  # should not mutate input
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Pace
# ---------------------------------------------------------------------------

class TestPace:
    def test_pace_positive(self):
        df = _player_row(fg_attempted=15, ft_attempted=4, turnovers=2, off_reb=1, minutes=40)
        result = add_pace(df)
        assert result["pace"].iloc[0] > 0

    def test_pace_zero_minutes_returns_nan(self):
        df = _player_row(minutes=0)
        result = add_pace(df)
        assert np.isnan(result["pace"].iloc[0])


# ---------------------------------------------------------------------------
# Offensive / Defensive Rating
# ---------------------------------------------------------------------------

class TestRatings:
    def test_offensive_rating_positive(self):
        df = _player_row(points=25)
        result = add_offensive_rating(df)
        assert result["offensive_rating"].iloc[0] > 0

    def test_defensive_rating_missing_col(self):
        df = _player_row()
        result = add_defensive_rating(df, opp_pts_col="opp_points")
        assert np.isnan(result["defensive_rating"].iloc[0])

    def test_defensive_rating_with_opp(self):
        df = _player_row()
        df["opp_points"] = 18.0
        result = add_defensive_rating(df, opp_pts_col="opp_points")
        assert result["defensive_rating"].iloc[0] > 0

    def test_net_rating_direction(self):
        df = _player_row(points=30)
        df["opp_points"] = 20.0
        result = add_net_rating(add_defensive_rating(add_offensive_rating(df)))
        # Scoring more than allowing → positive net rating
        assert result["net_rating"].iloc[0] > 0


# ---------------------------------------------------------------------------
# build_game_features
# ---------------------------------------------------------------------------

class TestBuildGameFeatures:
    def test_adds_all_columns(self):
        df = _player_row()
        result = build_game_features(df)
        for col in ("true_shooting_pct", "pace", "offensive_rating", "net_rating"):
            assert col in result.columns

    def test_no_mutation(self):
        df = _player_row()
        cols_before = list(df.columns)
        build_game_features(df)
        assert list(df.columns) == cols_before
