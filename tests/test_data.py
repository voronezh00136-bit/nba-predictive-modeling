"""
Tests for src/data/cleaner.py and src/data/aggregator.py
"""

import pandas as pd
import numpy as np
import pytest

from src.data.cleaner import clean_box_scores, clean_player_logs
from src.data.aggregator import (
    add_back_to_back_flag,
    aggregate_team_rolling,
    aggregate_player_rolling,
    merge_injury_flags,
)


# ---------------------------------------------------------------------------
# clean_box_scores
# ---------------------------------------------------------------------------

def _raw_box_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["Mon, Oct 23, 2023", "Tue, Oct 24, 2023", "bad_date"],
            "home_team": ["Lakers", "Celtics", "Warriors"],
            "away_team": ["Nuggets", "76ers", "Suns"],
            "home_pts": ["114", "108", "notanumber"],
            "away_pts": ["107", "112", "99"],
        }
    )


class TestCleanBoxScores:
    def test_drops_bad_scores(self):
        df = clean_box_scores(_raw_box_scores())
        # Row with 'notanumber' in home_pts should be dropped
        assert len(df) == 2

    def test_parses_dates(self):
        df = clean_box_scores(_raw_box_scores())
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_home_win_column(self):
        df = clean_box_scores(_raw_box_scores())
        # Game 1: home 114 > away 107 → home_win=1
        assert df.loc[0, "home_win"] == 1
        # Game 2: home 108 < away 112 → home_win=0
        assert df.loc[1, "home_win"] == 0

    def test_scores_are_int(self):
        df = clean_box_scores(_raw_box_scores())
        assert df["home_pts"].dtype == int
        assert df["away_pts"].dtype == int


# ---------------------------------------------------------------------------
# clean_player_logs
# ---------------------------------------------------------------------------

def _raw_player_logs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_game": ["2023-10-24", "2023-10-26", ""],
            "pts": ["22", "18", "Did Not Play"],
            "trb": ["5", "7", "0"],
            "ast": ["3", "4", "0"],
            "fg": ["8", "6", "0"],
            "fga": ["16", "14", "0"],
            "ft": ["6", "6", "0"],
            "fta": ["7", "7", "0"],
            "mp": ["35", "32", "0"],
            "tov": ["2", "1", "0"],
            "player_slug": ["jamesle01", "jamesle01", "jamesle01"],
        }
    )


class TestCleanPlayerLogs:
    def test_drops_non_numeric_pts(self):
        df = clean_player_logs(_raw_player_logs())
        # Row with "Did Not Play" in pts should be dropped
        assert len(df) == 2

    def test_renames_columns(self):
        df = clean_player_logs(_raw_player_logs())
        assert "points" in df.columns
        assert "rebounds" in df.columns
        assert "assists" in df.columns

    def test_numeric_types(self):
        df = clean_player_logs(_raw_player_logs())
        assert pd.api.types.is_numeric_dtype(df["points"])


# ---------------------------------------------------------------------------
# add_back_to_back_flag
# ---------------------------------------------------------------------------

def _game_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-10-23", "2023-10-24", "2023-10-26"]),
            "home_team": ["Lakers", "Lakers", "Lakers"],
            "home_pts": [114, 108, 120],
            "away_team": ["Nuggets", "Bulls", "Heat"],
            "away_pts": [107, 112, 115],
        }
    )


class TestBackToBack:
    def test_second_game_is_b2b(self):
        df = add_back_to_back_flag(_game_df(), team_col="home_team")
        assert df["back_to_back"].iloc[1]

    def test_first_game_not_b2b(self):
        df = add_back_to_back_flag(_game_df(), team_col="home_team")
        assert not df["back_to_back"].iloc[0]

    def test_gap_game_not_b2b(self):
        df = add_back_to_back_flag(_game_df(), team_col="home_team")
        # 3rd game is 2 days after 2nd → not B2B
        assert not df["back_to_back"].iloc[2]


# ---------------------------------------------------------------------------
# aggregate_team_rolling
# ---------------------------------------------------------------------------

class TestTeamRolling:
    def test_rolling_pts_added(self):
        df = _game_df()
        result = aggregate_team_rolling(df, window=2, team_col="home_team", pts_col="home_pts")
        assert "rolling_pts" in result.columns

    def test_first_row_nan_or_number(self):
        df = _game_df()
        result = aggregate_team_rolling(df, window=2, team_col="home_team", pts_col="home_pts")
        # First row has no prior game → rolling is NaN (shift) or just NaN
        assert pd.isna(result["rolling_pts"].iloc[0])

    def test_shifted(self):
        """Rolling should use shift(1) so it never leaks future data."""
        df = _game_df()
        result = aggregate_team_rolling(df, window=10, team_col="home_team", pts_col="home_pts")
        # Second game's rolling_pts should equal the first game's pts
        assert result["rolling_pts"].iloc[1] == pytest.approx(114.0)


# ---------------------------------------------------------------------------
# merge_injury_flags
# ---------------------------------------------------------------------------

class TestMergeInjuryFlags:
    def test_flags_team_with_out(self):
        game_df = _game_df()
        injury_df = pd.DataFrame(
            {"team": ["Lakers"], "player": ["LeBron James"], "status": ["Out"], "description": ["Knee"]}
        )
        result = merge_injury_flags(game_df, injury_df, team_col="home_team")
        assert result["key_player_out"].all()

    def test_empty_injury_df(self):
        game_df = _game_df()
        result = merge_injury_flags(game_df, pd.DataFrame(), team_col="home_team")
        assert not result["key_player_out"].any()
