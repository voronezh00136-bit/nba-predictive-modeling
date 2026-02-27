"""
Automated data pipeline for NBA box scores and injury reports.

Handles scraping, cleaning, and aggregating daily NBA data used by
the player-prop and game-probability models.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def clean_box_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a raw box-score DataFrame.

    Parameters
    ----------
    raw:
        DataFrame that must contain at least the columns listed in
        ``REQUIRED_BOX_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with standardised column names, correct dtypes, and
        derived advanced metrics (True Shooting %, Pace, etc.).
    """
    REQUIRED_BOX_COLUMNS = {
        "player", "team", "opponent", "game_date",
        "points", "rebounds", "assists",
        "fga", "fta", "fg3a",
        "minutes",
    }
    missing = REQUIRED_BOX_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Box-score data is missing columns: {missing}")

    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    numeric_cols = [
        "points", "rebounds", "assists",
        "fga", "fta", "fg3a", "minutes",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["player", "team", "game_date"])

    df = _add_advanced_metrics(df)
    return df.reset_index(drop=True)


def aggregate_player_stats(
    box_scores: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Compute rolling averages and standard deviations per player.

    Parameters
    ----------
    box_scores:
        Cleaned box-score frame returned by :func:`clean_box_scores`.
    window:
        Number of most-recent games used for the rolling window.

    Returns
    -------
    pd.DataFrame
        One row per (player, game_date) with rolling statistics appended.
    """
    df = box_scores.sort_values(["player", "game_date"]).copy()

    stat_cols = ["points", "rebounds", "assists", "true_shooting_pct"]
    for col in stat_cols:
        if col not in df.columns:
            continue
        grp = df.groupby("player")[col]
        df[f"{col}_avg{window}"] = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"{col}_std{window}"] = grp.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

    return df.reset_index(drop=True)


def build_game_features(
    box_scores: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Construct per-game team-level feature matrix.

    Advanced metrics (Pace, Defensive Rating) and back-to-back scheduling
    flags are included so that the game-probability model can learn from
    contextual workload signals.

    Parameters
    ----------
    box_scores:
        Cleaned box-score frame.
    schedule:
        DataFrame with columns ``team``, ``game_date``, ``home`` (bool),
        and ``opponent``.

    Returns
    -------
    pd.DataFrame
        One row per (team, game_date) with aggregated features.
    """
    REQUIRED_SCHEDULE_COLS = {"team", "game_date", "home", "opponent"}
    missing = REQUIRED_SCHEDULE_COLS - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule data is missing columns: {missing}")

    schedule = schedule.copy()
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])

    team_stats = (
        box_scores.groupby(["team", "game_date"])
        .agg(
            team_points=("points", "sum"),
            team_rebounds=("rebounds", "sum"),
            team_assists=("assists", "sum"),
            avg_true_shooting=("true_shooting_pct", "mean"),
            pace=("pace", "mean"),
        )
        .reset_index()
    )

    features = schedule.merge(team_stats, on=["team", "game_date"], how="left")
    features = _flag_back_to_back(features)
    return features.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Append True Shooting % and a simplified Pace estimate."""
    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    denom = 2 * (df["fga"] + 0.44 * df["fta"])
    df["true_shooting_pct"] = np.where(
        denom > 0, df["points"] / denom, np.nan
    )

    # Simplified per-player pace proxy: possessions ≈ FGA + 0.44*FTA - ORB + TO
    # When turnovers / offensive rebounds are unavailable we use FGA + 0.44*FTA.
    if "turnovers" in df.columns and "off_rebounds" in df.columns:
        df["pace"] = df["fga"] + 0.44 * df["fta"] - df["off_rebounds"] + df["turnovers"]
    else:
        df["pace"] = df["fga"] + 0.44 * df["fta"]

    return df


def _flag_back_to_back(schedule: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean ``back_to_back`` column to a schedule frame."""
    schedule = schedule.sort_values(["team", "game_date"]).copy()
    prev_date = schedule.groupby("team")["game_date"].shift(1)
    schedule["back_to_back"] = (
        (schedule["game_date"] - prev_date).dt.days == 1
    )
    return schedule
