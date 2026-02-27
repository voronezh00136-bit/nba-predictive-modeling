"""
Advanced NBA feature engineering.

Computes standard analytics metrics used throughout the pipeline:

* **True Shooting %** (TS%)  — shooting efficiency accounting for 3-pointers and free throws.
* **Pace**                   — estimated possessions per 48 minutes.
* **Offensive Rating**       — points per 100 possessions (offensive end).
* **Defensive Rating**       — points allowed per 100 possessions.
* **Net Rating**             — offensive minus defensive rating.

All functions accept a ``pd.DataFrame`` and return a new DataFrame with the
additional columns appended.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_true_shooting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``true_shooting_pct`` column.

    Formula:
        TS% = PTS / (2 * (FGA + 0.44 * FTA))

    Columns required: ``points``, ``fg_attempted``, ``ft_attempted``.
    """
    df = df.copy()
    denominator = 2 * (df["fg_attempted"] + 0.44 * df["ft_attempted"])
    df["true_shooting_pct"] = np.where(
        denominator > 0,
        df["points"] / denominator,
        np.nan,
    )
    return df


def add_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate **Pace** (possessions per 48 minutes) for a team game log.

    Approximation (single-team version):
        Possessions ≈ FGA + 0.4 * FTA - 1.07 * (ORB / (ORB + Opp_DRB)) * (FGA - FGM) + TOV

    When opponent rebounds are unavailable a simplified version is used:
        Possessions ≈ FGA - ORB + TOV + 0.4 * FTA

    Columns used (when available): ``fg_attempted``, ``ft_attempted``,
    ``turnovers``, and ``off_reb``.
    """
    df = df.copy()
    fga = df.get("fg_attempted", pd.Series(0, index=df.index))
    fta = df.get("ft_attempted", pd.Series(0, index=df.index))
    tov = df.get("turnovers", pd.Series(0, index=df.index))
    orb = df.get("off_reb", pd.Series(0, index=df.index))
    minutes = df.get("minutes", pd.Series(40, index=df.index))

    possessions = fga - orb + tov + 0.4 * fta
    df["pace"] = np.where(
        minutes > 0,
        possessions * (48 / minutes),
        np.nan,
    )
    return df


def add_offensive_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``offensive_rating`` (points scored per 100 estimated possessions).

    Requires columns: ``points``, and the possessions estimate produced by
    :func:`add_pace` (column ``pace``).  If ``pace`` is absent it is computed
    on-the-fly.
    """
    if "pace" not in df.columns:
        df = add_pace(df)

    df = df.copy()
    minutes = df.get("minutes", pd.Series(40, index=df.index))
    # Convert pace (per 48 min) back to raw possessions for the actual game
    possessions = df["pace"] * (minutes / 48)
    df["offensive_rating"] = np.where(
        possessions > 0,
        df["points"] / possessions * 100,
        np.nan,
    )
    return df


def add_defensive_rating(
    df: pd.DataFrame,
    opp_pts_col: str = "opp_points",
) -> pd.DataFrame:
    """
    Add ``defensive_rating`` (opponent points per 100 estimated possessions).

    Parameters
    ----------
    df:
        DataFrame containing the team's stats and opponent points.
    opp_pts_col:
        Column name for the opponent's points scored.
    """
    if opp_pts_col not in df.columns:
        logger.warning("Column '%s' not found; defensive_rating will be NaN.", opp_pts_col)
        df = df.copy()
        df["defensive_rating"] = np.nan
        return df

    if "pace" not in df.columns:
        df = add_pace(df)

    df = df.copy()
    minutes = df.get("minutes", pd.Series(40, index=df.index))
    possessions = df["pace"] * (minutes / 48)
    df["defensive_rating"] = np.where(
        possessions > 0,
        df[opp_pts_col] / possessions * 100,
        np.nan,
    )
    return df


def add_net_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``net_rating`` = offensive_rating − defensive_rating.

    Both component columns must already exist (or will be computed).
    """
    if "offensive_rating" not in df.columns:
        df = add_offensive_rating(df)
    if "defensive_rating" not in df.columns:
        df = add_defensive_rating(df)

    df = df.copy()
    df["net_rating"] = df["offensive_rating"] - df["defensive_rating"]
    return df


def build_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature-engineering stack to a cleaned game DataFrame.

    Returns the DataFrame with all advanced metric columns appended.
    """
    df = add_true_shooting(df)
    df = add_pace(df)
    df = add_offensive_rating(df)
    df = add_defensive_rating(df)
    df = add_net_rating(df)
    return df
