"""
Data aggregation utilities.

Combines per-game box scores and player logs into team-level rolling
features, and detects back-to-back scheduling.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_back_to_back_flag(df: pd.DataFrame, team_col: str = "home_team") -> pd.DataFrame:
    """
    Add a boolean column ``back_to_back`` that is *True* when a team played
    a game the previous calendar day.

    The function works on a DataFrame that contains one row **per team per
    game** (long format).  It expects a ``date`` column of type
    ``datetime64``.

    Parameters
    ----------
    df:
        Game-level DataFrame with at least ``date`` and *team_col* columns.
    team_col:
        Name of the column that identifies the team.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``back_to_back`` boolean column.
    """
    df = df.copy().sort_values(["date"])
    df["prev_game_date"] = df.groupby(team_col)["date"].shift(1)
    df["back_to_back"] = (df["date"] - df["prev_game_date"]).dt.days == 1
    df.drop(columns=["prev_game_date"], inplace=True)
    return df


def aggregate_team_rolling(
    df: pd.DataFrame,
    window: int = 10,
    team_col: str = "home_team",
    pts_col: str = "home_pts",
) -> pd.DataFrame:
    """
    Compute a *window*-game rolling mean of team offensive output.

    Parameters
    ----------
    df:
        Cleaned box-score DataFrame.
    window:
        Rolling window size (number of games).
    team_col:
        Column identifying the team.
    pts_col:
        Column with the team's points scored each game.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``rolling_pts`` column.
    """
    df = df.copy().sort_values(["date"])
    df["rolling_pts"] = (
        df.groupby(team_col)[pts_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    return df


def aggregate_player_rolling(
    df: pd.DataFrame,
    stat_cols: list[str] | None = None,
    window: int = 5,
    player_col: str = "player_slug",
) -> pd.DataFrame:
    """
    Compute rolling averages for player stats over the last *window* games.

    Parameters
    ----------
    df:
        Cleaned player game-log DataFrame.
    stat_cols:
        Columns to roll.  Defaults to ``["points", "rebounds", "assists"]``.
    window:
        Rolling window size.
    player_col:
        Column identifying the player.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``rolling_<stat>`` columns.
    """
    if stat_cols is None:
        stat_cols = ["points", "rebounds", "assists"]

    df = df.copy().sort_values(["date"])
    for col in stat_cols:
        if col not in df.columns:
            continue
        df[f"rolling_{col}"] = (
            df.groupby(player_col)[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df


def merge_injury_flags(
    game_df: pd.DataFrame,
    injury_df: pd.DataFrame,
    team_col: str = "home_team",
) -> pd.DataFrame:
    """
    Add a column ``key_player_out`` (bool) that flags games where a player
    with ``"Out"`` status appears in the injury report for the given team.

    Parameters
    ----------
    game_df:
        Box-score DataFrame with a *team_col* column.
    injury_df:
        Injury-report DataFrame with ``team`` and ``status`` columns.

    Returns
    -------
    pd.DataFrame
        game_df with an additional ``key_player_out`` boolean column.
    """
    if injury_df.empty:
        game_df = game_df.copy()
        game_df["key_player_out"] = False
        return game_df

    teams_with_out = set(
        injury_df.loc[
            injury_df["status"].str.upper() == "OUT", "team"
        ].unique()
    )
    game_df = game_df.copy()
    game_df["key_player_out"] = game_df[team_col].isin(teams_with_out)
    return game_df
