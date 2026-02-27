"""
Data cleaning utilities.

Converts raw scraped data into well-typed, analysis-ready DataFrames stored
in ``data/processed/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_box_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw box-score data.

    * Converts score columns to numeric, dropping unparseable rows.
    * Parses the *date* column to ``datetime``.
    * Derives a binary *home_win* target column.

    Parameters
    ----------
    df:
        Raw DataFrame produced by :func:`~src.data.scraper.scrape_box_scores`.

    Returns
    -------
    pd.DataFrame
        Cleaned box-score DataFrame.
    """
    df = df.copy()

    # Coerce score columns
    for col in ("away_pts", "home_pts"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["away_pts", "home_pts"], inplace=True)
    df["away_pts"] = df["away_pts"].astype(int)
    df["home_pts"] = df["home_pts"].astype(int)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # Target: 1 if home team wins
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)

    # Strip whitespace from team names
    for col in ("home_team", "away_team"):
        df[col] = df[col].str.strip()

    df.reset_index(drop=True, inplace=True)
    return df


def clean_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw player game-log data.

    * Removes non-game rows (e.g., "Did Not Play", header repeats).
    * Converts stat columns to numeric.
    * Renames Basketball-Reference column codes to human-readable names.

    Parameters
    ----------
    df:
        Raw DataFrame produced by
        :func:`~src.data.scraper.scrape_player_game_logs`.

    Returns
    -------
    pd.DataFrame
        Cleaned player log DataFrame.
    """
    df = df.copy()

    # Drop header-repeat rows embedded in the table body
    if "reason" in df.columns:
        df = df[df["reason"] != "Did Not Play"]
    if "game_season" in df.columns:
        df = df[df["game_season"] != "Rk"]

    rename_map = {
        "date_game": "date",
        "pts": "points",
        "trb": "rebounds",
        "ast": "assists",
        "fg": "fg_made",
        "fga": "fg_attempted",
        "fg3": "fg3_made",
        "fg3a": "fg3_attempted",
        "ft": "ft_made",
        "fta": "ft_attempted",
        "mp": "minutes",
        "stl": "steals",
        "blk": "blocks",
        "tov": "turnovers",
        "plus_minus": "plus_minus",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    numeric_cols = [
        "points", "rebounds", "assists", "fg_made", "fg_attempted",
        "fg3_made", "fg3_attempted", "ft_made", "ft_attempted",
        "minutes", "steals", "blocks", "turnovers", "plus_minus",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.dropna(subset=["points"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """
    Persist a cleaned DataFrame to ``data/processed/<name>.csv``.

    Returns the path of the saved file.
    """
    out_path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved cleaned data to %s (%d rows)", out_path, len(df))
    return out_path


def load_processed(name: str) -> pd.DataFrame:
    """Load a previously saved cleaned DataFrame from ``data/processed/``."""
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
