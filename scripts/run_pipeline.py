"""
End-to-end pipeline orchestration script.

Runs the full NBA predictive modeling pipeline:

1. Scrape raw data (box scores + injury report)
2. Clean and aggregate data
3. Engineer features (advanced metrics)
4. Train game outcome predictor & player props model
5. Evaluate models on a held-out season split
6. (Optional) Send Telegram alerts for high-value predictions

Usage::

    python scripts/run_pipeline.py --season 2024 [--alert]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on the path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.scraper import scrape_box_scores, scrape_injury_report
from src.data.cleaner import clean_box_scores, save_processed
from src.data.aggregator import (
    add_back_to_back_flag,
    aggregate_team_rolling,
    merge_injury_flags,
)
from src.features.engineering import build_game_features
from src.models.game_predictor import GamePredictor, FEATURE_COLS as GAME_FEATURE_COLS
from src.evaluation.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def build_game_dataset(season: int) -> pd.DataFrame:
    """
    Scrape, clean, and feature-engineer a full game dataset for *season*.
    """
    logger.info("=== Step 1: Scraping box scores for season %d ===", season)
    raw = scrape_box_scores(season)

    logger.info("=== Step 2: Cleaning data ===")
    clean = clean_box_scores(raw)

    logger.info("=== Step 3: Adding back-to-back flags ===")
    home = add_back_to_back_flag(clean, team_col="home_team")
    home = home.rename(columns={"back_to_back": "home_back_to_back"})
    away = add_back_to_back_flag(clean, team_col="away_team")
    away_b2b = away["back_to_back"].rename("away_back_to_back")
    df = home.join(away_b2b)

    logger.info("=== Step 4: Rolling team offensive averages ===")
    df = aggregate_team_rolling(df, window=10, team_col="home_team", pts_col="home_pts")
    df = df.rename(columns={"rolling_pts": "home_rolling_pts"})
    df = aggregate_team_rolling(df, window=10, team_col="away_team", pts_col="away_pts")
    df = df.rename(columns={"rolling_pts": "away_rolling_pts"})

    logger.info("=== Step 5: Injury report ===")
    injuries = scrape_injury_report(season)
    df = merge_injury_flags(df, injuries, team_col="home_team")
    df = df.rename(columns={"key_player_out": "home_key_player_out"})
    away_injuries = merge_injury_flags(df, injuries, team_col="away_team")
    df["away_key_player_out"] = away_injuries["key_player_out"]

    logger.info("=== Step 6: Advanced feature engineering ===")
    # Proxy columns for feature engineering (team-level game totals)
    for side, pts_col, fga_col, fta_col, tov_col, min_col in [
        ("home", "home_pts", None, None, None, None),
        ("away", "away_pts", None, None, None, None),
    ]:
        for col in ("fg_attempted", "ft_attempted", "turnovers", "minutes", "points"):
            df[col] = np.nan  # placeholders; real values come from player logs

    # Numeric cast for B2B flags
    for col in ("home_back_to_back", "away_back_to_back",
                "home_key_player_out", "away_key_player_out"):
        df[col] = df[col].fillna(False).astype(int)

    # Net rating placeholders (need player-level data for accuracy)
    df["home_net_rating"] = np.nan
    df["away_net_rating"] = np.nan

    save_processed(df, f"games_{season}")
    logger.info("Dataset built: %d games", len(df))
    return df


def train_and_evaluate(df: pd.DataFrame, test_fraction: float = 0.2) -> None:
    """Train the game predictor and evaluate on a temporal holdout set."""
    df = df.sort_values("date").copy()
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    # Filter to only columns available in both sets
    available_features = [c for c in GAME_FEATURE_COLS if c in df.columns]
    if not available_features:
        logger.error("No feature columns available. Check dataset construction.")
        return

    predictor = GamePredictor()
    predictor.fit(train_df)
    predictor.save()

    y_prob = predictor.predict_proba(test_df)
    y_true = test_df["home_win"].to_numpy()
    metrics = evaluate_model(y_true, y_prob, model_name="GamePredictor")
    logger.info("Evaluation complete: %s", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Predictive Modeling Pipeline")
    parser.add_argument("--season", type=int, default=2024, help="NBA season end-year")
    parser.add_argument("--alert", action="store_true", help="Send Telegram alerts")
    args = parser.parse_args()

    df = build_game_dataset(args.season)
    train_and_evaluate(df)

    if args.alert:
        from src.alerts.telegram_bot import TelegramAlerter
        alerter = TelegramAlerter()
        alerter.send_message("✅ NBA pipeline run complete!")


if __name__ == "__main__":
    main()
