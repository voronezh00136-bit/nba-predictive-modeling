"""
Player prop market model.

Predicts whether a player will **exceed** a given line (points, rebounds, or
assists) in the next game.  The output probability represents the model's
estimate of the "over" hitting.

Statistical inefficiency detection
-----------------------------------
After generating predictions the model compares implied model probability
against the market-implied probability derived from American or decimal odds.
A positive **edge** (model_prob − market_implied_prob) indicates a value bet.

Feature expectations
---------------------
Required columns in the training / inference DataFrame:

* ``rolling_points``    — player rolling avg points (last N games)
* ``rolling_rebounds``  — player rolling avg rebounds
* ``rolling_assists``   — player rolling avg assists
* ``true_shooting_pct`` — player True Shooting % (rolling)
* ``minutes``           — player average minutes
* ``back_to_back``      — 1 if player's team plays back-to-back
* ``home_game``         — 1 if the game is at home

The target column depends on the stat being predicted:
``over_points``, ``over_rebounds``, or ``over_assists`` (binary).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

Stat = Literal["points", "rebounds", "assists"]

FEATURE_COLS = [
    "rolling_points",
    "rolling_rebounds",
    "rolling_assists",
    "true_shooting_pct",
    "minutes",
    "back_to_back",
    "home_game",
]


def american_odds_to_implied_prob(odds: float) -> float:
    """
    Convert American moneyline odds to implied probability.

    Parameters
    ----------
    odds:
        Positive (e.g. +150) or negative (e.g. -110) American odds.

    Returns
    -------
    float
        Implied probability in [0, 1].
    """
    if odds >= 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def decimal_odds_to_implied_prob(odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Parameters
    ----------
    odds:
        Decimal odds (e.g. 1.91 for -110 American).
    """
    if odds <= 0:
        raise ValueError("Decimal odds must be positive.")
    return 1.0 / odds


def detect_inefficiency(
    model_prob: float,
    market_odds: float,
    odds_format: Literal["american", "decimal"] = "american",
    min_edge: float = 0.03,
) -> dict:
    """
    Compare model probability to market-implied probability.

    Parameters
    ----------
    model_prob:
        Model's probability for the "over".
    market_odds:
        Market odds for the "over" in the given format.
    odds_format:
        ``"american"`` or ``"decimal"``.
    min_edge:
        Minimum edge (model_prob − market_prob) to flag as a value opportunity.

    Returns
    -------
    dict
        ``{"market_implied": ..., "edge": ..., "value": bool}``
    """
    if odds_format == "american":
        market_implied = american_odds_to_implied_prob(market_odds)
    else:
        market_implied = decimal_odds_to_implied_prob(market_odds)

    edge = model_prob - market_implied
    return {
        "market_implied": round(market_implied, 4),
        "edge": round(edge, 4),
        "value": edge >= min_edge,
    }


def _build_estimators() -> dict:
    """Return a dictionary of named estimator objects."""
    estimators: dict = {
        "logistic": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
        ),
    }
    if HAS_XGB:
        estimators["xgboost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    if HAS_LGB:
        estimators["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    return estimators


class PlayerPropsModel:
    """
    Player prop "over/under" classifier.

    Trains one model per stat (points / rebounds / assists) that predicts
    the probability of a player exceeding a given statistical line.
    """

    def __init__(self, stat: Stat = "points") -> None:
        self.stat = stat
        self._target_col = f"over_{stat}"
        self._estimators: dict = _build_estimators()
        self._is_fitted: bool = False
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_over_target(
        df: pd.DataFrame,
        stat: Stat,
        line: float,
    ) -> pd.DataFrame:
        """
        Add a binary target column ``over_<stat>`` to *df*.

        Parameters
        ----------
        df:
            Player log DataFrame with the raw stat column.
        stat:
            One of ``"points"``, ``"rebounds"``, or ``"assists"``.
        line:
            The prop line (e.g. 24.5 for a points over/under of 24.5).
        """
        df = df.copy()
        df[f"over_{stat}"] = (df[stat] > line).astype(int)
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> "PlayerPropsModel":
        """
        Train the player prop model.

        Parameters
        ----------
        df:
            DataFrame containing feature columns and the binary target
            ``over_<stat>``.
        feature_cols:
            Override the default :data:`FEATURE_COLS`.
        """
        cols = feature_cols or FEATURE_COLS
        available = [c for c in cols if c in df.columns]
        if not available:
            raise ValueError("No recognised feature columns found in training DataFrame.")
        if self._target_col not in df.columns:
            raise ValueError(
                f"Target column '{self._target_col}' not found. "
                f"Call PlayerPropsModel.create_over_target() first."
            )

        X = df[available].fillna(0).to_numpy()
        y = df[self._target_col].to_numpy()

        for name, est in self._estimators.items():
            logger.info("Fitting %s for stat=%s...", name, self.stat)
            est.fit(X, y)
            cv_auc = cross_val_score(est, X, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()
            logger.info("%s cross-val ROC-AUC (%s): %.4f", name, self.stat, cv_auc)

        self._feature_cols = available
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return averaged "over" probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict_proba().")

        X = df[self._feature_cols].fillna(0).to_numpy()
        probs = np.stack(
            [est.predict_proba(X)[:, 1] for est in self._estimators.values()],
            axis=1,
        )
        return probs.mean(axis=1)

    def scan_for_value(
        self,
        df: pd.DataFrame,
        lines: pd.Series,
        market_odds: pd.Series,
        odds_format: Literal["american", "decimal"] = "american",
        min_edge: float = 0.03,
    ) -> pd.DataFrame:
        """
        Attach model probabilities and inefficiency flags to *df*.

        Parameters
        ----------
        df:
            Inference DataFrame.
        lines:
            Series of prop lines aligned with *df*'s index.
        market_odds:
            Series of market odds aligned with *df*'s index.
        odds_format:
            ``"american"`` or ``"decimal"``.
        min_edge:
            Minimum edge to flag as value.

        Returns
        -------
        pd.DataFrame
            df with added columns: ``model_prob``, ``market_implied``,
            ``edge``, ``value``.
        """
        model_probs = self.predict_proba(df)
        df = df.copy()
        df["model_prob"] = model_probs

        results = [
            detect_inefficiency(p, o, odds_format, min_edge)
            for p, o in zip(model_probs, market_odds)
        ]
        df["market_implied"] = [r["market_implied"] for r in results]
        df["edge"] = [r["edge"] for r in results]
        df["value"] = [r["value"] for r in results]
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: Optional[str] = None) -> Path:
        """Pickle the fitted model."""
        fname = name or f"player_props_{self.stat}"
        path = MODELS_DIR / f"{fname}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, stat: Stat = "points", name: Optional[str] = None) -> "PlayerPropsModel":
        """Load a pickled model."""
        fname = name or f"player_props_{stat}"
        path = MODELS_DIR / f"{fname}.pkl"
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("Model loaded from %s", path)
        return model
