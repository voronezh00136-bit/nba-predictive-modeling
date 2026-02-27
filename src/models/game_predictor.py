"""
Game outcome probability model.

Trains an ensemble of XGBoost, LightGBM, and a Logistic Regression classifier
to predict the probability that the **home team wins** a given NBA game.

Feature expectations
--------------------
The model expects a DataFrame with at least the following columns when calling
:meth:`GamePredictor.predict_proba`:

* ``home_rolling_pts``   — home team rolling points scored (last N games)
* ``away_rolling_pts``   — away team rolling points scored
* ``home_back_to_back``  — 1 if home team played yesterday else 0
* ``away_back_to_back``  — 1 if away team played yesterday else 0
* ``home_net_rating``    — home team rolling net rating
* ``away_net_rating``    — away team rolling net rating
* ``home_key_player_out``— 1 if a key home player is listed as Out
* ``away_key_player_out``— 1 if a key away player is listed as Out

The target variable is ``home_win`` (1 = home team won, 0 = away team won).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

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

FEATURE_COLS = [
    "home_rolling_pts",
    "away_rolling_pts",
    "home_back_to_back",
    "away_back_to_back",
    "home_net_rating",
    "away_net_rating",
    "home_key_player_out",
    "away_key_player_out",
]
TARGET_COL = "home_win"


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
            max_depth=4,
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


class GamePredictor:
    """
    Ensemble game-outcome predictor.

    Trains multiple classifiers independently and averages their probability
    outputs at inference time.
    """

    def __init__(self) -> None:
        self._estimators: dict = _build_estimators()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_cols: Optional[list[str]] = None) -> "GamePredictor":
        """
        Fit all estimators on *df*.

        Parameters
        ----------
        df:
            Training DataFrame containing feature columns and ``home_win``.
        feature_cols:
            Override the default :data:`FEATURE_COLS`.

        Returns
        -------
        GamePredictor
            Returns ``self`` for method chaining.
        """
        cols = feature_cols or FEATURE_COLS
        available = [c for c in cols if c in df.columns]
        if not available:
            raise ValueError("No recognised feature columns found in training DataFrame.")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")

        X = df[available].fillna(0).to_numpy()
        y = df[TARGET_COL].to_numpy()

        for name, est in self._estimators.items():
            logger.info("Fitting %s...", name)
            est.fit(X, y)
            cv_auc = cross_val_score(est, X, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()
            logger.info("%s cross-val ROC-AUC: %.4f", name, cv_auc)

        self._feature_cols = available
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return average home-win probabilities across all fitted estimators.

        Parameters
        ----------
        df:
            Feature DataFrame (same columns used during training).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability that the home team wins each game.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict_proba().")

        X = df[self._feature_cols].fillna(0).to_numpy()
        probs = np.stack(
            [est.predict_proba(X)[:, 1] for est in self._estimators.values()],
            axis=1,
        )
        return probs.mean(axis=1)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions (1 = home win, 0 = away win).

        Parameters
        ----------
        threshold:
            Decision boundary; default 0.5.
        """
        return (self.predict_proba(df) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "game_predictor") -> Path:
        """Pickle the fitted predictor to ``models/<name>.pkl``."""
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, name: str = "game_predictor") -> "GamePredictor":
        """Load a pickled predictor from ``models/<name>.pkl``."""
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as fh:
            predictor = pickle.load(fh)
        logger.info("Model loaded from %s", path)
        return predictor
