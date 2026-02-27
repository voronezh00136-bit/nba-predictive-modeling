"""
Game probability model.

Forecasts NBA game outcomes by learning from team-level advanced metrics,
back-to-back scheduling flags, and historical home/away dynamics using a
gradient-boosting classifier.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Features expected in the game-level feature matrix
GAME_FEATURES: List[str] = [
    "home",
    "back_to_back",
    "team_points",
    "team_rebounds",
    "team_assists",
    "avg_true_shooting",
    "pace",
    "opp_team_points",
    "opp_team_rebounds",
    "opp_avg_true_shooting",
    "opp_pace",
    "opp_back_to_back",
]


def _default_classifier():
    """Return the best available gradient-boosting classifier."""
    if _XGB_AVAILABLE:
        return _XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    if _LGB_AVAILABLE:
        return _LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )


class GameProbabilityModel:
    """Predict win probabilities for NBA games.

    The raw classifier is wrapped in
    :class:`~sklearn.calibration.CalibratedClassifierCV` (isotonic
    regression) so that the output probabilities are well-calibrated and
    can be directly interpreted as win likelihoods.

    Parameters
    ----------
    features:
        Column names to use as input features.  Defaults to
        :data:`GAME_FEATURES`.
    classifier:
        Any scikit-learn-compatible binary classifier.  When ``None`` the
        best available gradient-boosting library is used.
    cv_folds:
        Number of cross-validation folds.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        classifier=None,
        cv_folds: int = 5,
    ) -> None:
        self.features = features or GAME_FEATURES
        self._base_classifier = classifier or _default_classifier()
        self.cv_folds = cv_folds
        self._pipeline: Optional[Pipeline] = None
        self.cv_roc_auc_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GameProbabilityModel":
        """Fit the calibrated game-probability pipeline.

        Parameters
        ----------
        X:
            Feature matrix (rows = team-game matchups).
        y:
            Binary win/loss label (1 = team won, 0 = team lost).

        Returns
        -------
        GameProbabilityModel
            ``self`` for method chaining.
        """
        available_features = [f for f in self.features if f in X.columns]
        X_fit = X[available_features].fillna(0).astype(float)
        y_fit = y.astype(int)

        base_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", clone(self._base_classifier)),
        ])

        # Cross-validate with ROC-AUC before fitting on the full dataset
        scores = cross_val_score(
            base_pipe, X_fit, y_fit,
            cv=self.cv_folds,
            scoring="roc_auc",
        )
        self.cv_roc_auc_ = float(scores.mean())
        logger.info("Game-prob CV ROC-AUC=%.4f", self.cv_roc_auc_)

        # Calibrate on the full training set
        calibrated = CalibratedClassifierCV(base_pipe, method="isotonic", cv=3)
        calibrated.fit(X_fit, y_fit)
        self._pipeline = calibrated
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return win-probability estimates for each row in *X*.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Columns are [P(loss), P(win)].
        """
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet.  Call fit() first.")

        available_features = [f for f in self.features if f in X.columns]
        X_pred = X[available_features].fillna(0).astype(float)
        return self._pipeline.predict_proba(X_pred)

    def predict_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Return the scalar win probability P(win) for each row.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        return self.predict_proba(X)[:, 1]
