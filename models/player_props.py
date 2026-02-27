"""
Player-prop market model.

Detects statistical inefficiencies in points, rebounds, and assists
markets by training a gradient-boosting regressor (XGBoost or LightGBM)
for each target and comparing model predictions against market lines.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

PROP_TARGETS: List[str] = ["points", "rebounds", "assists"]

# Features shared across all three prop targets
BASE_FEATURES: List[str] = [
    "points_avg10", "points_std10",
    "rebounds_avg10", "rebounds_std10",
    "assists_avg10", "assists_std10",
    "true_shooting_pct_avg10",
    "minutes",
    "back_to_back",
]


def _default_regressor():
    """Return the best available gradient-boosting regressor."""
    if _XGB_AVAILABLE:
        return _XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    if _LGB_AVAILABLE:
        return _LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    # Fallback to a simple sklearn estimator so tests can run without optional deps
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )


class PlayerPropsModel:
    """Train one regressor per prop target and detect market inefficiencies.

    Parameters
    ----------
    features:
        Column names to use as input features.  Defaults to
        :data:`BASE_FEATURES`.
    regressor:
        Any scikit-learn-compatible regressor.  When ``None`` the best
        available gradient-boosting library is used automatically.
    cv_folds:
        Number of cross-validation folds used during :meth:`fit`.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        regressor=None,
        cv_folds: int = 5,
    ) -> None:
        self.features = features or BASE_FEATURES
        self._base_regressor = regressor or _default_regressor()
        self.cv_folds = cv_folds
        self._pipelines: Dict[str, Pipeline] = {}
        self.cv_scores_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, targets: pd.DataFrame) -> "PlayerPropsModel":
        """Fit one model per prop target.

        Parameters
        ----------
        X:
            Feature matrix (rows = player-games).
        targets:
            DataFrame whose columns are the prop targets (points, rebounds,
            assists).  Must share the same index as *X*.

        Returns
        -------
        PlayerPropsModel
            ``self`` for method chaining.
        """
        available_features = [f for f in self.features if f in X.columns]
        X_fit = X[available_features].fillna(0)

        for target in PROP_TARGETS:
            if target not in targets.columns:
                logger.warning("Target '%s' not found – skipping.", target)
                continue

            y = targets[target].fillna(0)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", clone(self._base_regressor)),
            ])
            # cross-val score uses negative MAE for regressors
            scores = cross_val_score(
                pipe, X_fit, y,
                cv=self.cv_folds,
                scoring="neg_mean_absolute_error",
            )
            self.cv_scores_[target] = float(-scores.mean())
            pipe.fit(X_fit, y)
            self._pipelines[target] = pipe
            logger.info(
                "Fitted '%s' model | CV MAE=%.3f", target, self.cv_scores_[target]
            )

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return per-target predictions for new player-game rows.

        Parameters
        ----------
        X:
            Feature matrix with the same columns used during :meth:`fit`.

        Returns
        -------
        pd.DataFrame
            Columns are the prop targets; index matches *X*.
        """
        if not self._pipelines:
            raise RuntimeError("Model has not been fitted yet.  Call fit() first.")

        available_features = [f for f in self.features if f in X.columns]
        X_pred = X[available_features].fillna(0)

        preds = {}
        for target, pipe in self._pipelines.items():
            preds[target] = pipe.predict(X_pred)

        return pd.DataFrame(preds, index=X.index)

    def find_inefficiencies(
        self,
        X: pd.DataFrame,
        market_lines: pd.DataFrame,
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Identify player-game rows where the model deviates from market lines.

        A positive ``edge`` value means the model projects a higher value than
        the market line (value *over*); negative means value *under*.

        Parameters
        ----------
        X:
            Feature matrix.
        market_lines:
            DataFrame with columns matching :data:`PROP_TARGETS` containing
            the posted market line for each row.
        threshold:
            Minimum absolute edge (in stat units) to flag as an inefficiency.

        Returns
        -------
        pd.DataFrame
            Rows where at least one prop has ``|edge| >= threshold``.
        """
        predictions = self.predict(X)
        results = []

        for target in PROP_TARGETS:
            if target not in predictions.columns or target not in market_lines.columns:
                continue
            edge = predictions[target] - market_lines[target]
            mask = edge.abs() >= threshold
            flagged = X[mask].copy()
            flagged["target"] = target
            flagged["predicted"] = predictions.loc[mask, target].values
            flagged["market_line"] = market_lines.loc[mask, target].values
            flagged["edge"] = edge[mask].values
            flagged["direction"] = np.where(flagged["edge"] > 0, "over", "under")
            results.append(flagged)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)
