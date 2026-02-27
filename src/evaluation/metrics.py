"""
Evaluation metrics for probabilistic models.

Wraps scikit-learn's scoring functions and provides a single convenience
function that reports all three key metrics at once:

* **Log Loss**   — penalises confident wrong predictions heavily.
* **Brier Score**— mean squared error between predicted probabilities and outcomes.
* **ROC-AUC**    — area under the receiver operating characteristic curve.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)

ArrayLike = Union[list, np.ndarray, pd.Series]


def compute_log_loss(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """
    Compute binary cross-entropy (log loss).

    Lower is better; a perfect model scores 0.

    Parameters
    ----------
    y_true:
        True binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Log loss value.
    """
    return float(log_loss(y_true, y_prob))


def compute_brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """
    Compute the Brier score (mean squared probability error).

    Lower is better; a perfect model scores 0.

    Parameters
    ----------
    y_true:
        True binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Brier score.
    """
    return float(brier_score_loss(y_true, y_prob))


def compute_roc_auc(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """
    Compute ROC-AUC.

    Higher is better; random chance = 0.5, perfect model = 1.0.

    Parameters
    ----------
    y_true:
        True binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        ROC-AUC score.
    """
    return float(roc_auc_score(y_true, y_prob))


def evaluate_model(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    model_name: str = "model",
) -> dict[str, float]:
    """
    Compute and log all three evaluation metrics.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    model_name:
        Label used in log output.

    Returns
    -------
    dict
        ``{"log_loss": ..., "brier_score": ..., "roc_auc": ...}``
    """
    metrics = {
        "log_loss": compute_log_loss(y_true, y_prob),
        "brier_score": compute_brier_score(y_true, y_prob),
        "roc_auc": compute_roc_auc(y_true, y_prob),
    }
    logger.info(
        "[%s] log_loss=%.4f | brier_score=%.4f | roc_auc=%.4f",
        model_name,
        metrics["log_loss"],
        metrics["brier_score"],
        metrics["roc_auc"],
    )
    return metrics
