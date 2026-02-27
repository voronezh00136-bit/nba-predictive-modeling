"""
Evaluation metrics for the NBA predictive-modeling pipeline.

Provides convenience wrappers around Log Loss, Brier Score, and ROC-AUC,
plus a combined ``evaluate_predictions`` summary helper.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the binary log loss.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Log-loss value (lower is better).
    """
    return float(log_loss(y_true, y_prob))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Brier score (lower is better, 0 = perfect).
    """
    return float(brier_score_loss(y_true, y_prob))


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the ROC-AUC score.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        ROC-AUC (higher is better, 1.0 = perfect, 0.5 = random).
    """
    return float(roc_auc_score(y_true, y_prob))


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute all three evaluation metrics in a single call.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Keys: ``log_loss``, ``brier_score``, ``roc_auc``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    return {
        "log_loss": compute_log_loss(y_true, y_prob),
        "brier_score": compute_brier_score(y_true, y_prob),
        "roc_auc": compute_roc_auc(y_true, y_prob),
    }
