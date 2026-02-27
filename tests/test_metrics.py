"""Tests for evaluation/metrics.py"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    compute_log_loss,
    compute_brier_score,
    compute_roc_auc,
    evaluate_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Y_TRUE = np.array([1, 0, 1, 1, 0, 0, 1, 0])
_Y_PROB = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15])


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def test_log_loss_perfect():
    y_true = np.array([1, 0, 1])
    y_prob = np.array([1 - 1e-7, 1e-7, 1 - 1e-7])
    assert compute_log_loss(y_true, y_prob) < 0.001


def test_log_loss_returns_float():
    assert isinstance(compute_log_loss(_Y_TRUE, _Y_PROB), float)


def test_brier_score_perfect():
    y_true = np.array([1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0])
    assert compute_brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_returns_float():
    assert isinstance(compute_brier_score(_Y_TRUE, _Y_PROB), float)


def test_roc_auc_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_roc_auc(y_true, y_prob) == pytest.approx(1.0)


def test_roc_auc_returns_float():
    assert isinstance(compute_roc_auc(_Y_TRUE, _Y_PROB), float)


# ---------------------------------------------------------------------------
# evaluate_predictions (combined)
# ---------------------------------------------------------------------------

def test_evaluate_predictions_returns_all_keys():
    result = evaluate_predictions(_Y_TRUE, _Y_PROB)
    assert set(result.keys()) == {"log_loss", "brier_score", "roc_auc"}


def test_evaluate_predictions_values_are_float():
    result = evaluate_predictions(_Y_TRUE, _Y_PROB)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float"


def test_evaluate_predictions_roc_auc_good_model():
    result = evaluate_predictions(_Y_TRUE, _Y_PROB)
    assert result["roc_auc"] >= 0.8


def test_evaluate_predictions_accepts_lists():
    result = evaluate_predictions(
        list(_Y_TRUE), list(_Y_PROB)
    )
    assert "log_loss" in result
