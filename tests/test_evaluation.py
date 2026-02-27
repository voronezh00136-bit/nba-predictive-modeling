"""
Tests for src/evaluation/metrics.py
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_log_loss,
    compute_brier_score,
    compute_roc_auc,
    evaluate_model,
)


Y_TRUE = [1, 0, 1, 1, 0, 0, 1, 0]
Y_PROB_GOOD = [0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15]
Y_PROB_BAD = [0.1, 0.9, 0.2, 0.3, 0.8, 0.7, 0.15, 0.85]


class TestLogLoss:
    def test_good_predictions_lower(self):
        good = compute_log_loss(Y_TRUE, Y_PROB_GOOD)
        bad = compute_log_loss(Y_TRUE, Y_PROB_BAD)
        assert good < bad

    def test_perfect_predictions_near_zero(self):
        y_true = [0, 1]
        y_prob = [0.001, 0.999]
        assert compute_log_loss(y_true, y_prob) < 0.02

    def test_returns_float(self):
        assert isinstance(compute_log_loss(Y_TRUE, Y_PROB_GOOD), float)


class TestBrierScore:
    def test_good_predictions_lower(self):
        good = compute_brier_score(Y_TRUE, Y_PROB_GOOD)
        bad = compute_brier_score(Y_TRUE, Y_PROB_BAD)
        assert good < bad

    def test_range(self):
        score = compute_brier_score(Y_TRUE, Y_PROB_GOOD)
        assert 0.0 <= score <= 1.0


class TestRocAuc:
    def test_good_predictions_higher(self):
        good = compute_roc_auc(Y_TRUE, Y_PROB_GOOD)
        bad = compute_roc_auc(Y_TRUE, Y_PROB_BAD)
        assert good > bad

    def test_perfect_model(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        assert compute_roc_auc(y_true, y_prob) == pytest.approx(1.0)

    def test_range(self):
        score = compute_roc_auc(Y_TRUE, Y_PROB_GOOD)
        assert 0.0 <= score <= 1.0


class TestEvaluateModel:
    def test_returns_all_metrics(self):
        result = evaluate_model(Y_TRUE, Y_PROB_GOOD, model_name="test")
        assert set(result.keys()) == {"log_loss", "brier_score", "roc_auc"}

    def test_metric_values_sensible(self):
        result = evaluate_model(Y_TRUE, Y_PROB_GOOD)
        assert result["log_loss"] > 0
        assert 0 < result["brier_score"] < 1
        assert result["roc_auc"] > 0.5
