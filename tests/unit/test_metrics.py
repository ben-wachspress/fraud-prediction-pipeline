import numpy as np
import pytest

from src.evaluation.metrics import evaluate, find_best_threshold


def test_find_best_threshold():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    threshold = find_best_threshold(y_true, y_proba, metric="f1")
    assert 0.0 < threshold < 1.0


def test_find_best_threshold_invalid_metric():
    with pytest.raises(ValueError):
        find_best_threshold(np.array([0, 1]), np.array([0.3, 0.7]), metric="accuracy")
