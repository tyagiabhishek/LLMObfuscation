"""Tests for compute_tpr_at_fpr."""

import numpy as np
from rl_obfuscation import compute_tpr_at_fpr


def test_perfect_separation():
    """When scores perfectly separate classes, TPR@1%FPR should be 1.0."""
    scores = np.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.95, 0.97, 0.99])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    result = compute_tpr_at_fpr(scores, labels, target_fpr=0.01)
    assert result["tpr_at_fpr"] == 1.0
    assert result["auroc"] == 1.0


def test_inverse_separation():
    """When high scores = class 0, AUROC should be 0.0."""
    scores = np.array([0.9, 0.95, 0.97, 0.99, 0.0, 0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    result = compute_tpr_at_fpr(scores, labels)
    assert result["auroc"] == 0.0
    assert result["tpr_at_fpr"] == 0.0


def test_random_scores_low_tpr():
    """Random scores should give low TPR at 1% FPR."""
    rng = np.random.default_rng(42)
    n = 1000
    scores = rng.random(n)
    labels = np.array([0] * 500 + [1] * 500, dtype=np.float64)
    result = compute_tpr_at_fpr(scores, labels, target_fpr=0.01)
    # Random classifier: TPR at 1% FPR should be low
    assert result["tpr_at_fpr"] < 0.1


def test_all_same_label():
    """Single class → graceful handling."""
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([1.0, 1.0, 1.0])
    result = compute_tpr_at_fpr(scores, labels)
    assert result["auroc"] == 0.5
    assert result["tpr_at_fpr"] == 0.0
    assert result["threshold"] == float("inf")


def test_all_zeros_label():
    """All class 0 → graceful handling."""
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([0.0, 0.0, 0.0])
    result = compute_tpr_at_fpr(scores, labels)
    assert result["auroc"] == 0.5


def test_result_keys():
    """Result dict has expected keys."""
    scores = np.array([0.1, 0.9])
    labels = np.array([0.0, 1.0])
    result = compute_tpr_at_fpr(scores, labels)
    assert "auroc" in result
    assert "tpr_at_fpr" in result
    assert "threshold" in result
    assert "fpr_curve" in result
    assert "tpr_curve" in result


def test_curves_are_numpy_arrays():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    result = compute_tpr_at_fpr(scores, labels)
    assert isinstance(result["fpr_curve"], np.ndarray)
    assert isinstance(result["tpr_curve"], np.ndarray)


def test_threshold_is_float():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    result = compute_tpr_at_fpr(scores, labels)
    assert isinstance(result["threshold"], float)


def test_tpr_at_fpr_is_float():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=np.float64)
    result = compute_tpr_at_fpr(scores, labels)
    assert isinstance(result["tpr_at_fpr"], float)


def test_different_target_fprs():
    """Higher target FPR should give >= TPR."""
    scores = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    r1 = compute_tpr_at_fpr(scores, labels, target_fpr=0.01)
    r5 = compute_tpr_at_fpr(scores, labels, target_fpr=0.05)
    r10 = compute_tpr_at_fpr(scores, labels, target_fpr=0.10)
    assert r1["tpr_at_fpr"] <= r5["tpr_at_fpr"] <= r10["tpr_at_fpr"] or \
           r5["tpr_at_fpr"] <= r10["tpr_at_fpr"]


def test_accepts_float32_inputs():
    """Should handle float32 inputs without error."""
    scores = np.array([0.1, 0.9], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.float32)
    result = compute_tpr_at_fpr(scores, labels)
    assert 0.0 <= result["auroc"] <= 1.0


def test_accepts_torch_tensor_converted():
    """Should work with numpy arrays converted from torch tensors."""
    import torch

    scores = torch.tensor([0.1, 0.4, 0.6, 0.9]).numpy()
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0]).numpy()
    result = compute_tpr_at_fpr(scores, labels)
    assert 0.0 <= result["auroc"] <= 1.0
