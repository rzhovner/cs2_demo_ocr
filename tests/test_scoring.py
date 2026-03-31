"""Unit tests for scoring/matching logic used across detectors.

Tests use synthetic data with known TP/FP/FN outcomes to verify that
matching algorithms produce correct results regardless of implementation
(greedy, cost-matrix, Hungarian).
"""

import numpy as np
import pytest


class TestKillfeedScore:
    """Tests for killfeed_detector.score() — greedy cost-matrix matching."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.killfeed_detector import score
        self.score = score

    def test_perfect_match(self):
        det = np.array([1.0, 2.0, 3.0])
        truth = np.array([1.0, 2.0, 3.0])
        result = self.score(det, truth)
        assert result["matched"] == 3
        assert result["recall"] == 100.0
        assert result["precision"] == 100.0

    def test_all_false_positives(self):
        det = np.array([1.0, 2.0, 3.0])
        truth = np.array([10.0, 20.0])
        result = self.score(det, truth)
        assert result["matched"] == 0
        assert len(result["false_positives"]) == 3
        assert len(result["missed"]) == 2

    def test_all_missed(self):
        det = np.array([])
        truth = np.array([1.0, 2.0])
        result = self.score(det, truth)
        assert result["matched"] == 0
        assert result["recall"] == 0

    def test_within_tolerance(self):
        det = np.array([1.05, 2.1])
        truth = np.array([1.0, 2.0])
        result = self.score(det, truth, tolerance_s=0.15)
        assert result["matched"] == 2

    def test_outside_tolerance(self):
        det = np.array([1.5])
        truth = np.array([1.0])
        result = self.score(det, truth, tolerance_s=0.1)
        assert result["matched"] == 0

    def test_no_greedy_stealing(self):
        """Closest-pair matching should not let an earlier detection steal
        a truth value from a closer later detection."""
        det = np.array([1.0, 1.3])
        truth = np.array([1.2])
        result = self.score(det, truth, tolerance_s=0.5)
        assert result["matched"] == 1
        # 1.3 is closer to 1.2 than 1.0 is — should match 1.3, not 1.0
        assert len(result["false_positives"]) == 1


class TestAudioOptimalMatch:
    """Tests for audio_kill_detector.optimal_match() — Hungarian algorithm."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.audio_kill_detector import optimal_match
        self.optimal_match = optimal_match

    def test_perfect_match(self):
        det = np.array([1.0, 2.0, 3.0])
        gt = np.array([1.0, 2.0, 3.0])
        tp, fp, fn, matched = self.optimal_match(det, gt, tolerance=0.5)
        assert tp == 3
        assert fp == 0
        assert fn == 0

    def test_empty_detections(self):
        det = np.array([])
        gt = np.array([1.0, 2.0])
        tp, fp, fn, _ = self.optimal_match(det, gt, tolerance=0.5)
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_empty_ground_truth(self):
        det = np.array([1.0, 2.0])
        gt = np.array([])
        tp, fp, fn, _ = self.optimal_match(det, gt, tolerance=0.5)
        assert tp == 0
        assert fp == 2
        assert fn == 0

    def test_partial_match(self):
        det = np.array([1.0, 5.0, 10.0])
        gt = np.array([1.1, 5.2])
        tp, fp, fn, _ = self.optimal_match(det, gt, tolerance=0.5)
        assert tp == 2
        assert fp == 1
        assert fn == 0

    def test_tolerance_boundary(self):
        det = np.array([1.0])
        gt = np.array([1.5])
        tp_in, _, _, _ = self.optimal_match(det, gt, tolerance=0.5)
        assert tp_in == 1
        tp_out, _, _, _ = self.optimal_match(det, gt, tolerance=0.4)
        assert tp_out == 0
