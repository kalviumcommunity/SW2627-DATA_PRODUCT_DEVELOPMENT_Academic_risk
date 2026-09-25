"""Unit tests for Concept #17: NumPy Vectorized Academic Calculations."""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.numpy_calculations import (
    vectorized_attendance_rates,
    vectorized_batch_linear_slopes,
    vectorized_clip_scores,
    vectorized_composite_risk_score,
    vectorized_curve_scores,
    vectorized_linear_slope,
    vectorized_minmax_normalize,
    vectorized_momentum,
    vectorized_percentile_rank,
    vectorized_robust_scale,
    vectorized_score_to_gpa,
    vectorized_weighted_average,
    vectorized_zscore_normalize,
)


class TestScoreTransformations:
    """Test suite for vectorized score transformations and curving."""

    def test_sqrt_curving(self):
        # 10 * sqrt(score)
        scores = np.array([0.0, 49.0, 64.0, 100.0, np.nan])
        curved = vectorized_curve_scores(scores, method="sqrt", max_score=100.0)

        assert curved[0] == 0.0
        assert curved[1] == 70.0
        assert curved[2] == 80.0
        assert curved[3] == 100.0
        assert np.isnan(curved[4])

    def test_linear_curving(self):
        scores = np.array([70.0, 85.0, 95.0, np.nan])
        curved = vectorized_curve_scores(scores, method="linear", offset=10.0, max_score=100.0)

        assert curved[0] == 80.0
        assert curved[1] == 95.0
        assert curved[2] == 100.0  # Capped at max_score
        assert np.isnan(curved[3])

    def test_max_scale_curving(self):
        scores = np.array([40.0, 80.0, np.nan])
        curved = vectorized_curve_scores(scores, method="max_scale", max_score=100.0)

        # max observed is 80 -> factor is 100/80 = 1.25 -> 40*1.25 = 50.0, 80*1.25 = 100.0
        assert curved[0] == 50.0
        assert curved[1] == 100.0
        assert np.isnan(curved[2])

    def test_curving_invalid_method(self):
        with pytest.raises(DataValidationError, match="Unsupported curving method"):
            vectorized_curve_scores(np.array([50.0, 80.0]), method="magic_curve")

    def test_curving_all_nans(self):
        scores = np.array([np.nan, np.nan])
        curved = vectorized_curve_scores(scores, method="sqrt")
        assert np.all(np.isnan(curved))

    def test_score_to_gpa_collegiate_scale(self):
        scores = np.array([96.0, 91.0, 88.0, 84.0, 81.0, 78.0, 74.0, 71.0, 68.0, 62.0, 55.0, np.nan])
        gpas = vectorized_score_to_gpa(scores)

        assert gpas[0] == 4.0
        assert gpas[1] == 3.7
        assert gpas[2] == 3.3
        assert gpas[3] == 3.0
        assert gpas[4] == 2.7
        assert gpas[5] == 2.3
        assert gpas[6] == 2.0
        assert gpas[7] == 1.7
        assert gpas[8] == 1.3
        assert gpas[9] == 1.0
        assert gpas[10] == 0.0
        assert np.isnan(gpas[11])

    def test_score_to_gpa_custom_scale(self):
        scores = np.array([95.0, 60.0])
        # On scale 10.0: 4.0 -> 10.0, 1.0 -> 2.5
        gpas = vectorized_score_to_gpa(scores, scale=10.0)
        assert gpas[0] == 10.0
        assert gpas[1] == 2.5

    def test_clip_scores(self):
        scores = np.array([-15.0, 0.0, 75.5, 105.0, np.nan])
        clipped = vectorized_clip_scores(scores, lower=0.0, upper=100.0)
        assert clipped[0] == 0.0
        assert clipped[1] == 0.0
        assert clipped[2] == 75.5
        assert clipped[3] == 100.0
        assert np.isnan(clipped[4])


class TestTrendCalculations:
    """Test suite for vectorized linear trend trajectories and batch slope operations."""

    def test_linear_slope_increasing(self):
        values = np.array([50.0, 60.0, 70.0, 80.0])
        slope = vectorized_linear_slope(values)
        assert slope == 10.0

    def test_linear_slope_decreasing(self):
        values = np.array([90.0, 80.0, 70.0])
        slope = vectorized_linear_slope(values)
        assert slope == -10.0

    def test_linear_slope_flat(self):
        values = np.array([75.0, 75.0, 75.0])
        slope = vectorized_linear_slope(values)
        assert slope == 0.0

    def test_linear_slope_with_custom_time_points(self):
        values = np.array([10.0, 30.0])
        time_points = np.array([2.0, 6.0])  # diff y = 20, diff t = 4 => slope = 5.0
        slope = vectorized_linear_slope(values, time_points=time_points)
        assert slope == 5.0

    def test_linear_slope_with_nan(self):
        values = np.array([50.0, np.nan, 70.0])
        # With default t=[0, 1, 2], valid are t=0 (50) and t=2 (70) => slope = 20/2 = 10.0
        slope = vectorized_linear_slope(values)
        assert slope == 10.0

    def test_linear_slope_insufficient_points(self):
        assert np.isnan(vectorized_linear_slope(np.array([50.0])))
        assert np.isnan(vectorized_linear_slope(np.array([np.nan, 50.0])))

    def test_linear_slope_zero_variance_in_time(self):
        values = np.array([50.0, 60.0])
        time_points = np.array([1.0, 1.0])
        assert vectorized_linear_slope(values, time_points=time_points) == 0.0

    def test_batch_linear_slopes(self):
        # 3 students across 4 time steps
        matrix = np.array([
            [50.0, 60.0, 70.0, 80.0],   # Student 1: slope +10.0
            [90.0, 85.0, 80.0, 75.0],   # Student 2: slope -5.0
            [60.0, 60.0, 60.0, 60.0],   # Student 3: slope 0.0
        ])
        slopes = vectorized_batch_linear_slopes(matrix)
        assert len(slopes) == 3
        assert slopes[0] == 10.0
        assert slopes[1] == -5.0
        assert slopes[2] == 0.0

    def test_batch_linear_slopes_with_nan(self):
        matrix = np.array([
            [50.0, np.nan, 70.0, 80.0],
            [90.0, 80.0, 70.0, 60.0],
        ])
        slopes = vectorized_batch_linear_slopes(matrix)
        assert len(slopes) == 2
        assert not np.isnan(slopes[0])
        assert slopes[1] == -10.0

    def test_batch_linear_slopes_invalid_ndim(self):
        with pytest.raises(DataValidationError, match="Expected 2D matrix"):
            vectorized_batch_linear_slopes(np.array([1.0, 2.0, 3.0]))

    def test_batch_linear_slopes_single_time_point(self):
        matrix = np.array([[50.0], [60.0]])
        slopes = vectorized_batch_linear_slopes(matrix)
        assert np.all(np.isnan(slopes))

    def test_vectorized_momentum(self):
        early = np.array([60.0, 80.0, np.nan])
        recent = np.array([75.0, 65.0, 70.0])
        momentum = vectorized_momentum(early, recent)
        assert momentum[0] == 15.0
        assert momentum[1] == -15.0
        assert np.isnan(momentum[2])


class TestNumericalNormalization:
    """Test suite for vectorized normalization (Z-score, Min-Max, Robust, Percentile)."""

    def test_zscore_normalize(self):
        data = np.array([10.0, 20.0, 30.0, np.nan])
        # mean = 20.0, std (ddof=1) = 10.0 -> [-1.0, 0.0, 1.0]
        norm = vectorized_zscore_normalize(data)
        assert norm[0] == -1.0
        assert norm[1] == 0.0
        assert norm[2] == 1.0
        assert np.isnan(norm[3])

    def test_zscore_zero_variance(self):
        data = np.array([50.0, 50.0, 50.0])
        norm = vectorized_zscore_normalize(data)
        np.testing.assert_array_equal(norm, np.array([0.0, 0.0, 0.0]))

    def test_minmax_normalize(self):
        data = np.array([0.0, 50.0, 100.0, np.nan])
        norm = vectorized_minmax_normalize(data, feature_range=(0.0, 1.0))
        assert norm[0] == 0.0
        assert norm[1] == 0.5
        assert norm[2] == 1.0
        assert np.isnan(norm[3])

    def test_minmax_custom_range(self):
        data = np.array([10.0, 20.0, 30.0])
        norm = vectorized_minmax_normalize(data, feature_range=(-1.0, 1.0))
        assert norm[0] == -1.0
        assert norm[1] == 0.0
        assert norm[2] == 1.0

    def test_minmax_constant_values(self):
        data = np.array([42.0, 42.0])
        norm = vectorized_minmax_normalize(data, feature_range=(0.0, 1.0))
        np.testing.assert_array_equal(norm, np.array([0.0, 0.0]))

    def test_robust_scale(self):
        # Median = 30.0, Q1 = 20.0, Q3 = 40.0 => IQR = 20.0
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, np.nan])
        scaled = vectorized_robust_scale(data)
        assert scaled[2] == 0.0  # Median is 30 -> 0.0
        assert scaled[1] == -0.5 # (20 - 30)/20 = -0.5
        assert scaled[3] == 0.5  # (40 - 30)/20 = 0.5
        assert np.isnan(scaled[5])

    def test_percentile_rank(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, np.nan])
        pct = vectorized_percentile_rank(data)
        assert pct[0] == 25.0
        assert pct[1] == 50.0
        assert pct[2] == 75.0
        assert pct[3] == 100.0
        assert np.isnan(pct[4])


class TestMetricCalculations:
    """Test suite for academic metric calculations and composite risk indices."""

    def test_attendance_rates(self):
        present = np.array([8.0, 5.0, 0.0])
        late = np.array([2.0, 0.0, 0.0])
        excused = np.array([1.0, 0.0, 10.0])
        total = np.array([10.0, 10.0, 10.0])

        rates = vectorized_attendance_rates(present, late, excused, total, late_weight=0.5)

        # Student 1: (8 + 0.5*2) / (10 - 1) = 9/9 = 100.0%
        assert rates[0] == 100.0
        # Student 2: 5 / 10 = 50.0%
        assert rates[1] == 50.0
        # Student 3: effective = 10 - 10 = 0 -> NaN
        assert np.isnan(rates[2])

    def test_weighted_average_1d(self):
        scores = np.array([80.0, 90.0])
        weights = np.array([0.4, 0.6])
        # 0.4*80 + 0.6*90 = 32 + 54 = 86.0
        res = vectorized_weighted_average(scores, weights)
        assert res == 86.0

    def test_weighted_average_1d_with_nan_reweighting(self):
        # Academic missing evaluation: missing exam dynamically re-weights homework
        scores = np.array([80.0, np.nan])
        weights = np.array([0.5, 0.5])
        res = vectorized_weighted_average(scores, weights)
        assert res == 80.0

    def test_weighted_average_2d(self):
        # 2 students, 3 components (attendance, assignment, exam)
        scores = np.array([
            [100.0, 80.0, 60.0],
            [70.0, np.nan, 90.0],
        ])
        weights = np.array([0.3, 0.3, 0.4])

        res = vectorized_weighted_average(scores, weights)
        # Student 1: 0.3*100 + 0.3*80 + 0.4*60 = 30 + 24 + 24 = 78.0
        assert res[0] == 78.0
        # Student 2: weights [0.3, 0, 0.4] sum=0.7; (0.3*70 + 0.4*90)/0.7 = (21 + 36)/0.7 = 57/0.7 = 81.43
        assert res[1] == 81.43

    def test_composite_risk_score(self):
        att = np.array([100.0, 0.0, 80.0])
        asn = np.array([100.0, 0.0, 70.0])
        exm = np.array([100.0, 0.0, 90.0])

        risks = vectorized_composite_risk_score(att, asn, exm, weights=(0.35, 0.35, 0.30))

        # Flawless student -> perf = 100.0 -> risk = 0.0
        assert risks[0] == 0.0
        # Fully disengaged -> perf = 0.0 -> risk = 100.0
        assert risks[1] == 100.0
        # Student 3: 0.35*80 + 0.35*70 + 0.30*90 = 28 + 24.5 + 27 = 79.5 -> risk = 20.5
        assert risks[2] == 20.5

    def test_composite_risk_score_with_missing(self):
        att = np.array([80.0])
        asn = np.array([80.0])
        exm = np.array([np.nan])

        risks = vectorized_composite_risk_score(att, asn, exm, weights=(0.35, 0.35, 0.30))
        # Available perf = (0.35*80 + 0.35*80)/0.70 = 80.0 -> risk = 20.0
        assert risks[0] == 20.0
