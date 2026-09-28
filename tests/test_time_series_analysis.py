"""Unit tests for time series analysis module."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.time_series_analysis import (
    ensure_datetime_column,
    calculate_trend,
    aggregate_time_series,
    calculate_rolling_metrics,
    analyze_attendance_trends,
    analyze_assignment_completion_trends,
    analyze_exam_performance_trends,
    analyze_student_engagement_trends,
    compare_time_series,
    generate_time_series_report_markdown,
)
from src.exceptions import DataValidationError


class TestEnsureDatetimeColumn(unittest.TestCase):
    """Test datetime column conversion."""

    def test_convert_string_dates(self):
        """Test conversion of string dates to datetime."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [1, 2, 3],
        })

        result = ensure_datetime_column(df, "date")

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))

    def test_already_datetime(self):
        """Test with already datetime column."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value": [1, 2],
        })

        result = ensure_datetime_column(df, "date")

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))

    def test_missing_column(self):
        """Test with missing column."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with self.assertRaises(DataValidationError):
            ensure_datetime_column(df, "date")

    def test_invalid_dates(self):
        """Test with invalid date strings."""
        df = pd.DataFrame({
            "date": ["invalid", "also_invalid"],
            "value": [1, 2],
        })

        with self.assertRaises(DataValidationError):
            ensure_datetime_column(df, "date")


class TestCalculateTrend(unittest.TestCase):
    """Test trend calculation."""

    def test_increasing_trend(self):
        """Test increasing trend detection."""
        series = pd.Series([1, 2, 3, 4, 5])
        direction, slope, r_squared = calculate_trend(series)

        self.assertEqual(direction, "increasing")
        self.assertGreater(slope, 0)
        self.assertIsNotNone(r_squared)

    def test_decreasing_trend(self):
        """Test decreasing trend detection."""
        series = pd.Series([5, 4, 3, 2, 1])
        direction, slope, r_squared = calculate_trend(series)

        self.assertEqual(direction, "decreasing")
        self.assertLess(slope, 0)
        self.assertIsNotNone(r_squared)

    def test_stable_trend(self):
        """Test stable trend detection."""
        series = pd.Series([5, 5, 5, 5, 5])
        direction, slope, r_squared = calculate_trend(series)

        self.assertEqual(direction, "stable")
        self.assertAlmostEqual(slope, 0, places=5)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        series = pd.Series([1, 2])
        direction, slope, r_squared = calculate_trend(series)

        self.assertEqual(direction, "unknown")
        self.assertIsNone(slope)
        self.assertIsNone(r_squared)

    def test_with_nan_values(self):
        """Test with NaN values."""
        series = pd.Series([1, 2, np.nan, 4, 5])
        direction, slope, r_squared = calculate_trend(series)

        self.assertIsNotNone(direction)


class TestAggregateTimeSeries(unittest.TestCase):
    """Test time series aggregation."""

    def test_weekly_aggregation(self):
        """Test weekly aggregation."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(30),
        })

        result = aggregate_time_series(df, "date", "value", aggregation="weekly", agg_func="mean")

        self.assertLess(len(result), 30)
        self.assertIn("date", result.columns)
        self.assertIn("value", result.columns)

    def test_monthly_aggregation(self):
        """Test monthly aggregation."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(90),
        })

        result = aggregate_time_series(df, "date", "value", aggregation="monthly", agg_func="mean")

        self.assertLess(len(result), 90)

    def test_sum_aggregation(self):
        """Test sum aggregation."""
        dates = pd.date_range("2024-01-01", periods=14, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": [1] * 14,
        })

        result = aggregate_time_series(df, "date", "value", aggregation="weekly", agg_func="sum")

        self.assertEqual(result["value"].iloc[0], 7)

    def test_invalid_aggregation(self):
        """Test with invalid aggregation level."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "value": [1] * 10,
        })

        with self.assertRaises(DataValidationError):
            aggregate_time_series(df, "date", "value", aggregation="yearly")

    def test_invalid_agg_func(self):
        """Test with invalid aggregation function."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "value": [1] * 10,
        })

        with self.assertRaises(DataValidationError):
            aggregate_time_series(df, "date", "value", aggregation="weekly", agg_func="median")


class TestCalculateRollingMetrics(unittest.TestCase):
    """Test rolling metrics calculation."""

    def test_rolling_metrics(self):
        """Test rolling metrics calculation."""
        series = pd.Series(np.random.randn(100))
        rolling = calculate_rolling_metrics(series, windows=[7, 14])

        self.assertIn("rolling_mean_7", rolling)
        self.assertIn("rolling_std_7", rolling)
        self.assertIn("rolling_mean_14", rolling)
        self.assertEqual(len(rolling["rolling_mean_7"]), 100)

    def test_insufficient_data_for_window(self):
        """Test with insufficient data for window."""
        series = pd.Series([1, 2, 3])
        rolling = calculate_rolling_metrics(series, windows=[7])

        # Should not include metrics for windows larger than data
        self.assertEqual(len(rolling), 0)


class TestAnalyzeAttendanceTrends(unittest.TestCase):
    """Test attendance trend analysis."""

    def test_attendance_trends(self):
        """Test attendance trend analysis."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "student_id": [1] * 30,
            "status": ["Present"] * 25 + ["Absent"] * 5,
        })

        result = analyze_attendance_trends(df, aggregation="weekly")

        self.assertEqual(result.metric_name, "attendance_trend")
        self.assertEqual(result.aggregation_level, "weekly")
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.rolling_metrics)

    def test_missing_date_column(self):
        """Test with missing date column."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "status": ["Present"] * 3,
        })

        with self.assertRaises(DataValidationError):
            analyze_attendance_trends(df)

    def test_missing_student_id_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "status": ["Present"] * 3,
        })

        with self.assertRaises(DataValidationError):
            analyze_attendance_trends(df)


class TestAnalyzeAssignmentCompletionTrends(unittest.TestCase):
    """Test assignment completion trend analysis."""

    def test_completion_trends(self):
        """Test completion trend analysis."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "submission_date": dates,
            "student_id": range(1, 31),
        })

        result = analyze_assignment_completion_trends(df, aggregation="weekly")

        self.assertEqual(result.metric_name, "assignment_completion_trend")
        self.assertEqual(result.aggregation_level, "weekly")
        self.assertIsNotNone(result.metrics)

    def test_missing_date_column(self):
        """Test with missing date column."""
        df = pd.DataFrame({"student_id": [1, 2, 3]})

        with self.assertRaises(DataValidationError):
            analyze_assignment_completion_trends(df)


class TestAnalyzeExamPerformanceTrends(unittest.TestCase):
    """Test exam performance trend analysis."""

    def test_exam_trends(self):
        """Test exam performance trend analysis."""
        dates = pd.date_range("2024-01-01", periods=10, freq="W")
        df = pd.DataFrame({
            "exam_date": dates,
            "score": np.random.uniform(50, 95, 10),
        })

        result = analyze_exam_performance_trends(df, aggregation="weekly")

        self.assertEqual(result.metric_name, "exam_performance_trend")
        self.assertIsNotNone(result.metrics.mean_value)

    def test_missing_score_column(self):
        """Test with missing score column."""
        df = pd.DataFrame({
            "exam_date": pd.date_range("2024-01-01", periods=3),
        })

        with self.assertRaises(DataValidationError):
            analyze_exam_performance_trends(df)


class TestAnalyzeStudentEngagementTrends(unittest.TestCase):
    """Test student engagement trend analysis."""

    def test_engagement_trends(self):
        """Test engagement trend analysis."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        att_df = pd.DataFrame({
            "date": dates,
            "student_id": range(1, 31),
        })
        sub_df = pd.DataFrame({
            "date": dates,
            "student_id": range(1, 31),
        })

        result = analyze_student_engagement_trends(att_df, sub_df, aggregation="weekly")

        self.assertEqual(result.metric_name, "student_engagement_trend")
        self.assertIsNotNone(result.metrics)

    def test_empty_dataframes(self):
        """Test with empty DataFrames."""
        att_df = pd.DataFrame(columns=["date", "student_id"])
        sub_df = pd.DataFrame(columns=["date", "student_id"])

        result = analyze_student_engagement_trends(att_df, sub_df)

        self.assertEqual(result.metric_name, "student_engagement_trend")


class TestCompareTimeSeries(unittest.TestCase):
    """Test time series comparison."""

    def test_compare_series(self):
        """Test comparing two time series."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([2, 3, 4, 5, 6])

        result = compare_time_series(series1, series2, "Series 1", "Series 2")

        self.assertEqual(result["name1"], "Series 1")
        self.assertEqual(result["name2"], "Series 2")
        self.assertIsNotNone(result["correlation"])
        self.assertEqual(result["overlapping_periods"], 5)

    def test_no_overlap(self):
        """Test with no overlapping data."""
        series1 = pd.Series([1, 2, 3], index=[0, 1, 2])
        series2 = pd.Series([4, 5, 6], index=[3, 4, 5])

        result = compare_time_series(series1, series2)

        self.assertIn("error", result)


class TestGenerateTimeSeriesReportMarkdown(unittest.TestCase):
    """Test markdown report generation."""

    def test_markdown_generation(self):
        """Test markdown report generation."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "student_id": [1] * 30,
            "status": ["Present"] * 25 + ["Absent"] * 5,
        })

        result = analyze_attendance_trends(df, aggregation="weekly")
        markdown = generate_time_series_report_markdown([result])

        self.assertIn("# Academic Time-Series Analysis Report", markdown)
        self.assertIn("Attendance Trend", markdown)
        self.assertIn("Trend Analysis", markdown)

    def test_empty_results(self):
        """Test with empty results list."""
        markdown = generate_time_series_report_markdown([])

        self.assertIn("# Academic Time-Series Analysis Report", markdown)
        self.assertIn("0", markdown)


if __name__ == "__main__":
    unittest.main()
