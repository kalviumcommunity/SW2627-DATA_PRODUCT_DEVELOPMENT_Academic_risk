"""Unit tests for distribution analysis module."""

import unittest
import numpy as np
import pandas as pd

from src.distribution_analysis import (
    calculate_distribution_summary,
    identify_distribution_type,
    detect_outliers_iqr,
    detect_outliers_zscore,
    generate_distribution_insights,
    analyze_distribution,
    analyze_assignment_completion_distribution,
    analyze_assignment_scores_distribution,
    analyze_exam_scores_distribution,
    analyze_submission_delays_distribution,
    compare_distributions,
    detect_academic_anomalies,
    generate_distribution_summary_markdown,
)
from src.exceptions import DataValidationError


class TestDistributionSummary(unittest.TestCase):
    """Test distribution summary calculation."""

    def test_normal_distribution(self):
        """Test summary calculation for normal distribution."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(70, 10, 100))
        summary = calculate_distribution_summary(data, "test_metric")

        self.assertEqual(summary.metric_name, "test_metric")
        self.assertEqual(summary.count, 100)
        self.assertEqual(summary.missing_count, 0)
        self.assertIsNotNone(summary.mean)
        self.assertIsNotNone(summary.median)
        self.assertIsNotNone(summary.std)
        self.assertIsNotNone(summary.skewness)
        self.assertIsNotNone(summary.kurtosis)

    def test_with_missing_values(self):
        """Test summary calculation with missing values."""
        data = pd.Series([80, 85, np.nan, 90, np.nan, 75])
        summary = calculate_distribution_summary(data, "test_metric")

        self.assertEqual(summary.count, 6)
        self.assertEqual(summary.missing_count, 2)
        self.assertEqual(summary.mean, 82.5)

    def test_empty_series(self):
        """Test summary calculation for empty series."""
        data = pd.Series([])
        summary = calculate_distribution_summary(data, "test_metric")

        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.median)

    def test_all_nan_series(self):
        """Test summary calculation for all-NaN series."""
        data = pd.Series([np.nan, np.nan, np.nan])
        summary = calculate_distribution_summary(data, "test_metric")

        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.missing_count, 3)
        self.assertIsNone(summary.mean)


class TestDistributionType(unittest.TestCase):
    """Test distribution type identification."""

    def test_normal_distribution(self):
        """Test identification of normal distribution."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(50, 10, 1000))
        summary = calculate_distribution_summary(data, "test")
        dist_type = identify_distribution_type(summary, data)

        self.assertEqual(dist_type, "normal")

    def test_right_skewed(self):
        """Test identification of right-skewed distribution."""
        np.random.seed(42)
        data = pd.Series(np.random.exponential(10, 1000))
        summary = calculate_distribution_summary(data, "test")
        dist_type = identify_distribution_type(summary, data)

        self.assertEqual(dist_type, "skewed_right")

    def test_insufficient_data(self):
        """Test distribution type with insufficient data."""
        data = pd.Series([1, 2])
        summary = calculate_distribution_summary(data, "test")
        dist_type = identify_distribution_type(summary, data)

        self.assertEqual(dist_type, "unknown")


class TestOutlierDetection(unittest.TestCase):
    """Test outlier detection methods."""

    def test_iqr_outliers(self):
        """Test IQR-based outlier detection."""
        data = pd.Series([10, 12, 11, 13, 100, 12, 11, 14, 12, 13])
        outlier_mask, lower, upper = detect_outliers_iqr(data)

        self.assertTrue(outlier_mask.any())
        self.assertLess(lower, 15)
        self.assertGreater(upper, 15)

    def test_iqr_no_outliers(self):
        """Test IQR with no outliers."""
        data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        outlier_mask, lower, upper = detect_outliers_iqr(data)

        self.assertFalse(outlier_mask.any())

    def test_zscore_outliers(self):
        """Test z-score based outlier detection."""
        data = pd.Series([10, 11, 12, 13, 14, 100, 12, 11, 13, 14])
        outlier_mask, threshold = detect_outliers_zscore(data, threshold=2.5)

        self.assertTrue(outlier_mask.any())

    def test_zscore_empty_series(self):
        """Test z-score with empty series."""
        data = pd.Series([])
        outlier_mask, threshold = detect_outliers_zscore(data)

        self.assertEqual(len(outlier_mask), 0)


class TestDistributionInsights(unittest.TestCase):
    """Test distribution insights generation."""

    def test_attendance_insights(self):
        """Test insights for attendance metric."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(60, 15, 100))
        summary = calculate_distribution_summary(data, "attendance_percentage")
        insights = generate_distribution_insights(data, summary, "attendance_percentage")

        self.assertEqual(insights.metric_name, "attendance_percentage")
        self.assertIsNotNone(insights.distribution_type)
        self.assertIn("low_risk", insights.risk_thresholds)
        self.assertIsInstance(insights.patterns, list)
        self.assertIsInstance(insights.anomalies, list)
        self.assertIsInstance(insights.recommendations, list)

    def test_score_insights(self):
        """Test insights for score metric."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(55, 20, 100))
        summary = calculate_distribution_summary(data, "exam_scores")
        insights = generate_distribution_insights(data, summary, "exam_scores")

        self.assertEqual(insights.metric_name, "exam_scores")
        self.assertIn("failing", insights.risk_thresholds)

    def test_empty_data_insights(self):
        """Test insights with empty data."""
        data = pd.Series([])
        summary = calculate_distribution_summary(data, "test")
        insights = generate_distribution_insights(data, summary, "test")

        self.assertEqual(insights.outlier_count, 0)
        self.assertIn("No data available", insights.patterns[0])


class TestAnalyzeDistribution(unittest.TestCase):
    """Test complete distribution analysis."""

    def test_basic_analysis(self):
        """Test basic distribution analysis."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(70, 10, 100))
        report = analyze_distribution(data, "test_metric")

        self.assertEqual(report.metric_name, "test_metric")
        self.assertIsNotNone(report.summary)
        self.assertIsNotNone(report.insights)

    def test_analysis_with_histogram(self):
        """Test analysis with histogram data."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(70, 10, 100))
        report = analyze_distribution(data, "test_metric", include_histogram=True, bins=10)

        self.assertIsNotNone(report.histogram_data)
        self.assertIn("counts", report.histogram_data)
        self.assertIn("bin_edges", report.histogram_data)

    def test_invalid_input(self):
        """Test analysis with invalid input."""
        with self.assertRaises(DataValidationError):
            analyze_distribution([1, 2, 3], "test_metric")


class TestAssignmentCompletionAnalysis(unittest.TestCase):
    """Test assignment completion distribution analysis."""

    def test_completion_analysis(self):
        """Test assignment completion analysis."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
        })
        report = analyze_assignment_completion_distribution(df)

        self.assertEqual(report.metric_name, "assignment_completion_rate")
        self.assertIsNotNone(report.summary)

    def test_missing_column(self):
        """Test with missing required column."""
        df = pd.DataFrame({"student_id": range(1, 101)})
        with self.assertRaises(DataValidationError):
            analyze_assignment_completion_distribution(df)


class TestAssignmentScoresAnalysis(unittest.TestCase):
    """Test assignment scores distribution analysis."""

    def test_scores_analysis(self):
        """Test assignment scores analysis."""
        df = pd.DataFrame({
            "submission_id": range(1, 101),
            "score": np.random.uniform(50, 100, 100),
        })
        report = analyze_assignment_scores_distribution(df)

        self.assertEqual(report.metric_name, "assignment_scores")
        self.assertIsNotNone(report.summary)

    def test_missing_column(self):
        """Test with missing required column."""
        df = pd.DataFrame({"submission_id": range(1, 101)})
        with self.assertRaises(DataValidationError):
            analyze_assignment_scores_distribution(df)


class TestExamScoresAnalysis(unittest.TestCase):
    """Test exam scores distribution analysis."""

    def test_exam_scores_analysis(self):
        """Test exam scores analysis."""
        df = pd.DataFrame({
            "exam_id": range(1, 101),
            "score": np.random.uniform(40, 95, 100),
        })
        report = analyze_exam_scores_distribution(df)

        self.assertEqual(report.metric_name, "exam_scores")
        self.assertIsNotNone(report.summary)

    def test_missing_column(self):
        """Test with missing required column."""
        df = pd.DataFrame({"exam_id": range(1, 101)})
        with self.assertRaises(DataValidationError):
            analyze_exam_scores_distribution(df)


class TestSubmissionDelaysAnalysis(unittest.TestCase):
    """Test submission delays distribution analysis."""

    def test_delays_analysis_days_late(self):
        """Test delays analysis with days_late column."""
        df = pd.DataFrame({
            "submission_id": range(1, 101),
            "days_late": np.random.exponential(2, 100),
        })
        report = analyze_submission_delays_distribution(df)

        self.assertEqual(report.metric_name, "submission_delays")
        self.assertIsNotNone(report.summary)

    def test_delays_analysis_is_late(self):
        """Test delays analysis with is_late column."""
        df = pd.DataFrame({
            "submission_id": range(1, 101),
            "is_late": np.random.choice([0, 1], 100),
        })
        report = analyze_submission_delays_distribution(df)

        self.assertEqual(report.metric_name, "submission_delays")

    def test_missing_column(self):
        """Test with missing required columns."""
        df = pd.DataFrame({"submission_id": range(1, 101)})
        with self.assertRaises(DataValidationError):
            analyze_submission_delays_distribution(df)


class TestCompareDistributions(unittest.TestCase):
    """Test distribution comparison."""

    def test_normal_comparison(self):
        """Test comparison of two similar distributions."""
        np.random.seed(42)
        series1 = pd.Series(np.random.normal(70, 10, 100))
        series2 = pd.Series(np.random.normal(72, 10, 100))
        result = compare_distributions(series1, series2, "group1", "group2")

        self.assertEqual(result["name1"], "group1")
        self.assertEqual(result["name2"], "group2")
        self.assertIn("mean1", result)
        self.assertIn("mean2", result)
        self.assertIn("mann_whitney_p_value", result)

    def test_different_distributions(self):
        """Test comparison of different distributions."""
        np.random.seed(42)
        series1 = pd.Series(np.random.normal(50, 10, 100))
        series2 = pd.Series(np.random.normal(80, 10, 100))
        result = compare_distributions(series1, series2, "group1", "group2")

        self.assertTrue(result.get("significantly_different", False))

    def test_empty_distribution(self):
        """Test comparison with empty distribution."""
        series1 = pd.Series([1, 2, 3])
        series2 = pd.Series([])
        result = compare_distributions(series1, series2, "group1", "group2")

        self.assertIn("error", result)


class TestDetectAcademicAnomalies(unittest.TestCase):
    """Test academic anomaly detection."""

    def test_critical_attendance(self):
        """Test detection of critical attendance anomalies."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [30.0] * 10 + [80.0] * 90,
        })
        anomalies = detect_academic_anomalies(df)

        self.assertEqual(len(anomalies["critical_attendance"]), 10)

    def test_critical_completion(self):
        """Test detection of critical completion anomalies."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "assignment_completion_rate": [35.0] * 5 + [85.0] * 95,
        })
        anomalies = detect_academic_anomalies(df)

        self.assertEqual(len(anomalies["critical_completion"]), 5)

    def test_critical_scores(self):
        """Test detection of critical score anomalies."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "average_assignment_score": [35.0] * 8 + [75.0] * 92,
        })
        anomalies = detect_academic_anomalies(df)

        self.assertEqual(len(anomalies["critical_scores"]), 8)

    def test_severe_delays(self):
        """Test detection of severe delay anomalies."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "late_submission_count": [6] * 3 + [1] * 97,
        })
        anomalies = detect_academic_anomalies(df)

        self.assertEqual(len(anomalies["severe_delays"]), 3)

    def test_declining_trends(self):
        """Test detection of declining trend anomalies."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_trend": [-15.0] * 4 + [2.0] * 96,
        })
        anomalies = detect_academic_anomalies(df)

        self.assertEqual(len(anomalies["declining_trends"]), 4)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        anomalies = detect_academic_anomalies(df)

        for key in anomalies:
            self.assertEqual(len(anomalies[key]), 0)


class TestMarkdownGeneration(unittest.TestCase):
    """Test markdown summary generation."""

    def test_markdown_generation(self):
        """Test markdown summary generation."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(70, 10, 100))
        report = analyze_distribution(data, "test_metric")
        reports = {"test_metric": report}

        markdown = generate_distribution_summary_markdown(reports)

        self.assertIn("# Academic Distribution Analysis Summary", markdown)
        self.assertIn("Test Metric", markdown)
        self.assertIn("Statistical Summary", markdown)

    def test_empty_reports(self):
        """Test with empty reports dictionary."""
        markdown = generate_distribution_summary_markdown({})

        self.assertIn("# Academic Distribution Analysis Summary", markdown)
        self.assertIn("0", markdown)


if __name__ == "__main__":
    unittest.main()
