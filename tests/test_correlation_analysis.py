"""Unit tests for correlation analysis module."""

import unittest
import numpy as np
import pandas as pd

from src.correlation_analysis import (
    interpret_correlation_strength,
    interpret_correlation_direction,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    calculate_kendall_correlation,
    analyze_correlation,
    analyze_attendance_exam_correlation,
    analyze_attendance_completion_correlation,
    analyze_completion_exam_correlation,
    analyze_recent_attendance_risk_correlation,
    generate_correlation_matrix,
    generate_correlation_insights,
    generate_correlation_report_markdown,
    partial_correlation,
)
from src.exceptions import DataValidationError


class TestCorrelationInterpretation(unittest.TestCase):
    """Test correlation interpretation functions."""

    def test_strength_negligible(self):
        """Test negligible strength interpretation."""
        self.assertEqual(interpret_correlation_strength(0.05), "negligible")
        self.assertEqual(interpret_correlation_strength(-0.05), "negligible")

    def test_strength_weak(self):
        """Test weak strength interpretation."""
        self.assertEqual(interpret_correlation_strength(0.2), "weak")
        self.assertEqual(interpret_correlation_strength(-0.2), "weak")

    def test_strength_moderate(self):
        """Test moderate strength interpretation."""
        self.assertEqual(interpret_correlation_strength(0.4), "moderate")
        self.assertEqual(interpret_correlation_strength(-0.4), "moderate")

    def test_strength_strong(self):
        """Test strong strength interpretation."""
        self.assertEqual(interpret_correlation_strength(0.8), "strong")
        self.assertEqual(interpret_correlation_strength(-0.8), "strong")

    def test_direction_positive(self):
        """Test positive direction interpretation."""
        self.assertEqual(interpret_correlation_direction(0.5), "positive")

    def test_direction_negative(self):
        """Test negative direction interpretation."""
        self.assertEqual(interpret_correlation_direction(-0.5), "negative")

    def test_direction_none(self):
        """Test none direction interpretation."""
        self.assertEqual(interpret_correlation_direction(0.05), "none")


class TestPearsonCorrelation(unittest.TestCase):
    """Test Pearson correlation calculation."""

    def test_positive_correlation(self):
        """Test positive correlation."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        corr, p_val, n = calculate_pearson_correlation(x, y)

        self.assertGreater(corr, 0.9)
        self.assertLess(p_val, 0.05)
        self.assertEqual(n, 5)

    def test_negative_correlation(self):
        """Test negative correlation."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([10, 8, 6, 4, 2])
        corr, p_val, n = calculate_pearson_correlation(x, y)

        self.assertLess(corr, -0.9)
        self.assertLess(p_val, 0.05)
        self.assertEqual(n, 5)

    def test_no_correlation(self):
        """Test no correlation."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = pd.Series(np.random.randn(100))
        corr, p_val, n = calculate_pearson_correlation(x, y)

        self.assertLess(abs(corr), 0.3)
        self.assertEqual(n, 100)

    def test_with_missing_values(self):
        """Test correlation with missing values."""
        x = pd.Series([1, 2, np.nan, 4, 5])
        y = pd.Series([2, 4, 6, np.nan, 10])
        corr, p_val, n = calculate_pearson_correlation(x, y)

        self.assertIsNotNone(corr)
        self.assertEqual(n, 3)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        x = pd.Series([1, 2])
        y = pd.Series([2, 4])
        corr, p_val, n = calculate_pearson_correlation(x, y)

        self.assertTrue(np.isnan(corr))
        self.assertTrue(np.isnan(p_val))
        self.assertEqual(n, 2)


class TestSpearmanCorrelation(unittest.TestCase):
    """Test Spearman correlation calculation."""

    def test_monotonic_relationship(self):
        """Test monotonic relationship."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 4, 9, 16, 25])  # Quadratic but monotonic
        corr, p_val, n = calculate_spearman_correlation(x, y)

        self.assertGreater(corr, 0.99)
        self.assertLess(p_val, 0.05)
        self.assertEqual(n, 5)


class TestKendallCorrelation(unittest.TestCase):
    """Test Kendall correlation calculation."""

    def test_ordinal_correlation(self):
        """Test ordinal correlation."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 3, 4, 5, 6])
        corr, p_val, n = calculate_kendall_correlation(x, y)

        self.assertGreater(corr, 0.99)
        self.assertLess(p_val, 0.05)
        self.assertEqual(n, 5)


class TestAnalyzeCorrelation(unittest.TestCase):
    """Test complete correlation analysis."""

    def test_pearson_analysis(self):
        """Test Pearson correlation analysis."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = x + pd.Series(np.random.randn(100) * 0.5)

        result = analyze_correlation(x, y, method="pearson", variable_x_name="x", variable_y_name="y")

        self.assertEqual(result.variable_x, "x")
        self.assertEqual(result.variable_y, "y")
        self.assertEqual(result.method, "pearson")
        self.assertIsNotNone(result.correlation_coefficient)
        self.assertIsNotNone(result.p_value)
        self.assertEqual(result.sample_size, 100)

    def test_spearman_analysis(self):
        """Test Spearman correlation analysis."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])

        result = analyze_correlation(x, y, method="spearman", variable_x_name="x", variable_y_name="y")

        self.assertEqual(result.method, "spearman")
        self.assertIsNotNone(result.correlation_coefficient)

    def test_invalid_method(self):
        """Test with invalid correlation method."""
        x = pd.Series([1, 2, 3])
        y = pd.Series([2, 4, 6])

        with self.assertRaises(DataValidationError):
            analyze_correlation(x, y, method="invalid")

    def test_interpretation_generation(self):
        """Test interpretation generation."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = x + pd.Series(np.random.randn(100) * 0.5)

        result = analyze_correlation(x, y, method="pearson")

        self.assertIsNotNone(result.interpretation)
        self.assertIsNotNone(result.strength)
        self.assertIsNotNone(result.direction)
        self.assertIsInstance(result.notes, list)


class TestAttendanceExamCorrelation(unittest.TestCase):
    """Test attendance vs exam correlation analysis."""

    def test_attendance_exam_correlation(self):
        """Test attendance vs exam correlation."""
        np.random.seed(42)
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = analyze_attendance_exam_correlation(df)

        self.assertEqual(result.variable_x, "attendance_percentage")
        self.assertEqual(result.variable_y, "average_exam_score")
        self.assertIsNotNone(result.correlation_coefficient)

    def test_missing_column(self):
        """Test with missing required column."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 100, 100),
        })

        with self.assertRaises(DataValidationError):
            analyze_attendance_exam_correlation(df)


class TestAttendanceCompletionCorrelation(unittest.TestCase):
    """Test attendance vs completion correlation analysis."""

    def test_attendance_completion_correlation(self):
        """Test attendance vs completion correlation."""
        np.random.seed(42)
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
        })

        result = analyze_attendance_completion_correlation(df)

        self.assertEqual(result.variable_x, "attendance_percentage")
        self.assertEqual(result.variable_y, "assignment_completion_rate")
        self.assertIsNotNone(result.correlation_coefficient)


class TestCompletionExamCorrelation(unittest.TestCase):
    """Test completion vs exam correlation analysis."""

    def test_completion_exam_correlation(self):
        """Test completion vs exam correlation."""
        np.random.seed(42)
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = analyze_completion_exam_correlation(df)

        self.assertEqual(result.variable_x, "assignment_completion_rate")
        self.assertEqual(result.variable_y, "average_exam_score")
        self.assertIsNotNone(result.correlation_coefficient)


class TestRecentAttendanceRiskCorrelation(unittest.TestCase):
    """Test recent attendance vs risk indicator correlation."""

    def test_attendance_trend_correlation(self):
        """Test attendance vs trend correlation."""
        np.random.seed(42)
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "attendance_trend": np.random.uniform(-20, 20, 100),
        })

        result = analyze_recent_attendance_risk_correlation(df)

        self.assertEqual(result.variable_x, "attendance_percentage")
        self.assertEqual(result.variable_y, "attendance_trend")
        self.assertEqual(result.method, "spearman")

    def test_custom_risk_indicator(self):
        """Test with custom risk indicator column."""
        np.random.seed(42)
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "custom_risk": np.random.uniform(0, 1, 100),
        })

        result = analyze_recent_attendance_risk_correlation(df, risk_indicator_column="custom_risk")

        self.assertEqual(result.variable_y, "custom_risk")


class TestCorrelationMatrix(unittest.TestCase):
    """Test correlation matrix generation."""

    def test_correlation_matrix(self):
        """Test correlation matrix generation."""
        np.random.seed(42)
        df = pd.DataFrame({
            "var1": np.random.randn(100),
            "var2": np.random.randn(100),
            "var3": np.random.randn(100),
        })

        matrix = generate_correlation_matrix(df, ["var1", "var2", "var3"], method="pearson")

        self.assertEqual(len(matrix.variables), 3)
        self.assertEqual(len(matrix.correlation_matrix), 3)
        self.assertEqual(matrix.method, "pearson")

    def test_diagonal_values(self):
        """Test diagonal values are 1.0."""
        np.random.seed(42)
        df = pd.DataFrame({
            "var1": np.random.randn(50),
            "var2": np.random.randn(50),
        })

        matrix = generate_correlation_matrix(df, ["var1", "var2"], method="pearson")

        self.assertEqual(matrix.correlation_matrix[0][0], 1.0)
        self.assertEqual(matrix.correlation_matrix[1][1], 1.0)

    def test_missing_column(self):
        """Test with missing column."""
        df = pd.DataFrame({"var1": [1, 2, 3]})

        with self.assertRaises(DataValidationError):
            generate_correlation_matrix(df, ["var1", "var2"])


class TestCorrelationInsights(unittest.TestCase):
    """Test correlation insights generation."""

    def test_insights_generation(self):
        """Test insights generation."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = x + pd.Series(np.random.randn(100) * 0.5)

        result = analyze_correlation(x, y, method="pearson", variable_x_name="attendance", variable_y_name="exam")
        insights = generate_correlation_insights(result)

        self.assertIsNotNone(insights.correlation_pair)
        self.assertIsInstance(insights.statistical_significance, bool)
        self.assertIsInstance(insights.practical_significance, bool)
        self.assertIsInstance(insights.potential_confounders, list)
        self.assertIsInstance(insights.limitations, list)
        self.assertIsInstance(insights.recommendations, list)
        self.assertIn("Correlation does not imply causation", insights.causation_warning)

    def test_confounders_for_attendance(self):
        """Test confounder identification for attendance-related variables."""
        result = analyze_correlation(
            pd.Series([1, 2, 3]),
            pd.Series([2, 4, 6]),
            variable_x_name="attendance_percentage",
            variable_y_name="exam_score",
        )
        insights = generate_correlation_insights(result)

        self.assertGreater(len(insights.potential_confounders), 0)


class TestMarkdownReport(unittest.TestCase):
    """Test markdown report generation."""

    def test_markdown_generation(self):
        """Test markdown report generation."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = x + pd.Series(np.random.randn(100) * 0.5)

        result = analyze_correlation(x, y, variable_x_name="x", variable_y_name="y")
        insights = generate_correlation_insights(result)

        markdown = generate_correlation_report_markdown([result], [insights])

        self.assertIn("# Academic Correlation Analysis Report", markdown)
        self.assertIn("Correlation does not imply causation", markdown)
        self.assertIn("x vs y", markdown)

    def test_empty_results(self):
        """Test with empty results list."""
        markdown = generate_correlation_report_markdown([])

        self.assertIn("# Academic Correlation Analysis Report", markdown)
        self.assertIn("0", markdown)


class TestPartialCorrelation(unittest.TestCase):
    """Test partial correlation calculation."""

    def test_partial_correlation(self):
        """Test partial correlation calculation."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        z = pd.Series(np.random.randn(100))
        y = x + 0.5 * z + pd.Series(np.random.randn(100) * 0.3)

        partial_r, n = partial_correlation(x, y, z)

        self.assertIsNotNone(partial_r)
        self.assertEqual(n, 100)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        x = pd.Series([1, 2])
        y = pd.Series([2, 4])
        z = pd.Series([3, 6])

        partial_r, n = partial_correlation(x, y, z)

        self.assertTrue(np.isnan(partial_r))
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
