"""Unit tests for KPI framework module."""

import unittest
import numpy as np
import pandas as pd

from src.kpi_framework import (
    calculate_total_students,
    calculate_average_attendance,
    calculate_assignment_completion_rate,
    calculate_average_exam_score,
    calculate_students_requiring_review,
    calculate_low_attendance_rate,
    calculate_missing_submission_rate,
    calculate_engagement_rate,
    calculate_course_performance,
    get_kpi_definition,
    get_all_kpi_definitions,
    generate_kpi_report_markdown,
)
from src.exceptions import DataValidationError


class TestCalculateTotalStudents(unittest.TestCase):
    """Test Total Students KPI calculation."""

    def test_total_students(self):
        """Test total students calculation."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4, 5],
            "course_id": ["C101"] * 5,
        })

        result = calculate_total_students(df)

        self.assertEqual(result.kpi_name, "Total Students")
        self.assertEqual(result.value, 5.0)
        self.assertFalse(result.is_percentage)

    def test_with_duplicates(self):
        """Test with duplicate enrollments."""
        df = pd.DataFrame({
            "student_id": [1, 1, 2, 3, 3],
            "course_id": ["C101", "C102", "C101", "C101", "C102"],
        })

        result = calculate_total_students(df)

        self.assertEqual(result.value, 3.0)

    def test_missing_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({"course_id": ["C101", "C102"]})

        with self.assertRaises(DataValidationError):
            calculate_total_students(df)


class TestCalculateAverageAttendance(unittest.TestCase):
    """Test Average Attendance KPI calculation."""

    def test_average_attendance(self):
        """Test average attendance calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 95, 100),
        })

        result = calculate_average_attendance(df)

        self.assertEqual(result.kpi_name, "Average Attendance")
        self.assertIsNotNone(result.value)
        self.assertTrue(result.is_percentage)

    def test_with_missing_values(self):
        """Test with missing values."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [80.0] * 50 + [np.nan] * 50,
        })

        result = calculate_average_attendance(df)

        self.assertEqual(result.value, 80.0)

    def test_missing_column(self):
        """Test with missing attendance column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_average_attendance(df)


class TestCalculateAssignmentCompletionRate(unittest.TestCase):
    """Test Assignment Completion Rate KPI calculation."""

    def test_completion_rate(self):
        """Test completion rate calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "assignment_completion_rate": np.random.uniform(50, 95, 100),
        })

        result = calculate_assignment_completion_rate(df)

        self.assertEqual(result.kpi_name, "Assignment Completion Rate")
        self.assertIsNotNone(result.value)
        self.assertTrue(result.is_percentage)

    def test_missing_column(self):
        """Test with missing completion column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_assignment_completion_rate(df)


class TestCalculateAverageExamScore(unittest.TestCase):
    """Test Average Exam Score KPI calculation."""

    def test_average_exam_score(self):
        """Test average exam score calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = calculate_average_exam_score(df)

        self.assertEqual(result.kpi_name, "Average Exam Score")
        self.assertIsNotNone(result.value)
        self.assertFalse(result.is_percentage)

    def test_missing_column(self):
        """Test with missing exam column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_average_exam_score(df)


class TestCalculateStudentsRequiringReview(unittest.TestCase):
    """Test Students Requiring Review KPI calculation."""

    def test_students_requiring_review(self):
        """Test students requiring review calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [80.0] * 50 + [60.0] * 50,
            "assignment_completion_rate": [85.0] * 50 + [65.0] * 50,
            "average_exam_score": [75.0] * 50 + [45.0] * 50,
        })

        result = calculate_students_requiring_review(df)

        self.assertEqual(result.kpi_name, "Students Requiring Review")
        self.assertEqual(result.value, 50.0)

    def test_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_students_requiring_review(df)


class TestCalculateLowAttendanceRate(unittest.TestCase):
    """Test Low Attendance Rate KPI calculation."""

    def test_low_attendance_rate(self):
        """Test low attendance rate calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [80.0] * 70 + [60.0] * 30,
        })

        result = calculate_low_attendance_rate(df)

        self.assertEqual(result.kpi_name, "Low Attendance Rate")
        self.assertEqual(result.value, 30.0)
        self.assertTrue(result.is_percentage)

    def test_custom_threshold(self):
        """Test with custom threshold."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [80.0] * 60 + [60.0] * 40,
        })

        result = calculate_low_attendance_rate(df, threshold=80.0)

        self.assertEqual(result.value, 40.0)

    def test_missing_column(self):
        """Test with missing attendance column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_low_attendance_rate(df)


class TestCalculateMissingSubmissionRate(unittest.TestCase):
    """Test Missing Submission Rate KPI calculation."""

    def test_missing_submission_rate(self):
        """Test missing submission rate calculation."""
        submissions_df = pd.DataFrame({
            "student_id": range(1, 81),
            "assignment_id": ["A1"] * 80,
        })
        assignments_df = pd.DataFrame({
            "assignment_id": ["A1"] * 100,
        })

        result = calculate_missing_submission_rate(submissions_df, assignments_df)

        self.assertEqual(result.kpi_name, "Missing Submission Rate")
        self.assertTrue(result.is_percentage)

    def test_missing_columns(self):
        """Test with missing columns."""
        submissions_df = pd.DataFrame({"student_id": [1, 2]})
        assignments_df = pd.DataFrame({"course_id": ["C101"]})

        with self.assertRaises(DataValidationError):
            calculate_missing_submission_rate(submissions_df, assignments_df)


class TestCalculateEngagementRate(unittest.TestCase):
    """Test Engagement Rate KPI calculation."""

    def test_engagement_rate(self):
        """Test engagement rate calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [80.0] * 100,
            "assignment_completion_rate": [85.0] * 100,
        })

        result = calculate_engagement_rate(df)

        self.assertEqual(result.kpi_name, "Engagement Rate")
        self.assertEqual(result.value, 82.5)
        self.assertTrue(result.is_percentage)

    def test_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_engagement_rate(df)


class TestCalculateCoursePerformance(unittest.TestCase):
    """Test Course Performance KPI calculation."""

    def test_course_performance(self):
        """Test course performance calculation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "assignment_completion_rate": [80.0] * 100,
            "average_exam_score": [75.0] * 100,
        })

        result = calculate_course_performance(df)

        self.assertEqual(result.kpi_name, "Course Performance")
        self.assertEqual(result.value, 77.0)
        self.assertFalse(result.is_percentage)

    def test_custom_weights(self):
        """Test with custom weights."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "assignment_completion_rate": [80.0] * 100,
            "average_exam_score": [75.0] * 100,
        })

        result = calculate_course_performance(df, assignment_weight=0.5, exam_weight=0.5)

        self.assertEqual(result.value, 77.5)

    def test_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            calculate_course_performance(df)


class TestGetKPIDefinition(unittest.TestCase):
    """Test KPI definition retrieval."""

    def test_get_definition(self):
        """Test getting a specific KPI definition."""
        definition = get_kpi_definition("total_students")

        self.assertEqual(definition.name, "Total Students")
        self.assertIsNotNone(definition.formula)
        self.assertIsNotNone(definition.data_source)
        self.assertIsNotNone(definition.meaning)
        self.assertGreater(len(definition.limitations), 0)

    def test_invalid_kpi_name(self):
        """Test with invalid KPI name."""
        with self.assertRaises(DataValidationError):
            get_kpi_definition("invalid_kpi")


class TestGetAllKPIDefinitions(unittest.TestCase):
    """Test getting all KPI definitions."""

    def test_all_definitions(self):
        """Test getting all KPI definitions."""
        definitions = get_all_kpi_definitions()

        self.assertEqual(len(definitions), 9)
        self.assertIn("total_students", definitions)
        self.assertIn("average_attendance", definitions)
        self.assertIn("assignment_completion_rate", definitions)
        self.assertIn("average_exam_score", definitions)
        self.assertIn("students_requiring_review", definitions)
        self.assertIn("low_attendance_rate", definitions)
        self.assertIn("missing_submission_rate", definitions)
        self.assertIn("engagement_rate", definitions)
        self.assertIn("course_performance", definitions)


class TestGenerateKPIReportMarkdown(unittest.TestCase):
    """Test markdown report generation."""

    def test_report_generation(self):
        """Test markdown report generation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": [80.0] * 100,
            "assignment_completion_rate": [85.0] * 100,
            "average_exam_score": [75.0] * 100,
        })

        results = [
            calculate_total_students(pd.DataFrame({"student_id": range(1, 101), "course_id": ["C101"] * 100})),
            calculate_average_attendance(df),
            calculate_assignment_completion_rate(df),
            calculate_average_exam_score(df),
        ]

        markdown = generate_kpi_report_markdown(results)

        self.assertIn("# Academic KPI Report", markdown)
        self.assertIn("KPI Results", markdown)
        self.assertIn("KPI Definitions and Limitations", markdown)

    def test_empty_results(self):
        """Test with empty results list."""
        markdown = generate_kpi_report_markdown([])

        self.assertIn("# Academic KPI Report", markdown)
        self.assertIn("0", markdown)


if __name__ == "__main__":
    unittest.main()
