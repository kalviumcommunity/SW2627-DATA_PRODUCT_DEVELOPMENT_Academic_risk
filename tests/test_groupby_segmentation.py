"""Unit tests for groupby segmentation module."""

import unittest
import numpy as np
import pandas as pd

from src.groupby_segmentation import (
    calculate_segment_metrics,
    segment_by_program,
    segment_by_year,
    segment_by_course,
    segment_by_risk_level,
    segment_by_engagement,
    create_engagement_segments,
    multi_level_segmentation,
    generate_segmentation_markdown,
    compare_segments,
)
from src.exceptions import DataValidationError


class TestCalculateSegmentMetrics(unittest.TestCase):
    """Test segment metrics calculation."""

    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "attendance_percentage": [80.0, 75.0, 70.0],
            "assignment_completion_rate": [90.0, 85.0, 80.0],
            "average_exam_score": [85.0, 80.0, 75.0],
        })

        metrics = calculate_segment_metrics(df, "test", "value1")

        self.assertEqual(metrics.segment_name, "test")
        self.assertEqual(metrics.segment_value, "value1")
        self.assertEqual(metrics.student_count, 3)
        self.assertIsNotNone(metrics.avg_attendance)
        self.assertIsNotNone(metrics.avg_completion)
        self.assertIsNotNone(metrics.avg_exam_score)

    def test_with_missing_values(self):
        """Test metrics with missing values."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "attendance_percentage": [80.0, np.nan, 70.0],
            "assignment_completion_rate": [90.0, 85.0, np.nan],
            "average_exam_score": [np.nan, 80.0, 75.0],
        })

        metrics = calculate_segment_metrics(df, "test", "value1")

        self.assertEqual(metrics.student_count, 3)
        self.assertEqual(metrics.missing_attendance_count, 1)
        self.assertEqual(metrics.missing_completion_count, 1)
        self.assertEqual(metrics.missing_exam_count, 1)

    def test_review_calculation(self):
        """Test students requiring review calculation."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4],
            "attendance_percentage": [80.0, 60.0, 75.0, 50.0],
            "assignment_completion_rate": [90.0, 85.0, 60.0, 70.0],
            "average_exam_score": [85.0, 80.0, 40.0, 75.0],
        })

        metrics = calculate_segment_metrics(
            df,
            "test",
            "value1",
            review_threshold_attendance=70.0,
            review_threshold_completion=70.0,
            review_threshold_exam=50.0,
        )

        # Students 2, 3, 4 should require review
        self.assertEqual(metrics.students_requiring_review, 3)
        self.assertEqual(metrics.review_percentage, 75.0)


class TestSegmentByProgram(unittest.TestCase):
    """Test program-based segmentation."""

    def test_program_segmentation(self):
        """Test segmentation by program."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "program": ["CS"] * 50 + ["DS"] * 30 + ["IT"] * 20,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = segment_by_program(df)

        self.assertEqual(result.segment_by, "program")
        self.assertEqual(result.total_students, 100)
        self.assertEqual(result.segment_count, 3)
        self.assertEqual(len(result.segments), 3)

    def test_missing_program_column(self):
        """Test with missing program column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            segment_by_program(df)


class TestSegmentByYear(unittest.TestCase):
    """Test year-based segmentation."""

    def test_year_segmentation(self):
        """Test segmentation by year."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "year": [1] * 30 + [2] * 35 + [3] * 25 + [4] * 10,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = segment_by_year(df)

        self.assertEqual(result.segment_by, "year")
        self.assertEqual(result.total_students, 100)
        self.assertEqual(result.segment_count, 4)

    def test_missing_year_column(self):
        """Test with missing year column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            segment_by_year(df)


class TestSegmentByCourse(unittest.TestCase):
    """Test course-based segmentation."""

    def test_course_segmentation(self):
        """Test segmentation by course."""
        student_df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        enrollments_df = pd.DataFrame({
            "student_id": range(1, 101),
            "course_id": ["C101"] * 40 + ["C102"] * 35 + ["C103"] * 25,
        })

        result = segment_by_course(student_df, enrollments_df)

        self.assertEqual(result.segment_by, "course")
        self.assertEqual(result.total_students, 100)
        self.assertEqual(result.segment_count, 3)

    def test_course_with_course_names(self):
        """Test segmentation with course names."""
        student_df = pd.DataFrame({
            "student_id": range(1, 51),
            "attendance_percentage": np.random.uniform(50, 100, 50),
        })

        enrollments_df = pd.DataFrame({
            "student_id": range(1, 51),
            "course_id": ["C101"] * 25 + ["C102"] * 25,
        })

        courses_df = pd.DataFrame({
            "course_id": ["C101", "C102"],
            "course_name": ["Intro to CS", "Data Structures"],
        })

        result = segment_by_course(student_df, enrollments_df, courses_df)

        self.assertEqual(result.segment_count, 2)

    def test_missing_student_id(self):
        """Test with missing student_id column."""
        student_df = pd.DataFrame({"attendance": [80, 75]})
        enrollments_df = pd.DataFrame({"student_id": [1, 2], "course_id": ["C101", "C102"]})

        with self.assertRaises(DataValidationError):
            segment_by_course(student_df, enrollments_df)


class TestSegmentByRiskLevel(unittest.TestCase):
    """Test risk level segmentation."""

    def test_risk_segmentation(self):
        """Test segmentation by risk level."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "risk_level": ["Low"] * 50 + ["Moderate"] * 30 + ["High"] * 20,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = segment_by_risk_level(df)

        self.assertEqual(result.segment_by, "risk_level")
        self.assertEqual(result.total_students, 100)
        self.assertEqual(result.segment_count, 3)

    def test_custom_risk_column(self):
        """Test with custom risk column."""
        df = pd.DataFrame({
            "student_id": range(1, 51),
            "custom_risk": ["A"] * 25 + ["B"] * 25,
            "attendance_percentage": np.random.uniform(50, 100, 50),
        })

        result = segment_by_risk_level(df, risk_column="custom_risk")

        self.assertEqual(result.segment_count, 2)

    def test_missing_risk_column(self):
        """Test with missing risk column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            segment_by_risk_level(df)


class TestSegmentByEngagement(unittest.TestCase):
    """Test engagement segment segmentation."""

    def test_engagement_segmentation(self):
        """Test segmentation by engagement segment."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "engagement_segment": ["High"] * 40 + ["Medium"] * 35 + ["Low"] * 25,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = segment_by_engagement(df)

        self.assertEqual(result.segment_by, "engagement_segment")
        self.assertEqual(result.total_students, 100)
        self.assertEqual(result.segment_count, 3)

    def test_missing_engagement_column(self):
        """Test with missing engagement column."""
        df = pd.DataFrame({"student_id": range(1, 101)})

        with self.assertRaises(DataValidationError):
            segment_by_engagement(df)


class TestCreateEngagementSegments(unittest.TestCase):
    """Test engagement segment creation."""

    def test_engagement_classification(self):
        """Test engagement classification."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4],
            "attendance_percentage": [80.0, 80.0, 60.0, 60.0],
            "assignment_completion_rate": [80.0, 60.0, 80.0, 60.0],
        })

        result = create_engagement_segments(df)

        self.assertIn("engagement_segment", result.columns)
        self.assertEqual(result.loc[0, "engagement_segment"], "Highly Engaged")
        self.assertEqual(result.loc[1, "engagement_segment"], "Attender (Low Completion)")
        self.assertEqual(result.loc[2, "engagement_segment"], "Completer (Low Attendance)")
        self.assertEqual(result.loc[3, "engagement_segment"], "Disengaged")

    def test_with_missing_values(self):
        """Test with missing values."""
        df = pd.DataFrame({
            "student_id": [1, 2],
            "attendance_percentage": [80.0, np.nan],
            "assignment_completion_rate": [80.0, 60.0],
        })

        result = create_engagement_segments(df)

        self.assertIn("engagement_segment", result.columns)
        self.assertEqual(result.loc[0, "engagement_segment"], "Highly Engaged")
        self.assertEqual(result.loc[1, "engagement_segment"], "Unknown")


class TestMultiLevelSegmentation(unittest.TestCase):
    """Test multi-level segmentation."""

    def test_program_year_segmentation(self):
        """Test program × year segmentation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "program": ["CS"] * 50 + ["DS"] * 50,
            "year": [1] * 25 + [2] * 25 + [1] * 25 + [2] * 25,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        results = multi_level_segmentation(df, primary_segment="program", secondary_segment="year")

        self.assertEqual(len(results), 2)  # 2 programs
        self.assertIn("CS", results)
        self.assertIn("DS", results)

    def test_missing_primary_column(self):
        """Test with missing primary column."""
        df = pd.DataFrame({"year": [1, 2]})

        with self.assertRaises(DataValidationError):
            multi_level_segmentation(df, primary_segment="program", secondary_segment="year")

    def test_missing_secondary_column(self):
        """Test with missing secondary column."""
        df = pd.DataFrame({"program": ["CS", "DS"]})

        with self.assertRaises(DataValidationError):
            multi_level_segmentation(df, primary_segment="program", secondary_segment="year")


class TestGenerateSegmentationMarkdown(unittest.TestCase):
    """Test markdown generation."""

    def test_markdown_generation(self):
        """Test markdown report generation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "program": ["CS"] * 50 + ["DS"] * 50,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = segment_by_program(df)
        markdown = generate_segmentation_markdown(result)

        self.assertIn("# Academic Segmentation Analysis", markdown)
        self.assertIn("Segment By", markdown)
        self.assertIn("Total Students", markdown)
        self.assertIn("Segment Details", markdown)

    def test_empty_segments(self):
        """Test with empty segments."""
        from src.groupby_segmentation import SegmentationResult, SegmentMetrics

        result = SegmentationResult(
            segment_by="test",
            total_students=0,
            segments=[],
            segment_count=0,
            analysis_date="2026-01-01",
        )

        markdown = generate_segmentation_markdown(result)

        self.assertIn("# Academic Segmentation Analysis", markdown)


class TestCompareSegments(unittest.TestCase):
    """Test segment comparison."""

    def test_compare_by_attendance(self):
        """Test comparison by attendance."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "program": ["CS"] * 50 + ["DS"] * 50,
            "attendance_percentage": np.random.uniform(50, 100, 100),
        })

        result = segment_by_program(df)
        comparison = compare_segments(result, metric="avg_attendance")

        self.assertIn("segment_value", comparison.columns)
        self.assertIn("student_count", comparison.columns)
        self.assertIn("avg_attendance", comparison.columns)

    def test_compare_by_review_percentage(self):
        """Test comparison by review percentage."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "program": ["CS"] * 50 + ["DS"] * 50,
            "attendance_percentage": np.random.uniform(50, 100, 100),
            "assignment_completion_rate": np.random.uniform(40, 100, 100),
            "average_exam_score": np.random.uniform(40, 95, 100),
        })

        result = segment_by_program(df)
        comparison = compare_segments(result, metric="review_percentage")

        self.assertIn("review_percentage", comparison.columns)

    def test_invalid_metric(self):
        """Test with invalid metric."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "program": ["CS"] * 50 + ["DS"] * 50,
            "attendance_percentage": np.random.uniform(50, 100, 100),
        })

        result = segment_by_program(df)

        with self.assertRaises(DataValidationError):
            compare_segments(result, metric="invalid_metric")


if __name__ == "__main__":
    unittest.main()
