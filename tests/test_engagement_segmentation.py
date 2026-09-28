"""Unit tests for engagement segmentation module."""

import unittest
import numpy as np
import pandas as pd

from src.engagement_segmentation import (
    check_consistent_segment,
    check_strong_performance_segment,
    check_inconsistent_segment,
    check_declining_engagement_segment,
    assign_student_segment,
    calculate_confidence,
    segment_students,
    generate_segmentation_report_markdown,
    get_segment_criteria_summary,
)
from src.exceptions import DataValidationError


class TestCheckConsistentSegment(unittest.TestCase):
    """Test consistent segment checking."""

    def test_meets_all_criteria(self):
        """Test student meeting all consistent criteria."""
        row = pd.Series({
            "attendance_percentage": 80.0,
            "assignment_completion_rate": 80.0,
            "average_exam_score": 70.0,
            "attendance_trend": 5.0,
        })

        is_segment, evidence, metrics, thresholds = check_consistent_segment(row)

        self.assertTrue(is_segment)
        self.assertGreater(len(evidence), 0)
        self.assertIn("attendance", metrics)

    def test_below_thresholds(self):
        """Test student below thresholds."""
        row = pd.Series({
            "attendance_percentage": 60.0,
            "assignment_completion_rate": 60.0,
            "average_exam_score": 50.0,
        })

        is_segment, evidence, metrics, thresholds = check_consistent_segment(row)

        self.assertFalse(is_segment)

    def test_with_missing_data(self):
        """Test with missing data."""
        row = pd.Series({
            "attendance_percentage": 80.0,
            "assignment_completion_rate": np.nan,
            "average_exam_score": np.nan,
        })

        is_segment, evidence, metrics, thresholds = check_consistent_segment(row)

        # Should still evaluate based on available data
        self.assertIsNotNone(is_segment)


class TestCheckStrongPerformanceSegment(unittest.TestCase):
    """Test strong performance segment checking."""

    def test_meets_high_thresholds(self):
        """Test student meeting high thresholds."""
        row = pd.Series({
            "attendance_percentage": 90.0,
            "assignment_completion_rate": 90.0,
            "average_exam_score": 80.0,
        })

        is_segment, evidence, metrics, thresholds = check_strong_performance_segment(row)

        self.assertTrue(is_segment)
        self.assertIn("high threshold", " ".join(evidence).lower())

    def test_below_high_thresholds(self):
        """Test student below high thresholds."""
        row = pd.Series({
            "attendance_percentage": 80.0,
            "assignment_completion_rate": 80.0,
            "average_exam_score": 70.0,
        })

        is_segment, evidence, metrics, thresholds = check_strong_performance_segment(row)

        self.assertFalse(is_segment)

    def test_partial_meeting(self):
        """Test student meeting only some criteria."""
        row = pd.Series({
            "attendance_percentage": 90.0,
            "assignment_completion_rate": 70.0,
            "average_exam_score": 80.0,
        })

        is_segment, evidence, metrics, thresholds = check_strong_performance_segment(row)

        # Must meet ALL criteria for strong performance
        self.assertFalse(is_segment)


class TestCheckInconsistentSegment(unittest.TestCase):
    """Test inconsistent segment checking."""

    def test_high_variance(self):
        """Test student with high variance across metrics."""
        row = pd.Series({
            "attendance_percentage": 90.0,
            "assignment_completion_rate": 50.0,
            "average_exam_score": 85.0,
        })

        is_segment, evidence, metrics, thresholds = check_inconsistent_segment(row)

        self.assertTrue(is_segment)
        self.assertIn("variance", metrics)

    def test_all_low(self):
        """Test student with all low metrics."""
        row = pd.Series({
            "attendance_percentage": 50.0,
            "assignment_completion_rate": 50.0,
            "average_exam_score": 45.0,
        })

        is_segment, evidence, metrics, thresholds = check_inconsistent_segment(row)

        # Not inconsistent if all are low
        self.assertFalse(is_segment)

    def test_all_high(self):
        """Test student with all high metrics."""
        row = pd.Series({
            "attendance_percentage": 90.0,
            "assignment_completion_rate": 90.0,
            "average_exam_score": 85.0,
        })

        is_segment, evidence, metrics, thresholds = check_inconsistent_segment(row)

        # Not inconsistent if all are high
        self.assertFalse(is_segment)


class TestCheckDecliningEngagementSegment(unittest.TestCase):
    """Test declining engagement segment checking."""

    def test_declining_trends(self):
        """Test student with declining trends."""
        row = pd.Series({
            "attendance_trend": -15.0,
            "assignment_trend": -12.0,
            "exam_trend": -8.0,
        })

        is_segment, evidence, metrics, thresholds = check_declining_engagement_segment(row)

        self.assertTrue(is_segment)
        self.assertIn("declining", " ".join(evidence).lower())

    def test_stable_trends(self):
        """Test student with stable trends."""
        row = pd.Series({
            "attendance_trend": 5.0,
            "assignment_trend": 2.0,
            "exam_trend": 3.0,
        })

        is_segment, evidence, metrics, thresholds = check_declining_engagement_segment(row)

        self.assertFalse(is_segment)

    def test_single_declining_trend(self):
        """Test student with only one declining trend."""
        row = pd.Series({
            "attendance_trend": -15.0,
            "assignment_trend": 5.0,
            "exam_trend": 3.0,
        })

        is_segment, evidence, metrics, thresholds = check_declining_engagement_segment(row)

        # Should be declining if at least one trend is declining
        self.assertTrue(is_segment)


class TestAssignStudentSegment(unittest.TestCase):
    """Test student segment assignment."""

    def test_strong_performance_priority(self):
        """Test strong performance takes priority."""
        row = pd.Series({
            "student_id": 1,
            "attendance_percentage": 90.0,
            "assignment_completion_rate": 90.0,
            "average_exam_score": 80.0,
            "attendance_trend": -15.0,  # Also declining, but strong performance takes priority
        })

        segment, evidence, metrics, thresholds = assign_student_segment(row)

        self.assertEqual(segment, "strong_performance")

    def test_declining_priority(self):
        """Test declining engagement priority."""
        row = pd.Series({
            "student_id": 1,
            "attendance_percentage": 70.0,
            "assignment_completion_rate": 70.0,
            "average_exam_score": 65.0,
            "attendance_trend": -15.0,
        })

        segment, evidence, metrics, thresholds = assign_student_segment(row)

        self.assertEqual(segment, "declining_engagement")

    def test_inconsistent_priority(self):
        """Test inconsistent priority."""
        row = pd.Series({
            "student_id": 1,
            "attendance_percentage": 90.0,
            "assignment_completion_rate": 50.0,
            "average_exam_score": 85.0,
        })

        segment, evidence, metrics, thresholds = assign_student_segment(row)

        self.assertEqual(segment, "inconsistent")

    def test_consistent_fallback(self):
        """Test consistent as fallback."""
        row = pd.Series({
            "student_id": 1,
            "attendance_percentage": 80.0,
            "assignment_completion_rate": 80.0,
            "average_exam_score": 70.0,
        })

        segment, evidence, metrics, thresholds = assign_student_segment(row)

        self.assertEqual(segment, "consistent")

    def test_no_segment(self):
        """Test student with insufficient data."""
        row = pd.Series({
            "student_id": 1,
            "attendance_percentage": 50.0,
        })

        segment, evidence, metrics, thresholds = assign_student_segment(row)

        self.assertIsNone(segment)


class TestCalculateConfidence(unittest.TestCase):
    """Test confidence calculation."""

    def test_high_confidence(self):
        """Test high confidence."""
        evidence = [
            "Attendance 80.0% meets threshold (75.0%)",
            "Completion 85.0% meets threshold (75.0%)",
            "Exam score 75.0 meets threshold (60.0)",
        ]

        confidence = calculate_confidence(evidence, 3)

        self.assertEqual(confidence, 1.0)

    def test_low_confidence(self):
        """Test low confidence."""
        evidence = [
            "Attendance 60.0% below threshold (75.0%)",
            "Completion 65.0% below threshold (75.0%)",
        ]

        confidence = calculate_confidence(evidence, 2)

        self.assertEqual(confidence, 0.0)

    def test_mixed_confidence(self):
        """Test mixed confidence."""
        evidence = [
            "Attendance 80.0% meets threshold (75.0%)",
            "Completion 60.0% below threshold (75.0%)",
        ]

        confidence = calculate_confidence(evidence, 2)

        self.assertEqual(confidence, 0.5)

    def test_no_checks(self):
        """Test with no checks."""
        confidence = calculate_confidence([], 0)

        self.assertEqual(confidence, 0.0)


class TestSegmentStudents(unittest.TestCase):
    """Test student segmentation."""

    def test_segmentation(self):
        """Test complete segmentation."""
        df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 95, 100),
            "assignment_completion_rate": np.random.uniform(50, 95, 100),
            "average_exam_score": np.random.uniform(40, 90, 100),
            "attendance_trend": np.random.uniform(-20, 20, 100),
            "assignment_trend": np.random.uniform(-20, 20, 100),
        })

        result = segment_students(df)

        self.assertEqual(result.total_students, 100)
        self.assertGreater(len(result.segment_counts), 0)
        self.assertEqual(len(result.student_evidence), 100 - len(result.unassigned_students))

    def test_missing_student_id(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({
            "attendance_percentage": [80.0, 75.0],
        })

        with self.assertRaises(DataValidationError):
            segment_students(df)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["student_id", "attendance_percentage"])

        result = segment_students(df)

        self.assertEqual(result.total_students, 0)
        self.assertEqual(len(result.segment_counts), 0)


class TestGenerateSegmentationReportMarkdown(unittest.TestCase):
    """Test markdown report generation."""

    def test_report_generation(self):
        """Test markdown report generation."""
        df = pd.DataFrame({
            "student_id": range(1, 51),
            "attendance_percentage": np.random.uniform(50, 95, 50),
            "assignment_completion_rate": np.random.uniform(50, 95, 50),
            "average_exam_score": np.random.uniform(40, 90, 50),
        })

        result = segment_students(df)
        markdown = generate_segmentation_report_markdown(result)

        self.assertIn("# Student Engagement Segmentation Report", markdown)
        self.assertIn("Segment Overview", markdown)
        self.assertIn("Segment Criteria", markdown)

    def test_empty_result(self):
        """Test with empty result."""
        from src.engagement_segmentation import SegmentationResult

        result = SegmentationResult(
            total_students=0,
            segment_counts={},
            segment_percentages={},
            student_evidence=[],
            unassigned_students=[],
            analysis_date="2026-01-01",
        )

        markdown = generate_segmentation_report_markdown(result)

        self.assertIn("# Student Engagement Segmentation Report", markdown)


class TestGetSegmentCriteriaSummary(unittest.TestCase):
    """Test segment criteria summary."""

    def test_criteria_summary(self):
        """Test getting criteria summary."""
        criteria = get_segment_criteria_summary()

        self.assertIn("consistent", criteria)
        self.assertIn("strong_performance", criteria)
        self.assertIn("inconsistent", criteria)
        self.assertIn("declining_engagement", criteria)

        # Check that criteria have required fields
        for segment, segment_criteria in criteria.items():
            self.assertIn("description", segment_criteria)


if __name__ == "__main__":
    unittest.main()
