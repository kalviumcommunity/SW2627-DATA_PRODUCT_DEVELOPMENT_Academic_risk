"""Unit tests for funnel analysis module."""

import unittest
import numpy as np
import pandas as pd

from src.funnel_analysis import (
    count_enrolled_students,
    count_active_students,
    count_assignment_participants,
    count_assessment_participants,
    count_consistently_engaged,
    calculate_drop_off_rate,
    identify_engagement_gaps,
    analyze_academic_funnel,
    generate_funnel_report_markdown,
    get_funnel_stage_definitions,
)
from src.exceptions import DataValidationError


class TestCountEnrolledStudents(unittest.TestCase):
    """Test enrolled student counting."""

    def test_count_enrolled(self):
        """Test counting enrolled students."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4, 5],
            "course_id": ["C101"] * 5,
        })

        count = count_enrolled_students(df)

        self.assertEqual(count, 5)

    def test_count_with_duplicates(self):
        """Test counting with duplicate enrollments."""
        df = pd.DataFrame({
            "student_id": [1, 1, 2, 3, 3, 3],
            "course_id": ["C101", "C102", "C101", "C101", "C102", "C103"],
        })

        count = count_enrolled_students(df)

        self.assertEqual(count, 3)

    def test_missing_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({"course_id": ["C101", "C102"]})

        with self.assertRaises(DataValidationError):
            count_enrolled_students(df)


class TestCountActiveStudents(unittest.TestCase):
    """Test active student counting."""

    def test_count_active(self):
        """Test counting active students."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4, 5],
            "date": pd.date_range("2024-01-01", periods=5),
        })

        count = count_active_students(df)

        self.assertEqual(count, 5)

    def test_count_with_threshold(self):
        """Test counting with minimum records threshold."""
        df = pd.DataFrame({
            "student_id": [1, 1, 2, 3, 4, 5],
            "date": pd.date_range("2024-01-01", periods=6),
        })

        count = count_active_students(df, min_attendance_records=2)

        self.assertEqual(count, 1)  # Only student 1 has 2+ records

    def test_missing_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3)})

        with self.assertRaises(DataValidationError):
            count_active_students(df)


class TestCountAssignmentParticipants(unittest.TestCase):
    """Test assignment participant counting."""

    def test_count_participants(self):
        """Test counting assignment participants."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4, 5],
            "assignment_id": ["A1", "A2", "A3", "A4", "A5"],
        })

        count = count_assignment_participants(df)

        self.assertEqual(count, 5)

    def test_count_with_threshold(self):
        """Test counting with minimum submissions threshold."""
        df = pd.DataFrame({
            "student_id": [1, 1, 2, 3, 4, 5],
            "assignment_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
        })

        count = count_assignment_participants(df, min_submissions=2)

        self.assertEqual(count, 1)

    def test_missing_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({"assignment_id": ["A1", "A2"]})

        with self.assertRaises(DataValidationError):
            count_assignment_participants(df)


class TestCountAssessmentParticipants(unittest.TestCase):
    """Test assessment participant counting."""

    def test_count_participants(self):
        """Test counting assessment participants."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4, 5],
            "exam_id": ["E1", "E2", "E3", "E4", "E5"],
        })

        count = count_assessment_participants(df)

        self.assertEqual(count, 5)

    def test_count_with_threshold(self):
        """Test counting with minimum exams threshold."""
        df = pd.DataFrame({
            "student_id": [1, 1, 2, 3, 4, 5],
            "exam_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        })

        count = count_assessment_participants(df, min_exams=2)

        self.assertEqual(count, 1)

    def test_missing_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({"exam_id": ["E1", "E2"]})

        with self.assertRaises(DataValidationError):
            count_assessment_participants(df)


class TestCountConsistentlyEngaged(unittest.TestCase):
    """Test consistently engaged counting."""

    def test_count_consistent(self):
        """Test counting consistently engaged students."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3, 4, 5],
            "attendance_percentage": [80.0, 75.0, 70.0, 65.0, 60.0],
            "assignment_completion_rate": [85.0, 80.0, 75.0, 70.0, 65.0],
        })

        count = count_consistently_engaged(df)

        self.assertEqual(count, 3)  # Students 1, 2, 3 meet thresholds

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "attendance_percentage": [80.0, 75.0, 70.0],
            "assignment_completion_rate": [85.0, 80.0, 75.0],
        })

        count = count_consistently_engaged(df, attendance_threshold=80.0, completion_threshold=80.0)

        self.assertEqual(count, 1)  # Only student 1 meets higher thresholds

    def test_missing_columns(self):
        """Test with missing columns."""
        df = pd.DataFrame({
            "student_id": [1, 2, 3],
            "attendance_percentage": [80.0, 75.0, 70.0],
        })

        with self.assertRaises(DataValidationError):
            count_consistently_engaged(df)


class TestCalculateDropOffRate(unittest.TestCase):
    """Test drop-off rate calculation."""

    def test_no_drop_off(self):
        """Test with no drop-off."""
        rate = calculate_drop_off_rate(100, 100)

        self.assertEqual(rate, 0.0)

    def test_partial_drop_off(self):
        """Test with partial drop-off."""
        rate = calculate_drop_off_rate(80, 100)

        self.assertEqual(rate, 20.0)

    def test_complete_drop_off(self):
        """Test with complete drop-off."""
        rate = calculate_drop_off_rate(0, 100)

        self.assertEqual(rate, 100.0)

    def test_zero_previous(self):
        """Test with zero previous count."""
        rate = calculate_drop_off_rate(50, 0)

        self.assertEqual(rate, 0.0)


class TestIdentifyEngagementGaps(unittest.TestCase):
    """Test engagement gap identification."""

    def test_identify_gaps(self):
        """Test gap identification."""
        from src.funnel_analysis import FunnelStage

        stages = [
            FunnelStage("enrolled", 100, 100.0, None, None, "All enrolled"),
            FunnelStage("active", 80, 80.0, 80.0, 20.0, "Active students"),
            FunnelStage("assignment_participation", 50, 50.0, 62.5, 37.5, "Assignment participants"),
            FunnelStage("assessment_participation", 45, 45.0, 90.0, 10.0, "Assessment participants"),
            FunnelStage("consistent_engagement", 40, 40.0, 88.9, 11.1, "Consistent engagement"),
        ]

        gaps = identify_engagement_gaps(stages, drop_off_threshold=15.0)

        self.assertEqual(len(gaps), 2)  # Both active and assignment_participation have >15% drop-off

    def test_no_gaps(self):
        """Test with no significant gaps."""
        from src.funnel_analysis import FunnelStage

        stages = [
            FunnelStage("enrolled", 100, 100.0, None, None, "All enrolled"),
            FunnelStage("active", 95, 95.0, 95.0, 5.0, "Active students"),
            FunnelStage("assignment_participation", 90, 90.0, 94.7, 5.3, "Assignment participants"),
        ]

        gaps = identify_engagement_gaps(stages, drop_off_threshold=20.0)

        self.assertEqual(len(gaps), 0)


class TestAnalyzeAcademicFunnel(unittest.TestCase):
    """Test complete funnel analysis."""

    def test_funnel_analysis(self):
        """Test complete funnel analysis."""
        enrollments_df = pd.DataFrame({
            "student_id": range(1, 101),
            "course_id": ["C101"] * 100,
        })

        attendance_df = pd.DataFrame({
            "student_id": list(range(1, 91)) + [1, 2, 3],  # 90 unique, some duplicates
            "date": pd.date_range("2024-01-01", periods=93),
        })

        submissions_df = pd.DataFrame({
            "student_id": list(range(1, 81)) + [1, 2],  # 80 unique
            "assignment_id": ["A1"] * 82,
        })

        exams_df = pd.DataFrame({
            "student_id": list(range(1, 71)) + [1],  # 70 unique
            "exam_id": ["E1"] * 71,
        })

        features_df = pd.DataFrame({
            "student_id": range(1, 101),
            "attendance_percentage": np.random.uniform(50, 95, 100),
            "assignment_completion_rate": np.random.uniform(50, 95, 100),
        })

        result = analyze_academic_funnel(
            enrollments_df,
            attendance_df,
            submissions_df,
            exams_df,
            features_df,
        )

        self.assertEqual(result.total_enrolled, 100)
        self.assertEqual(len(result.stages), 5)
        self.assertIsNotNone(result.overall_drop_off_rate)
        self.assertGreaterEqual(result.largest_drop_off_rate, 0.0)

    def test_empty_dataframes(self):
        """Test with empty DataFrames."""
        enrollments_df = pd.DataFrame(columns=["student_id", "course_id"])
        attendance_df = pd.DataFrame(columns=["student_id", "date"])
        submissions_df = pd.DataFrame(columns=["student_id", "assignment_id"])
        exams_df = pd.DataFrame(columns=["student_id", "exam_id"])
        features_df = pd.DataFrame(columns=["student_id", "attendance_percentage", "assignment_completion_rate"])

        result = analyze_academic_funnel(
            enrollments_df,
            attendance_df,
            submissions_df,
            exams_df,
            features_df,
        )

        self.assertEqual(result.total_enrolled, 0)
        self.assertEqual(len(result.stages), 5)


class TestGenerateFunnelReportMarkdown(unittest.TestCase):
    """Test markdown report generation."""

    def test_report_generation(self):
        """Test markdown report generation."""
        enrollments_df = pd.DataFrame({
            "student_id": range(1, 51),
            "course_id": ["C101"] * 50,
        })

        attendance_df = pd.DataFrame({
            "student_id": range(1, 46),
            "date": pd.date_range("2024-01-01", periods=45),
        })

        submissions_df = pd.DataFrame({
            "student_id": range(1, 41),
            "assignment_id": ["A1"] * 40,
        })

        exams_df = pd.DataFrame({
            "student_id": range(1, 36),
            "exam_id": ["E1"] * 35,
        })

        features_df = pd.DataFrame({
            "student_id": range(1, 51),
            "attendance_percentage": np.random.uniform(50, 95, 50),
            "assignment_completion_rate": np.random.uniform(50, 95, 50),
        })

        result = analyze_academic_funnel(
            enrollments_df,
            attendance_df,
            submissions_df,
            exams_df,
            features_df,
        )
        markdown = generate_funnel_report_markdown(result)

        self.assertIn("# Academic Engagement Funnel Analysis", markdown)
        self.assertIn("Funnel Stages", markdown)
        self.assertIn("Key Insights", markdown)


class TestGetFunnelStageDefinitions(unittest.TestCase):
    """Test funnel stage definitions."""

    def test_stage_definitions(self):
        """Test getting stage definitions."""
        definitions = get_funnel_stage_definitions()

        self.assertIn("enrolled", definitions)
        self.assertIn("active", definitions)
        self.assertIn("assignment_participation", definitions)
        self.assertIn("assessment_participation", definitions)
        self.assertIn("consistent_engagement", definitions)

        # Check that definitions are strings
        for stage, definition in definitions.items():
            self.assertIsInstance(definition, str)


if __name__ == "__main__":
    unittest.main()
