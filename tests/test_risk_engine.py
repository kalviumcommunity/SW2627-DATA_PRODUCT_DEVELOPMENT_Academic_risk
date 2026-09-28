"""Comprehensive unit tests for risk engine module."""

import unittest
import numpy as np
import pandas as pd

from src.risk_engine import (
    check_attendance_risk,
    check_completion_risk,
    check_exam_risk,
    check_declining_trend_risk,
    check_submission_risk,
    determine_risk_level,
    assess_student_risk,
    batch_assess_risk,
    get_risk_thresholds,
    get_risk_level_descriptions,
    generate_risk_report_markdown,
)
from src.exceptions import DataValidationError


class TestCheckAttendanceRisk(unittest.TestCase):
    """Test attendance risk checking."""

    def test_critical_attendance(self):
        """Test critically low attendance."""
        signal = check_attendance_risk(45.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "low_attendance")
        self.assertEqual(signal.severity, "high")

    def test_low_attendance(self):
        """Test low attendance (below threshold but not critical)."""
        signal = check_attendance_risk(60.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "low_attendance")
        self.assertEqual(signal.severity, "moderate")

    def test_normal_attendance(self):
        """Test normal attendance (no risk)."""
        signal = check_attendance_risk(80.0)

        self.assertIsNone(signal)

    def test_none_attendance(self):
        """Test with None attendance."""
        signal = check_attendance_risk(None)

        self.assertIsNone(signal)

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        custom_thresholds = {"attendance_low_threshold": 85.0, "attendance_critical_threshold": 60.0}
        signal = check_attendance_risk(55.0, custom_thresholds)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.severity, "high")


class TestCheckCompletionRisk(unittest.TestCase):
    """Test completion risk checking."""

    def test_critical_completion(self):
        """Test critically low completion."""
        signal = check_completion_risk(40.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "low_completion")
        self.assertEqual(signal.severity, "high")

    def test_low_completion(self):
        """Test low completion."""
        signal = check_completion_risk(60.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.severity, "moderate")

    def test_normal_completion(self):
        """Test normal completion."""
        signal = check_completion_risk(85.0)

        self.assertIsNone(signal)


class TestCheckExamRisk(unittest.TestCase):
    """Test exam risk checking."""

    def test_critical_exam(self):
        """Test critically low exam score."""
        signal = check_exam_risk(35.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "low_exam_score")
        self.assertEqual(signal.severity, "high")

    def test_low_exam(self):
        """Test low exam score."""
        signal = check_exam_risk(55.0)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.severity, "moderate")

    def test_normal_exam(self):
        """Test normal exam score."""
        signal = check_exam_risk(75.0)

        self.assertIsNone(signal)


class TestCheckDecliningTrendRisk(unittest.TestCase):
    """Test declining trend risk checking."""

    def test_critical_decline(self):
        """Test critical decline."""
        signal = check_declining_trend_risk(-25.0, "attendance")

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "declining_performance")
        self.assertEqual(signal.severity, "high")

    def test_moderate_decline(self):
        """Test moderate decline."""
        signal = check_declining_trend_risk(-15.0, "attendance")

        self.assertIsNotNone(signal)
        self.assertEqual(signal.severity, "moderate")

    def test_stable_trend(self):
        """Test stable trend."""
        signal = check_declining_trend_risk(-5.0, "attendance")

        self.assertIsNone(signal)

    def test_improving_trend(self):
        """Test improving trend."""
        signal = check_declining_trend_risk(10.0, "attendance")

        self.assertIsNone(signal)


class TestCheckSubmissionRisk(unittest.TestCase):
    """Test submission risk checking."""

    def test_missing_submissions(self):
        """Test missing submissions."""
        signals = check_submission_risk(missing_count=5)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "missing_submissions")

    def test_late_submissions(self):
        """Test late submissions."""
        signals = check_submission_risk(late_count=5)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "late_submissions")

    def test_high_late_rate(self):
        """Test high late submission rate."""
        signals = check_submission_risk(late_rate=0.5)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "high_late_rate")

    def test_multiple_submission_risks(self):
        """Test multiple submission risks."""
        signals = check_submission_risk(missing_count=4, late_count=4, late_rate=0.4)

        self.assertEqual(len(signals), 3)

    def test_no_submission_risks(self):
        """Test no submission risks."""
        signals = check_submission_risk(missing_count=1, late_count=1, late_rate=0.1)

        self.assertEqual(len(signals), 0)


class TestDetermineRiskLevel(unittest.TestCase):
    """Test risk level determination."""

    def test_elevated_high_severity(self):
        """Test elevated with multiple high severity signals."""
        level = determine_risk_level(3, 2)

        self.assertEqual(level, "ELEVATED")

    def test_elevated_many_signals(self):
        """Test elevated with many signals."""
        level = determine_risk_level(5, 1)

        self.assertEqual(level, "ELEVATED")

    def test_moderate(self):
        """Test moderate risk."""
        level = determine_risk_level(2, 0)

        self.assertEqual(level, "MODERATE")

    def test_low(self):
        """Test low risk."""
        level = determine_risk_level(0, 0)

        self.assertEqual(level, "LOW")

    def test_single_signal_moderate(self):
        """Test single signal results in moderate."""
        level = determine_risk_level(1, 0)

        self.assertEqual(level, "MODERATE")


class TestAssessStudentRisk(unittest.TestCase):
    """Test complete student risk assessment."""

    def test_low_risk_student(self):
        """Test assessment for low-risk student."""
        assessment = assess_student_risk(
            student_id=1,
            attendance_percentage=85.0,
            completion_rate=90.0,
            exam_score=80.0,
        )

        self.assertEqual(assessment.risk_level, "LOW")
        self.assertEqual(assessment.risk_score, 0)
        self.assertIn("No concerning signals detected", assessment.reasons[0])
        self.assertIsNotNone(assessment.disclaimer)

    def test_moderate_risk_student(self):
        """Test assessment for moderate-risk student."""
        assessment = assess_student_risk(
            student_id=2,
            attendance_percentage=65.0,
            completion_rate=85.0,
            exam_score=75.0,
        )

        self.assertEqual(assessment.risk_level, "MODERATE")
        self.assertGreater(assessment.risk_score, 0)
        self.assertGreater(len(assessment.reasons), 0)

    def test_elevated_risk_student(self):
        """Test assessment for elevated-risk student."""
        assessment = assess_student_risk(
            student_id=3,
            attendance_percentage=45.0,
            completion_rate=40.0,
            exam_score=35.0,
            attendance_trend=-25.0,
            missing_submissions=5,
        )

        self.assertEqual(assessment.risk_level, "ELEVATED")
        self.assertGreater(assessment.risk_score, 3)
        self.assertGreater(len(assessment.reasons), 0)
        self.assertGreater(len(assessment.recommendations), 0)

    def test_declining_trend_detection(self):
        """Test declining trend detection."""
        assessment = assess_student_risk(
            student_id=4,
            attendance_percentage=75.0,
            completion_rate=75.0,
            exam_score=70.0,
            attendance_trend=-15.0,
            assignment_trend=-12.0,
        )

        self.assertEqual(assessment.risk_level, "MODERATE")
        self.assertTrue(any("declining" in reason.lower() for reason in assessment.reasons))

    def test_submission_risk_detection(self):
        """Test submission risk detection."""
        assessment = assess_student_risk(
            student_id=5,
            attendance_percentage=80.0,
            completion_rate=75.0,
            exam_score=75.0,
            missing_submissions=4,
            late_submissions=5,
        )

        self.assertEqual(assessment.risk_level, "MODERATE")
        self.assertTrue(any("submission" in reason.lower() for reason in assessment.reasons))

    def test_disclaimer_present(self):
        """Test that disclaimer is always present."""
        assessment = assess_student_risk(student_id=1)

        self.assertIsNotNone(assessment.disclaimer)
        self.assertIn("academic-support indicator", assessment.disclaimer.lower())
        self.assertIn("not a definitive prediction", assessment.disclaimer.lower())

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        custom_thresholds = {
            "attendance_low_threshold": 90.0,
            "attendance_critical_threshold": 70.0,
            "completion_low_threshold": 90.0,
            "completion_critical_threshold": 70.0,
            "exam_low_threshold": 70.0,
            "exam_critical_threshold": 50.0,
            "trend_decline_threshold": -5.0,
            "trend_critical_decline": -15.0,
            "missing_submission_threshold": 2,
            "late_submission_threshold": 2,
            "late_submission_rate_threshold": 0.2,
        }

        assessment = assess_student_risk(
            student_id=6,
            attendance_percentage=65.0,  # Below critical threshold
            completion_rate=65.0,  # Below critical threshold
            exam_score=45.0,  # Below critical threshold
            thresholds=custom_thresholds,
        )

        # Should be elevated due to stricter thresholds and multiple critical signals
        self.assertEqual(assessment.risk_level, "ELEVATED")


class TestBatchAssessRisk(unittest.TestCase):
    """Test batch risk assessment."""

    def test_batch_assessment(self):
        """Test batch assessment for multiple students."""
        df = pd.DataFrame({
            "student_id": range(1, 11),
            "attendance_percentage": [85.0, 65.0, 45.0, 80.0, 70.0, 60.0, 50.0, 75.0, 55.0, 40.0],
            "assignment_completion_rate": [90.0, 85.0, 40.0, 80.0, 75.0, 65.0, 55.0, 70.0, 60.0, 45.0],
            "average_exam_score": [80.0, 75.0, 35.0, 78.0, 72.0, 65.0, 45.0, 70.0, 55.0, 40.0],
        })

        assessments = batch_assess_risk(df)

        self.assertEqual(len(assessments), 10)
        self.assertTrue(all(a.student_id is not None for a in assessments))
        self.assertTrue(all(a.disclaimer is not None for a in assessments))

    def test_missing_student_id_column(self):
        """Test with missing student_id column."""
        df = pd.DataFrame({
            "attendance_percentage": [80.0, 75.0],
        })

        with self.assertRaises(DataValidationError):
            batch_assess_risk(df)


class TestGetRiskThresholds(unittest.TestCase):
    """Test risk threshold retrieval."""

    def test_get_thresholds(self):
        """Test getting risk thresholds."""
        thresholds = get_risk_thresholds()

        self.assertIn("attendance_low_threshold", thresholds)
        self.assertIn("completion_low_threshold", thresholds)
        self.assertIn("exam_low_threshold", thresholds)
        self.assertIn("trend_decline_threshold", thresholds)
        self.assertIn("missing_submission_threshold", thresholds)


class TestGetRiskLevelDescriptions(unittest.TestCase):
    """Test risk level description retrieval."""

    def test_get_descriptions(self):
        """Test getting risk level descriptions."""
        descriptions = get_risk_level_descriptions()

        self.assertIn("LOW", descriptions)
        self.assertIn("MODERATE", descriptions)
        self.assertIn("ELEVATED", descriptions)

        # Check structure
        for level, info in descriptions.items():
            self.assertIn("description", info)
            self.assertIn("signal_count_range", info)
            self.assertIn("color", info)


class TestGenerateRiskReportMarkdown(unittest.TestCase):
    """Test markdown report generation."""

    def test_report_generation(self):
        """Test markdown report generation."""
        assessment = assess_student_risk(
            student_id=1,
            attendance_percentage=65.0,
            completion_rate=60.0,
            exam_score=55.0,
        )

        markdown = generate_risk_report_markdown(assessment)

        self.assertIn("# Academic Risk Assessment Report", markdown)
        self.assertIn("Student ID", markdown)
        self.assertIn("Risk Level", markdown)
        self.assertIn("Risk Signals", markdown)
        self.assertIn("Summary", markdown)
        self.assertIn("IMPORTANT", markdown)

    def test_report_with_no_signals(self):
        """Test report with no signals."""
        assessment = assess_student_risk(student_id=1, attendance_percentage=85.0)

        markdown = generate_risk_report_markdown(assessment)

        self.assertIn("No concerning signals detected", markdown)


if __name__ == "__main__":
    unittest.main()
