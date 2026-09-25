"""Tests for domain-aware academic missing value detection and handling."""

import unittest
import numpy as np
import pandas as pd

from src.missing_values import (
    MissingValueReport,
    detect_missing_values,
    handle_academic_missing_values,
    handle_attendance_missing_values,
    handle_demographics_missing_values,
    handle_exams_missing_values,
    handle_submissions_missing_values,
)


class TestMissingValues(unittest.TestCase):
    """Test suite ensuring missing data is handled according to academic domain rules."""

    def setUp(self):
        """Prepare academic test fixtures with realistic missingness patterns."""
        self.attendance_df = pd.DataFrame({
            "student_id": ["STU-1", "STU-2", "STU-3"],
            "course_id": ["CS101", "CS101", "CS101"],
            "date": ["2026-02-01", "2026-02-02", "2026-02-03"],
            "status": ["Present", np.nan, "Absent"],  # Row 1 status missing
        })

        self.exams_df = pd.DataFrame({
            "exam_id": ["EX-1", "EX-2", "EX-3"],
            "student_id": ["STU-1", "STU-2", "STU-3"],
            "course_id": ["CS101", "CS101", "CS101"],
            "exam_type": ["Midterm", "Midterm", "Midterm"],
            "score": [85.0, np.nan, 72.0],  # Row 1 score missing
        })

        self.submissions_df = pd.DataFrame({
            "assignment_id": ["ASN-1", "ASN-2", "ASN-3"],
            "student_id": ["STU-1", "STU-2", "STU-3"],
            "submission_date": ["2026-02-10", np.nan, "2026-02-11"],  # Row 1 unsubmitted
            "score": [90.0, np.nan, np.nan],  # Row 2 submitted but ungraded
        })

        self.students_df = pd.DataFrame({
            "student_id": ["STU-1", "STU-2"],
            "name": ["Aarav", "Priya"],
            "program": ["Computer Science", np.nan],  # Row 1 missing program
            "year": [2, 3],
        })

    def test_detect_missing_values(self):
        """Verify detect_missing_values calculates counts and percentages accurately."""
        diag = detect_missing_values(self.attendance_df, entity_name="attendance")
        self.assertEqual(diag["total_rows"], 3)
        self.assertEqual(diag["missing_counts"]["status"], 1)
        self.assertEqual(diag["rows_with_any_missing"], 1)
        self.assertAlmostEqual(diag["missing_percentages"]["status"], 33.33, places=1)

    def test_attendance_missing_values_not_converted_to_zero_or_absent(self):
        """Verify missing attendance is NOT converted to 0 or Absent, but marked Unrecorded."""
        processed, report = handle_attendance_missing_values(self.attendance_df)

        # Crucial academic rule check: Never convert missing attendance to 0 or Absent
        self.assertNotIn(0, processed["status"].values)
        self.assertNotIn(0.0, processed["status"].values)
        self.assertEqual(processed.iloc[1]["status"], "Unrecorded")

        # Indicator column preserved
        self.assertIn("attendance_recorded", processed.columns)
        self.assertTrue(processed.iloc[0]["attendance_recorded"])
        self.assertFalse(processed.iloc[1]["attendance_recorded"])
        self.assertTrue(processed.iloc[2]["attendance_recorded"])

        self.assertIsInstance(report, MissingValueReport)
        self.assertEqual(report.before_missing_counts["status"], 1)
        self.assertEqual(report.after_missing_counts["status"], 0)

    def test_exams_missing_values_not_converted_to_zero(self):
        """Verify missing exam scores are NOT converted to 0, but preserved as NaN with status flags."""
        processed, report = handle_exams_missing_values(self.exams_df)

        # Crucial academic rule check: Never convert missing exam score to 0
        self.assertTrue(np.isnan(processed.iloc[1]["score"]))
        self.assertNotEqual(processed.iloc[1]["score"], 0.0)

        # Indicator columns preserved
        self.assertIn("exam_score_recorded", processed.columns)
        self.assertFalse(processed.iloc[1]["exam_score_recorded"])
        self.assertEqual(processed.iloc[1]["exam_status"], "Pending / Absent / Deferred")
        self.assertEqual(processed.iloc[0]["exam_status"], "Graded")

    def test_submissions_explicit_missing_handling(self):
        """Verify assignment submissions distinguish between unsubmitted and submitted-ungraded."""
        processed, report = handle_submissions_missing_values(self.submissions_df)

        self.assertIn("is_missing_submission", processed.columns)
        self.assertIn("submission_status", processed.columns)

        # Row 0: submitted & graded
        self.assertFalse(processed.iloc[0]["is_missing_submission"])
        self.assertEqual(processed.iloc[0]["submission_status"], "Graded")
        self.assertEqual(processed.iloc[0]["score"], 90.0)

        # Row 1: not submitted
        self.assertTrue(processed.iloc[1]["is_missing_submission"])
        self.assertEqual(processed.iloc[1]["submission_status"], "Not Submitted")

        # Row 2: submitted but ungraded (score should remain NaN, NOT converted to 0)
        self.assertFalse(processed.iloc[2]["is_missing_submission"])
        self.assertEqual(processed.iloc[2]["submission_status"], "Submitted - Ungraded")
        self.assertTrue(np.isnan(processed.iloc[2]["score"]))

    def test_demographics_missing_handling(self):
        """Verify demographic missing values are filled with informative defaults."""
        processed, report = handle_demographics_missing_values(self.students_df, "students")

        self.assertEqual(processed.iloc[1]["program"], "Undeclared")
        self.assertTrue(processed.iloc[1]["program_imputed"])

    def test_handle_academic_missing_values_batch_and_report_markdown(self):
        """Verify batch handling across multiple datasets generates markdown audit reports."""
        datasets = {
            "attendance": self.attendance_df,
            "exams": self.exams_df,
            "submissions": self.submissions_df,
            "students": self.students_df,
        }
        processed, reports = handle_academic_missing_values(datasets)

        self.assertEqual(len(processed), 4)
        self.assertEqual(len(reports), 4)

        # Verify Markdown report formatting
        att_md = reports["attendance"].to_markdown()
        self.assertIn("Missing Value Quality Report: `attendance`", att_md)
        self.assertIn("Total Missing Cells Before", att_md)
        self.assertIn("Domain Strategies Applied", att_md)


if __name__ == "__main__":
    unittest.main()
