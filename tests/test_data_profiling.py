"""Tests for academic data profiling and data-quality reporting."""

import unittest
import numpy as np
import pandas as pd

from src.data_profiling import (
    AcademicQualityReport,
    ColumnProfile,
    DatasetProfile,
    profile_academic_database,
    profile_academic_dataset,
    profile_column,
    profile_dataset,
)
from src.exceptions import DataValidationError


class TestDataProfiling(unittest.TestCase):
    """Test suite for data profiling metrics, statistical summaries, and quality reports."""

    def setUp(self):
        """Prepare academic test fixtures."""
        self.scores = pd.Series([70.0, 80.0, 90.0, np.nan, 100.0], name="exam_score")
        self.categories = pd.Series(["CS", "IT", "CS", "ME", "CS"], name="program")

        self.students_df = pd.DataFrame({
            "student_id": ["STU-1", "STU-2", "STU-1"],  # Duplicate ID
            "name": ["Aarav", "Priya", "Duplicate Aarav"],
            "program": ["CS", "IT", "CS"],
            "year": [2, 3, 2],
        })

        self.attendance_df = pd.DataFrame({
            "student_id": ["STU-1", "STU-2", "STU-3", "STU-4"],
            "course_id": ["CS101", "CS101", "CS101", "CS101"],
            "date": ["2026-02-01"] * 4,
            "status": ["Present", "Absent", "Late", "InvalidStatus"],
        })

        self.submissions_df = pd.DataFrame({
            "assignment_id": ["ASN-1", "ASN-2", "ASN-3"],
            "student_id": ["STU-1", "STU-2", "STU-3"],
            "submission_date": ["2026-02-10"] * 3,
            "score": [85.0, 115.0, -10.0],  # Anomalous scores
        })

    def test_profile_numerical_column(self):
        """Verify numerical column statistics (min, max, mean, median, quartiles)."""
        cp = profile_column(self.scores)
        self.assertEqual(cp.column_name, "exam_score")
        self.assertTrue(cp.is_numerical)
        self.assertEqual(cp.missing_count, 1)
        self.assertEqual(cp.missing_percentage, 20.0)

        stats = cp.numerical_stats
        self.assertIsNotNone(stats)
        self.assertEqual(stats["min"], 70.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertEqual(stats["mean"], 85.0)
        self.assertEqual(stats["median"], 85.0)
        self.assertIn("q25", stats)
        self.assertIn("q75", stats)

    def test_profile_categorical_column(self):
        """Verify categorical column unique counts and top value distributions."""
        cp = profile_column(self.categories)
        self.assertEqual(cp.column_name, "program")
        self.assertFalse(cp.is_numerical)
        self.assertEqual(cp.missing_count, 0)
        self.assertEqual(cp.unique_count, 3)
        self.assertIsNotNone(cp.top_values)
        self.assertEqual(cp.top_values["CS"], 3)

    def test_profile_dataset_alerts(self):
        """Verify dataset profiling generates alerts for duplicates and constant columns."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "constant_col": ["Fixed", "Fixed", "Fixed", "Fixed"],
            "mostly_missing": [np.nan, np.nan, np.nan, 10],  # 75% missing
        })
        prof = profile_dataset(df, dataset_name="alert_test")
        self.assertEqual(prof.row_count, 4)
        self.assertEqual(prof.column_count, 3)
        self.assertTrue(any("duplicate rows" in a.lower() for a in prof.quality_alerts))
        self.assertTrue(any("high missingness" in a.lower() for a in prof.quality_alerts))
        self.assertTrue(any("constant value" in a.lower() for a in prof.quality_alerts))

    def test_profile_dataset_non_dataframe_raises(self):
        """Verify profile_dataset raises DataValidationError for non-DataFrame input."""
        with self.assertRaises(DataValidationError):
            profile_dataset("not a dataframe")

    def test_academic_profile_students_primary_key_anomaly(self):
        """Verify student profiling detects primary key collisions."""
        report = profile_academic_dataset(self.students_df, entity_name="students")
        self.assertIsInstance(report, AcademicQualityReport)
        self.assertTrue(any("duplicate student_id" in a for a in report.anomalies_detected))
        self.assertLess(report.quality_score, 100.0)

    def test_academic_profile_attendance(self):
        """Verify attendance profiling calculates presence rate and detects non-standard status."""
        report = profile_academic_dataset(self.attendance_df, entity_name="attendance")
        self.assertIn("status_distribution", report.academic_metrics)
        self.assertIn("overall_presence_rate", report.academic_metrics)
        self.assertEqual(report.academic_metrics["overall_presence_rate"], "25.0%")
        self.assertTrue(any("Non-standard attendance status" in a for a in report.anomalies_detected))

    def test_academic_profile_scores_out_of_bounds(self):
        """Verify submission and exam profiling flags scores outside [0, 100]."""
        report = profile_academic_dataset(self.submissions_df, entity_name="submissions")
        self.assertTrue(any("outside standard [0, 100] range" in a for a in report.anomalies_detected))
        self.assertLess(report.quality_score, 100.0)

    def test_academic_quality_report_markdown(self):
        """Verify to_markdown outputs structured readable report."""
        report = profile_academic_dataset(self.attendance_df, entity_name="attendance")
        md = report.to_markdown()
        self.assertIn("## Academic Data Quality Report: `attendance`", md)
        self.assertIn("Overall Presence Rate", md)
        self.assertIn("Quality Alerts & Anomalies", md)

    def test_profile_academic_database_batch(self):
        """Verify batch profiling across multiple academic entities."""
        db = {
            "students": self.students_df,
            "attendance": self.attendance_df,
            "submissions": self.submissions_df,
        }
        reports = profile_academic_database(db)
        self.assertEqual(len(reports), 3)
        self.assertIn("students", reports)
        self.assertIn("attendance", reports)
        self.assertIn("submissions", reports)


if __name__ == "__main__":
    unittest.main()
