"""Tests for academic data type standardization."""

import unittest
import numpy as np
import pandas as pd

from src.data_standardizer import (
    standardize_academic_dataset,
    standardize_assignments,
    standardize_attendance,
    standardize_attendance_status_series,
    standardize_categorical_series,
    standardize_courses,
    standardize_date_series,
    standardize_enrollments,
    standardize_exams,
    standardize_id_series,
    standardize_percentage_series,
    standardize_score_series,
    standardize_students,
    standardize_submissions,
    standardize_year_series,
)


class TestDataStandardizer(unittest.TestCase):
    """Test suite ensuring data types, identifiers, dates, and scores standardize cleanly."""

    def test_standardize_id_series(self):
        """Verify IDs are uppercase, trimmed, and float strings normalized."""
        s = pd.Series([" stu-101 ", "cs102", "1005.0", np.nan, ""])
        cleaned = standardize_id_series(s)
        self.assertEqual(cleaned.iloc[0], "STU-101")
        self.assertEqual(cleaned.iloc[1], "CS102")
        self.assertEqual(cleaned.iloc[2], "1005")
        self.assertTrue(pd.isna(cleaned.iloc[3]))
        self.assertTrue(pd.isna(cleaned.iloc[4]))

    def test_standardize_date_series(self):
        """Verify date series normalizes to ISO YYYY-MM-DD and invalid entries coerce to NaT."""
        s = pd.Series(["2026-02-15", "2026/02/16", "Feb 17, 2026", "invalid-date", np.nan])
        cleaned = standardize_date_series(s)
        self.assertEqual(cleaned.iloc[0], "2026-02-15")
        self.assertEqual(cleaned.iloc[1], "2026-02-16")
        self.assertEqual(cleaned.iloc[2], "2026-02-17")
        self.assertTrue(pd.isna(cleaned.iloc[3]))
        self.assertTrue(pd.isna(cleaned.iloc[4]))

    def test_standardize_score_series(self):
        """Verify scores strip non-numeric symbols and coerce invalid values to NaN."""
        s = pd.Series(["85.5", "92/100", " 74 pts ", "60%", "absent", np.nan])
        cleaned = standardize_score_series(s)
        self.assertEqual(cleaned.iloc[0], 85.5)
        self.assertEqual(cleaned.iloc[1], 92.0)
        self.assertEqual(cleaned.iloc[2], 74.0)
        self.assertEqual(cleaned.iloc[3], 60.0)
        self.assertTrue(np.isnan(cleaned.iloc[4]))
        self.assertTrue(np.isnan(cleaned.iloc[5]))

    def test_standardize_percentage_series(self):
        """Verify percentage values scale decimals to 100.0 properly."""
        decimal_series = pd.Series([0.85, 0.72, 0.90])
        scaled = standardize_percentage_series(decimal_series, scale_to_100=True)
        self.assertEqual(scaled.iloc[0], 85.0)
        self.assertEqual(scaled.iloc[1], 72.0)

        formatted_series = pd.Series(["85%", "72.5%", "90 %"])
        parsed = standardize_percentage_series(formatted_series)
        self.assertEqual(parsed.iloc[0], 85.0)
        self.assertEqual(parsed.iloc[1], 72.5)

    def test_standardize_year_series(self):
        """Verify academic years extract valid 1-4 integers."""
        s = pd.Series(["Year 2", "2nd Year", "3", 4.0, "Graduate", np.nan])
        cleaned = standardize_year_series(s)
        self.assertEqual(cleaned.iloc[0], 2)
        self.assertEqual(cleaned.iloc[1], 2)
        self.assertEqual(cleaned.iloc[2], 3)
        self.assertEqual(cleaned.iloc[3], 4)
        self.assertTrue(pd.isna(cleaned.iloc[4]))
        self.assertTrue(pd.isna(cleaned.iloc[5]))

    def test_standardize_attendance_status_series(self):
        """Verify attendance codes normalize to canonical status categories."""
        s = pd.Series(["p", "PRESENT", "1", "a", "absent", "0", "l", "late", "e", "medical", "unknown", np.nan])
        cleaned = standardize_attendance_status_series(s)
        self.assertEqual(cleaned.iloc[0], "Present")
        self.assertEqual(cleaned.iloc[1], "Present")
        self.assertEqual(cleaned.iloc[2], "Present")
        self.assertEqual(cleaned.iloc[3], "Absent")
        self.assertEqual(cleaned.iloc[4], "Absent")
        self.assertEqual(cleaned.iloc[5], "Absent")
        self.assertEqual(cleaned.iloc[6], "Late")
        self.assertEqual(cleaned.iloc[7], "Late")
        self.assertEqual(cleaned.iloc[8], "Excused")
        self.assertEqual(cleaned.iloc[9], "Excused")
        self.assertEqual(cleaned.iloc[10], "Unrecorded")
        self.assertEqual(cleaned.iloc[11], "Unrecorded")

    def test_standardize_categorical_series(self):
        """Verify whitespace collapsing and Title Case formatting."""
        s = pd.Series(["  computer   science ", "data  engineering", np.nan])
        cleaned = standardize_categorical_series(s, title_case=True)
        self.assertEqual(cleaned.iloc[0], "Computer Science")
        self.assertEqual(cleaned.iloc[1], "Data Engineering")
        self.assertTrue(pd.isna(cleaned.iloc[2]))

    def test_standardize_students_dataframe(self):
        """Verify student DataFrame entity standardization."""
        df = pd.DataFrame({
            "student_id": [" stu-101 "],
            "name": [" aarav  sharma "],
            "program": [" computer science "],
            "year": ["Year 2"],
        })
        std = standardize_students(df)
        self.assertEqual(std.iloc[0]["student_id"], "STU-101")
        self.assertEqual(std.iloc[0]["name"], "Aarav Sharma")
        self.assertEqual(std.iloc[0]["program"], "Computer Science")
        self.assertEqual(std.iloc[0]["year"], 2)

    def test_standardize_academic_dataset_batch(self):
        """Verify batch standardization across all entities in the database."""
        db = {
            "students": pd.DataFrame({"student_id": ["stu-1"], "name": ["aarav"], "program": ["cs"], "year": [2]}),
            "courses": pd.DataFrame({"course_id": ["cs101"], "course_name": ["data structures"], "faculty": ["dr. smith"]}),
            "attendance": pd.DataFrame({"student_id": ["stu-1"], "course_id": ["cs101"], "date": ["2026/02/01"], "status": ["p"]}),
            "exams": pd.DataFrame({"exam_id": ["ex-1"], "student_id": ["stu-1"], "course_id": ["cs101"], "exam_type": ["mid-term"], "score": ["85%"]}),
        }
        std_db = standardize_academic_dataset(db)
        self.assertEqual(std_db["students"].iloc[0]["student_id"], "STU-1")
        self.assertEqual(std_db["courses"].iloc[0]["course_id"], "CS101")
        self.assertEqual(std_db["attendance"].iloc[0]["date"], "2026-02-01")
        self.assertEqual(std_db["attendance"].iloc[0]["status"], "Present")
        self.assertEqual(std_db["exams"].iloc[0]["exam_type"], "Midterm")
        self.assertEqual(std_db["exams"].iloc[0]["score"], 85.0)


if __name__ == "__main__":
    unittest.main()
