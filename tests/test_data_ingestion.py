"""Tests for CSV & JSON academic data ingestion and dedicated entity loaders."""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from src.data_intake import (
    load_and_validate_academic_dataset,
    load_assignments,
    load_attendance,
    load_courses,
    load_enrollments,
    load_exams,
    load_students,
    load_submissions,
)
from src.data_io import load_json, save_json
from src.exceptions import DataLoadError, DataValidationError


class TestDataIngestion(unittest.TestCase):
    """Test suite for CSV and JSON academic dataset ingestion."""

    def setUp(self):
        """Prepare sample academic data fixtures in memory and on disk."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        self.students_data = pd.DataFrame([
            {"student_id": "STU-101", "name": "Aarav Sharma", "program": "Computer Science", "year": 2},
            {"student_id": "STU-102", "name": "Priya Mehta", "program": "Information Technology", "year": 3},
        ])

        self.courses_data = pd.DataFrame([
            {"course_id": "CS101", "course_name": "Data Structures", "faculty": "Dr. Smith"},
            {"course_id": "CS102", "course_name": "Algorithms", "faculty": "Prof. Rao"},
        ])

        self.enrollments_data = pd.DataFrame([
            {"student_id": "STU-101", "course_id": "CS101"},
            {"student_id": "STU-102", "course_id": "CS102"},
        ])

        self.attendance_data = pd.DataFrame([
            {"student_id": "STU-101", "course_id": "CS101", "date": "2026-02-01", "status": "Present"},
            {"student_id": "STU-102", "course_id": "CS102", "date": "2026-02-01", "status": "Absent"},
        ])

        self.assignments_data = pd.DataFrame([
            {"assignment_id": "ASN-1", "course_id": "CS101", "title": "Trees & Graphs", "due_date": "2026-02-15"},
            {"assignment_id": "ASN-2", "course_id": "CS102", "title": "Dynamic Programming", "due_date": "2026-02-18"},
        ])

        self.submissions_data = pd.DataFrame([
            {"assignment_id": "ASN-1", "student_id": "STU-101", "submission_date": "2026-02-14", "score": 85.0},
            {"assignment_id": "ASN-2", "student_id": "STU-102", "submission_date": "2026-02-17", "score": 90.0},
        ])

        self.exams_data = pd.DataFrame([
            {"exam_id": "EX-1", "student_id": "STU-101", "course_id": "CS101", "exam_type": "Midterm", "score": 78.0},
            {"exam_id": "EX-2", "student_id": "STU-102", "course_id": "CS102", "exam_type": "Midterm", "score": 84.0},
        ])

    def tearDown(self):
        """Clean up test directory."""
        self.temp_dir.cleanup()

    def test_json_save_and_load_roundtrip(self):
        """Verify save_json and load_json operate correctly."""
        json_file = self.data_dir / "test_students.json"
        save_json(self.students_data, json_file)
        self.assertTrue(json_file.exists())

        loaded_df = load_json(json_file)
        self.assertEqual(len(loaded_df), 2)
        self.assertListEqual(list(loaded_df["student_id"]), ["STU-101", "STU-102"])

    def test_json_load_nonexistent_raises(self):
        """Verify load_json raises DataLoadError on missing file."""
        with self.assertRaises(DataLoadError):
            load_json(self.data_dir / "nonexistent.json")

    def test_load_students_from_csv_and_json(self):
        """Verify load_students works with both CSV and JSON sources."""
        csv_path = self.data_dir / "students.csv"
        json_path = self.data_dir / "students.json"
        self.students_data.to_csv(csv_path, index=False)
        save_json(self.students_data, json_path)

        df_csv, res_csv = load_students(csv_path)
        self.assertTrue(res_csv.is_valid)
        self.assertEqual(len(df_csv), 2)

        df_json, res_json = load_students(json_path)
        self.assertTrue(res_json.is_valid)
        self.assertEqual(len(df_json), 2)

    def test_load_all_dedicated_entity_loaders_json(self):
        """Verify all dedicated entity loaders work seamlessly on JSON files."""
        loaders_and_data = [
            (load_students, "students.json", self.students_data),
            (load_courses, "courses.json", self.courses_data),
            (load_enrollments, "enrollments.json", self.enrollments_data),
            (load_attendance, "attendance.json", self.attendance_data),
            (load_assignments, "assignments.json", self.assignments_data),
            (load_submissions, "submissions.json", self.submissions_data),
            (load_exams, "exams.json", self.exams_data),
        ]

        for loader_func, filename, df in loaders_and_data:
            file_path = self.data_dir / filename
            save_json(df, file_path)
            loaded_df, res = loader_func(file_path)
            self.assertTrue(res.is_valid, f"{filename} failed validation: {res.errors}")
            self.assertEqual(len(loaded_df), 2)

    def test_dedicated_entity_loader_in_memory_dataframe(self):
        """Verify dedicated entity loaders accept in-memory pandas DataFrames."""
        df_loaded, res = load_courses(self.courses_data)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(df_loaded), 2)

    def test_batch_ingestion_mixed_csv_and_json(self):
        """Verify load_and_validate_academic_dataset handles mixed CSV and JSON files in one directory."""
        # Save some as CSV, some as JSON
        self.students_data.to_csv(self.data_dir / "students.csv", index=False)
        save_json(self.courses_data, self.data_dir / "courses.json")
        self.enrollments_data.to_csv(self.data_dir / "enrollments.csv", index=False)
        save_json(self.attendance_data, self.data_dir / "attendance.json")
        self.assignments_data.to_csv(self.data_dir / "assignments.csv", index=False)
        save_json(self.submissions_data, self.data_dir / "submissions.json")
        self.exams_data.to_csv(self.data_dir / "exams.csv", index=False)

        datasets, results = load_and_validate_academic_dataset(self.data_dir, strict=True)
        self.assertEqual(len(datasets), 7)
        for entity in ["students", "courses", "enrollments", "attendance", "assignments", "submissions", "exams"]:
            self.assertIn(entity, datasets)
            self.assertTrue(results[entity].is_valid)

    def test_invalid_json_content_fails_validation(self):
        """Verify malformed JSON or JSON with missing required columns fails validation."""
        bad_json = self.data_dir / "bad_students.json"
        bad_json.write_text("{ this is not valid json }", encoding="utf-8")

        with self.assertRaises(DataValidationError):
            load_students(bad_json)

    def test_json_missing_columns_fails_validation(self):
        """Verify valid JSON with missing required columns fails validation."""
        incomplete_json = self.data_dir / "incomplete_courses.json"
        save_json(pd.DataFrame([{"course_id": "CS101"}]), incomplete_json)

        with self.assertRaises(DataValidationError):
            load_courses(incomplete_json)


if __name__ == "__main__":
    unittest.main()
