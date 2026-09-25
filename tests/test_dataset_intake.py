"""Unit and integration tests for academic dataset intake and validation."""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from src.data_intake import (
    ENTITY_SCHEMAS,
    ValidationResult,
    load_and_validate_academic_dataset,
    load_and_validate_entity,
    validate_dataset_file,
)
from src.exceptions import DataValidationError


class TestDatasetIntake(unittest.TestCase):
    """Test suite covering academic dataset intake, validation, and edge cases."""

    def setUp(self):
        """Create temporary test fixtures for valid and invalid datasets."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        # 1. Valid data dictionaries
        self.valid_data = {
            "students": pd.DataFrame({
                "student_id": ["STU-1001", "STU-1002"],
                "name": ["Aarav Sharma", "Priya Mehta"],
                "program": ["Computer Science", "Information Technology"],
                "year": [2, 3],
            }),
            "courses": pd.DataFrame({
                "course_id": ["CS101", "CS102"],
                "course_name": ["Data Structures", "Database Systems"],
                "faculty": ["Dr. Smith", "Prof. Johnson"],
            }),
            "enrollments": pd.DataFrame({
                "student_id": ["STU-1001", "STU-1002"],
                "course_id": ["CS101", "CS102"],
            }),
            "attendance": pd.DataFrame({
                "student_id": ["STU-1001", "STU-1002"],
                "course_id": ["CS101", "CS102"],
                "date": ["2026-02-01", "2026-02-01"],
                "status": ["Present", "Absent"],
            }),
            "assignments": pd.DataFrame({
                "assignment_id": ["ASN-01", "ASN-02"],
                "course_id": ["CS101", "CS102"],
                "title": ["Linked List Implementation", "SQL Query Optimization"],
                "due_date": ["2026-02-10", "2026-02-12"],
            }),
            "submissions": pd.DataFrame({
                "assignment_id": ["ASN-01", "ASN-02"],
                "student_id": ["STU-1001", "STU-1002"],
                "submission_date": ["2026-02-09", "2026-02-11"],
                "score": [88.0, 74.5],
            }),
            "exams": pd.DataFrame({
                "exam_id": ["EXM-01", "EXM-02"],
                "student_id": ["STU-1001", "STU-1002"],
                "course_id": ["CS101", "CS102"],
                "exam_type": ["Midterm", "Midterm"],
                "score": [78.5, 62.0],
            }),
        }

        # Write all valid files to disk
        for entity, df in self.valid_data.items():
            df.to_csv(self.data_dir / f"{entity}.csv", index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_all_seven_entities_defined_in_schemas(self):
        """Verify all 7 expected academic entities have schemas configured."""
        expected_entities = {
            "students",
            "courses",
            "enrollments",
            "attendance",
            "assignments",
            "submissions",
            "exams",
        }
        self.assertEqual(set(ENTITY_SCHEMAS.keys()), expected_entities)

    def test_valid_entities_validation(self):
        """Verify that all 7 valid datasets pass intake validation."""
        for entity in ENTITY_SCHEMAS:
            path = self.data_dir / f"{entity}.csv"
            df, result = validate_dataset_file(file_path=path, entity_name=entity)
            self.assertTrue(result.is_valid, f"Entity '{entity}' failed validation: {result.errors}")
            self.assertIsNotNone(df)
            self.assertEqual(result.row_count, 2)
            self.assertEqual(len(result.missing_columns), 0)

    def test_missing_file_validation(self):
        """Verify validation fails gracefully when file does not exist."""
        nonexistent = self.data_dir / "missing_students.csv"
        df, result = validate_dataset_file(file_path=nonexistent, entity_name="students")
        self.assertFalse(result.is_valid)
        self.assertIsNone(df)
        self.assertTrue(any("File not found" in err for err in result.errors))

    def test_unsupported_file_format(self):
        """Verify unsupported extensions (e.g. .xlsx, .txt) are rejected."""
        unsupported_file = self.data_dir / "students.xlsx"
        unsupported_file.write_text("some,text,data")
        df, result = validate_dataset_file(file_path=unsupported_file, entity_name="students")
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Unsupported file format" in err for err in result.errors))

    def test_empty_zero_byte_file(self):
        """Verify completely empty 0-byte file fails validation."""
        empty_file = self.data_dir / "empty_attendance.csv"
        empty_file.touch()
        df, result = validate_dataset_file(file_path=empty_file, entity_name="attendance")
        self.assertFalse(result.is_valid)
        self.assertTrue(any("empty" in err.lower() for err in result.errors))

    def test_empty_rows_dataset(self):
        """Verify CSV with header only (0 rows) fails validation."""
        header_only = self.data_dir / "header_only_courses.csv"
        pd.DataFrame(columns=["course_id", "course_name", "faculty"]).to_csv(header_only, index=False)
        df, result = validate_dataset_file(file_path=header_only, entity_name="courses")
        self.assertFalse(result.is_valid)
        self.assertTrue(any("no data rows" in err for err in result.errors))

    def test_missing_required_columns(self):
        """Verify validation detects missing required columns."""
        invalid_students = self.data_dir / "incomplete_students.csv"
        pd.DataFrame({
            "student_id": ["STU-99"],
            "name": ["Test Student"],
            # Missing 'program' and 'year'
        }).to_csv(invalid_students, index=False)

        df, result = validate_dataset_file(file_path=invalid_students, entity_name="students")
        self.assertFalse(result.is_valid)
        self.assertIn("program", result.missing_columns)
        self.assertIn("year", result.missing_columns)

    def test_column_alias_mapping(self):
        """Verify recognized aliases (e.g. attendance_status -> status) are mapped."""
        alias_attendance = self.data_dir / "alias_attendance.csv"
        pd.DataFrame({
            "student_id": ["STU-1001"],
            "course_id": ["CS101"],
            "date": ["2026-02-01"],
            "attendance_status": ["Present"],  # Alias for status
        }).to_csv(alias_attendance, index=False)

        df, result = validate_dataset_file(file_path=alias_attendance, entity_name="attendance")
        self.assertTrue(result.is_valid)
        self.assertIn("status", df.columns)
        self.assertTrue(any("attendance_status" in w for w in result.warnings))

    def test_load_and_validate_entity_success_and_raise(self):
        """Verify load_and_validate_entity returns df on success and raises on failure."""
        valid_path = self.data_dir / "students.csv"
        df, res = load_and_validate_entity(valid_path, "students")
        self.assertEqual(len(df), 2)

        bad_path = self.data_dir / "nonexistent.csv"
        with self.assertRaises(DataValidationError):
            load_and_validate_entity(bad_path, "students")

    def test_load_and_validate_academic_dataset_batch(self):
        """Verify full directory batch ingestion loads all 7 entities."""
        datasets, results = load_and_validate_academic_dataset(self.data_dir, strict=True)
        self.assertEqual(len(datasets), 7)
        self.assertEqual(len(results), 7)
        for entity in ENTITY_SCHEMAS:
            self.assertIn(entity, datasets)
            self.assertTrue(results[entity].is_valid)

    def test_load_and_validate_academic_dataset_strict_failure(self):
        """Verify batch ingestion raises DataValidationError if an entity file is missing."""
        (self.data_dir / "exams.csv").unlink()
        with self.assertRaises(DataValidationError):
            load_and_validate_academic_dataset(self.data_dir, strict=True)

        # When strict=False, should return results without raising
        datasets, results = load_and_validate_academic_dataset(self.data_dir, strict=False)
        self.assertNotIn("exams", datasets)
        self.assertFalse(results["exams"].is_valid)


if __name__ == "__main__":
    unittest.main()
