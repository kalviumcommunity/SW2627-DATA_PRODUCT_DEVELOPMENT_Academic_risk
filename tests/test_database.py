"""Unit tests for Concept #27: SQLite Academic Database."""

from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.exceptions import DataLoadError, DataSaveError, DataValidationError
from src.database import (
    CANONICAL_TABLES,
    DEFAULT_SCHEMA_PATH,
    TOPOLOGICAL_INGESTION_ORDER,
    AcademicDatabase,
    DatabaseValidationReport,
    ForeignKeyViolation,
    get_connection,
    get_table_row_counts,
    init_database,
    load_cleaned_datasets_to_sqlite,
    load_table_data,
    sanitize_dataframe_for_table,
    validate_academic_database,
    validate_database_relationships,
    validate_foreign_keys,
    validate_row_counts,
    verify_schema,
)


class TestDatabaseBase(unittest.TestCase):
    """Base test class providing synthetic clean academic datasets."""

    def setUp(self):
        """Create temporary directory, database path, and synthetic DataFrames."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "academic_test.db"

        # 1. Students
        self.students_df = pd.DataFrame({
            "student_id": ["STU-001", "STU-002", "STU-003"],
            "name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
            "program": ["Computer Science", "Information Technology", "Computer Science"],
            "year": [1, 2, 3],
        })

        # 2. Courses
        self.courses_df = pd.DataFrame({
            "course_id": ["CS101", "DS202"],
            "course_name": ["Intro to CS", "Data Structures"],
            "faculty": ["Dr. Smith", "Prof. Davis"],
        })

        # 3. Enrollments
        self.enrollments_df = pd.DataFrame({
            "student_id": ["STU-001", "STU-002", "STU-003", "STU-001"],
            "course_id": ["CS101", "CS101", "DS202", "DS202"],
        })

        # 4. Attendance
        self.attendance_df = pd.DataFrame({
            "student_id": ["STU-001", "STU-002", "STU-001"],
            "course_id": ["CS101", "CS101", "DS202"],
            "date": ["2026-01-10", "2026-01-10", "2026-01-12"],
            "status": ["Present", "Late", "Present"],
        })

        # 5. Assignments
        self.assignments_df = pd.DataFrame({
            "assignment_id": ["ASN-01", "ASN-02"],
            "course_id": ["CS101", "DS202"],
            "title": ["Variables & Loops", "Binary Trees"],
            "due_date": ["2026-01-20", "2026-02-01"],
        })

        # 6. Submissions
        self.submissions_df = pd.DataFrame({
            "assignment_id": ["ASN-01", "ASN-01", "ASN-02"],
            "student_id": ["STU-001", "STU-002", "STU-003"],
            "submission_date": ["2026-01-19", "2026-01-21", "2026-01-30"],
            "score": [95.0, 78.5, 88.0],
        })

        # 7. Exams
        self.exams_df = pd.DataFrame({
            "exam_id": ["EXM-01", "EXM-02"],
            "student_id": ["STU-001", "STU-002"],
            "course_id": ["CS101", "CS101"],
            "exam_type": ["Midterm", "Midterm"],
            "score": [88.0, 72.0],
        })

        # 8. Interventions
        self.interventions_df = pd.DataFrame({
            "student_id": ["STU-002"],
            "date": ["2026-01-25"],
            "type": ["Academic check-in"],
            "status": ["Open"],
            "notes": ["Discussed late submission and coursework pacing."],
        })

        self.complete_datasets = {
            "students": self.students_df,
            "courses": self.courses_df,
            "enrollments": self.enrollments_df,
            "attendance": self.attendance_df,
            "assignments": self.assignments_df,
            "submissions": self.submissions_df,
            "exams": self.exams_df,
            "interventions": self.interventions_df,
        }

    def tearDown(self):
        """Cleanup temporary directory."""
        self.temp_dir.cleanup()


class TestDatabaseSchema(TestDatabaseBase):
    """Test schema execution, tables, and column structures."""

    def test_schema_initialization(self):
        """Verify schema initializes all 8 canonical tables."""
        conn = init_database(self.db_path)
        try:
            schema_info = verify_schema(conn)
            self.assertEqual(len(schema_info), 8)
            for tbl in CANONICAL_TABLES:
                self.assertIn(tbl, schema_info)
        finally:
            conn.close()

    def test_verify_schema_columns(self):
        """Verify specific expected columns exist in schema."""
        init_database(self.db_path).close()
        schema_info = verify_schema(self.db_path)

        self.assertIn("student_id", schema_info["students"])
        self.assertIn("course_id", schema_info["courses"])
        self.assertIn("status", schema_info["attendance"])
        self.assertIn("due_date", schema_info["assignments"])
        self.assertIn("score", schema_info["submissions"])
        self.assertIn("exam_type", schema_info["exams"])
        self.assertIn("notes", schema_info["interventions"])

    def test_verify_schema_missing_table_raises(self):
        """Verify DataValidationError when a table is missing."""
        conn = init_database(self.db_path)
        conn.execute("DROP TABLE interventions;")
        conn.commit()
        try:
            with self.assertRaises(DataValidationError):
                verify_schema(conn)
        finally:
            conn.close()

    def test_schema_file_not_found(self):
        """Verify DataLoadError when schema file does not exist."""
        with self.assertRaises(DataLoadError):
            init_database(self.db_path, schema_path="/nonexistent/path/schema.sql")

    def test_reinitialize_drop_existing(self):
        """Verify re-initialization drops existing tables and recreates cleanly."""
        init_database(self.db_path).close()
        # Insert a student
        conn = get_connection(self.db_path)
        conn.execute("INSERT INTO students VALUES ('S1', 'Name', 'Prog', 1);")
        conn.commit()
        conn.close()

        # Re-initialize with drop_existing
        init_database(self.db_path, drop_existing=True).close()
        counts = get_table_row_counts(self.db_path)
        self.assertEqual(counts["students"], 0)


class TestDatabaseLoading(TestDatabaseBase):
    """Test loading cleaned DataFrames into SQLite."""

    def test_load_cleaned_datasets_to_sqlite(self):
        """Verify loading complete 8-entity dataset into SQLite."""
        report = load_cleaned_datasets_to_sqlite(
            db_path=self.db_path,
            datasets=self.complete_datasets,
            init_schema=True,
            validate=True,
        )

        self.assertTrue(report.is_valid)
        self.assertTrue(report.row_counts_match)
        self.assertEqual(report.table_counts["students"], 3)
        self.assertEqual(report.table_counts["courses"], 2)
        self.assertEqual(report.table_counts["enrollments"], 4)
        self.assertEqual(report.table_counts["attendance"], 3)
        self.assertEqual(report.table_counts["assignments"], 2)
        self.assertEqual(report.table_counts["submissions"], 3)
        self.assertEqual(report.table_counts["exams"], 2)
        self.assertEqual(report.table_counts["interventions"], 1)

    def test_sanitize_dataframe_for_table(self):
        """Verify DataFrame sanitizer discards unexpected columns and handles whitespace."""
        conn = init_database(self.db_path)
        try:
            extra_df = pd.DataFrame({
                "student_id": ["STU-999"],
                "name": ["Test"],
                "program": ["CS"],
                "year": [1],
                "extra_column": ["ignore_me"],
                "another_junk": [123],
            })
            sanitized = sanitize_dataframe_for_table(extra_df, "students", conn)
            self.assertIn("student_id", sanitized.columns)
            self.assertNotIn("extra_column", sanitized.columns)
            self.assertNotIn("another_junk", sanitized.columns)
        finally:
            conn.close()

    def test_load_empty_or_none_dataset(self):
        """Verify handling of empty or None datasets."""
        init_database(self.db_path).close()
        conn = get_connection(self.db_path)
        try:
            inserted = load_table_data(conn, "students", pd.DataFrame())
            self.assertEqual(inserted, 0)
        finally:
            conn.close()

    def test_topological_ingestion_order_honors_foreign_keys(self):
        """Verify data loads in correct parent-before-child sequence."""
        # Intentionally order dictionary with children first
        scrambled_datasets = {
            "submissions": self.submissions_df,
            "enrollments": self.enrollments_df,
            "exams": self.exams_df,
            "attendance": self.attendance_df,
            "assignments": self.assignments_df,
            "interventions": self.interventions_df,
            "students": self.students_df,
            "courses": self.courses_df,
        }

        report = load_cleaned_datasets_to_sqlite(
            db_path=self.db_path,
            datasets=scrambled_datasets,
            validate=True,
        )
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.foreign_key_violations), 0)


class TestForeignKeyConstraints(TestDatabaseBase):
    """Test foreign key constraint enforcement in SQLite."""

    def test_foreign_key_violation_on_insert_rejected(self):
        """Verify inserting an orphan foreign key directly fails constraint."""
        init_database(self.db_path).close()
        conn = get_connection(self.db_path, foreign_keys=True)
        try:
            # Insert enrollment for student STU-999 who does not exist
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO enrollments (student_id, course_id) VALUES ('STU-999', 'CS101');")
        finally:
            conn.close()

    def test_cascade_delete(self):
        """Verify deleting a student cascades to child records."""
        load_cleaned_datasets_to_sqlite(
            db_path=self.db_path,
            datasets=self.complete_datasets,
            validate=False,
        )

        conn = get_connection(self.db_path, foreign_keys=True)
        try:
            # Delete student STU-001
            conn.execute("DELETE FROM students WHERE student_id = 'STU-001';")
            conn.commit()

            # Check that STU-001 enrollments and submissions were cascaded
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM enrollments WHERE student_id = 'STU-001';")
            self.assertEqual(cur.fetchone()[0], 0)

            cur.execute("SELECT COUNT(*) FROM submissions WHERE student_id = 'STU-001';")
            self.assertEqual(cur.fetchone()[0], 0)
        finally:
            conn.close()


class TestValidationUtilities(TestDatabaseBase):
    """Test relationship and row count validation utilities."""

    def test_validate_foreign_keys_clean(self):
        """Verify 0 violations on clean dataset."""
        load_cleaned_datasets_to_sqlite(self.db_path, self.complete_datasets)
        violations = validate_foreign_keys(self.db_path)
        self.assertEqual(len(violations), 0)

    def test_validate_database_relationships_detects_violations(self):
        """Verify that relationships validation detects an orphan key."""
        init_database(self.db_path).close()

        # Temporarily disable foreign keys to forcefully insert an orphan
        conn = get_connection(self.db_path, foreign_keys=False)
        try:
            conn.execute("INSERT INTO courses VALUES ('CS101', 'Intro to CS', 'Dr. Smith');")
            # Orphan student in enrollments
            conn.execute("INSERT INTO enrollments VALUES ('ORPHAN_STU', 'CS101');")
            conn.commit()
        finally:
            conn.close()

        violations = validate_database_relationships(self.db_path)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any(v.table_name == "enrollments" for v in violations))

    def test_validate_row_counts_matching(self):
        """Verify validate_row_counts returns matches=True when matching."""
        load_cleaned_datasets_to_sqlite(self.db_path, self.complete_datasets)
        expected = {
            "students": 3,
            "courses": 2,
            "enrollments": 4,
            "attendance": 3,
            "assignments": 2,
            "submissions": 3,
            "exams": 2,
            "interventions": 1,
        }
        res = validate_row_counts(self.db_path, expected)
        self.assertTrue(res["matches"])
        self.assertEqual(res["differences"]["students"], 0)

    def test_validate_row_counts_mismatch(self):
        """Verify validate_row_counts detects count differences."""
        load_cleaned_datasets_to_sqlite(self.db_path, self.complete_datasets)
        expected = {
            "students": 10,  # Expected 10, actual is 3
        }
        res = validate_row_counts(self.db_path, expected)
        self.assertFalse(res["matches"])
        self.assertEqual(res["differences"]["students"], -7)

    def test_full_database_validation_report(self):
        """Verify complete validation report generation and serialization."""
        report = load_cleaned_datasets_to_sqlite(self.db_path, self.complete_datasets)
        self.assertIsInstance(report, DatabaseValidationReport)
        self.assertTrue(report.is_valid)

        report_dict = report.to_dict()
        self.assertIsInstance(report_dict, dict)
        self.assertEqual(report_dict["is_valid"], True)
        self.assertIn("table_counts", report_dict)
        self.assertIn("summary", report_dict)


class TestAcademicDatabaseClass(TestDatabaseBase):
    """Test object-oriented AcademicDatabase manager class."""

    def test_database_manager_workflow(self):
        """Verify full lifecycle using AcademicDatabase wrapper."""
        db = AcademicDatabase(self.db_path)
        db.initialize()

        report = db.load_datasets(self.complete_datasets)
        self.assertTrue(report.is_valid)

        # Query using manager
        query_df = db.query("SELECT COUNT(*) AS cnt FROM students WHERE year = ?", params=(1,))
        self.assertEqual(query_df.iloc[0]["cnt"], 1)

        counts = db.get_row_counts()
        self.assertEqual(counts["students"], 3)

        val_report = db.validate()
        self.assertTrue(val_report.is_valid)

    def test_database_manager_context_manager(self):
        """Verify context manager syntax."""
        with AcademicDatabase(self.db_path) as db:
            db.initialize()
            counts = db.get_row_counts()
            self.assertEqual(counts["courses"], 0)


if __name__ == "__main__":
    unittest.main()
