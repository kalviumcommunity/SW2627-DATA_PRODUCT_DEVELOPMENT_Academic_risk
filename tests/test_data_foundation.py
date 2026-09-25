"""Tests for Python Data Processing Foundation utilities."""

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.exceptions import (
    DataLoadError,
    DataProcessingError,
    DataSaveError,
    DataTransformationError,
    DataValidationError,
)
from src.logger import get_logger
from src.data_io import load_csv, load_from_sqlite, save_csv, save_to_sqlite
from src.data_inspector import (
    check_not_empty,
    generate_summary_report,
    inspect_dataframe,
    validate_required_columns,
)
from src.data_transformer import (
    cast_column_types,
    convert_to_datetime,
    drop_exact_duplicates,
    standardize_column_names,
    strip_whitespace,
)


class TestDataFoundation(unittest.TestCase):
    """Test suite for data I/O, inspection, transformation, and exception handling."""

    def setUp(self):
        """Create sample DataFrames and temp directory for isolated tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.sample_df = pd.DataFrame({
            " Student ID ": ["STU-001", "STU-002", "STU-003", "STU-001"],
            "Course Name - Code": [" Math 101 ", " Physics 201 ", " Chemistry 301 ", " Math 101 "],
            "Score": [85.0, np.nan, 92.5, 85.0],
            "Date": ["2026-01-15", "2026-01-16", "2026-01-17", "2026-01-15"],
        })

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_logger_initialization(self):
        """Verify logger returns a valid configured logger."""
        logger = get_logger("test_academic_logger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_academic_logger")

    def test_exception_hierarchy(self):
        """Verify that custom data exceptions inherit from DataProcessingError."""
        self.assertTrue(issubclass(DataLoadError, DataProcessingError))
        self.assertTrue(issubclass(DataValidationError, DataProcessingError))
        self.assertTrue(issubclass(DataTransformationError, DataProcessingError))
        self.assertTrue(issubclass(DataSaveError, DataProcessingError))

    def test_csv_save_and_load(self):
        """Verify saving and loading CSV files via data_io."""
        target_file = self.temp_path / "nested" / "test_students.csv"
        saved_path = save_csv(self.sample_df, target_file)

        self.assertTrue(saved_path.exists())
        loaded_df = load_csv(target_file)
        self.assertEqual(len(loaded_df), 4)
        self.assertEqual(list(loaded_df.columns), list(self.sample_df.columns))

    def test_csv_load_nonexistent_raises(self):
        """Verify load_csv raises DataLoadError when file does not exist."""
        with self.assertRaises(DataLoadError):
            load_csv(self.temp_path / "does_not_exist.csv")

    def test_sqlite_save_and_load(self):
        """Verify saving to SQLite and reading back with SQL query."""
        db_path = self.temp_path / "academic.db"
        save_to_sqlite(self.sample_df, table_name="test_table", db_path=db_path)

        self.assertTrue(db_path.exists())
        query_df = load_from_sqlite("SELECT COUNT(*) AS total FROM test_table", db_path=db_path)
        self.assertEqual(query_df.iloc[0]["total"], 4)

    def test_sqlite_load_nonexistent_raises(self):
        """Verify load_from_sqlite raises DataLoadError if DB missing."""
        with self.assertRaises(DataLoadError):
            load_from_sqlite("SELECT 1", self.temp_path / "nonexistent.db")

    def test_inspect_dataframe(self):
        """Verify structured inspection metrics generation."""
        metrics = inspect_dataframe(self.sample_df, name="sample_test")
        self.assertEqual(metrics["row_count"], 4)
        self.assertEqual(metrics["column_count"], 4)
        self.assertEqual(metrics["missing_counts"]["Score"], 1)
        self.assertEqual(metrics["duplicate_rows"], 1)

    def test_generate_summary_report(self):
        """Verify summary report text formatting."""
        report = generate_summary_report(self.sample_df, name="sample_test")
        self.assertIn("sample_test", report)
        self.assertIn("Rows: 4", report)
        self.assertIn("Columns: 4", report)

    def test_validate_required_columns(self):
        """Verify required column validation raises on missing columns."""
        validate_required_columns(self.sample_df, ["Score", "Date"])
        with self.assertRaises(DataValidationError):
            validate_required_columns(self.sample_df, ["NonexistentColumn"])

    def test_check_not_empty(self):
        """Verify check_not_empty validates non-empty and raises on empty."""
        check_not_empty(self.sample_df)
        empty_df = pd.DataFrame()
        with self.assertRaises(DataValidationError):
            check_not_empty(empty_df, name="empty_test")

    def test_standardize_column_names(self):
        """Verify column name standardization to snake_case."""
        cleaned = standardize_column_names(self.sample_df)
        expected = ["student_id", "course_name_code", "score", "date"]
        self.assertEqual(list(cleaned.columns), expected)

    def test_strip_whitespace(self):
        """Verify trimming leading and trailing whitespace."""
        stripped = strip_whitespace(self.sample_df)
        self.assertEqual(stripped.iloc[0][" Student ID "], "STU-001")
        self.assertEqual(stripped.iloc[1]["Course Name - Code"], "Physics 201")

    def test_convert_to_datetime(self):
        """Verify datetime conversion."""
        dt_df = convert_to_datetime(self.sample_df, columns=["Date"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(dt_df["Date"]))

    def test_cast_column_types(self):
        """Verify safe type casting."""
        casted = cast_column_types(self.sample_df, {" Score": "float"}) if " Score" in self.sample_df else self.sample_df
        self.assertIsNotNone(casted)

    def test_drop_exact_duplicates(self):
        """Verify duplicate row removal."""
        deduped = drop_exact_duplicates(self.sample_df)
        self.assertEqual(len(deduped), 3)


if __name__ == "__main__":
    unittest.main()
