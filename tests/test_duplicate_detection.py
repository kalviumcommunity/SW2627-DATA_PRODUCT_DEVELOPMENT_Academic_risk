"""Unit tests for Concept #10: Academic Duplicate Detection & Resolution."""

import numpy as np
import pandas as pd
import pytest

from src.duplicate_detection import (
    DEFAULT_BUSINESS_KEYS,
    BatchDuplicateReport,
    DuplicateMetrics,
    DuplicateReport,
    clean_duplicates,
    compute_duplicate_metrics,
    deduplicate_academic_dataset,
    deduplicate_assignments,
    deduplicate_attendance,
    deduplicate_courses,
    deduplicate_exams,
    deduplicate_students,
    deduplicate_submissions,
    detect_duplicates,
    resolve_business_keys,
)
from src.exceptions import DataValidationError


class TestDuplicateDetection:
    """Test suite for duplicate detection, business key resolution, and audit reporting."""

    def test_resolve_business_keys_defaults_and_fallbacks(self):
        """Test resolving business keys across canonical defaults and fallbacks."""
        # Students default
        df_students = pd.DataFrame({"student_id": ["S01"], "name": ["Alice"]})
        assert resolve_business_keys(df_students, "students") == ["student_id"]

        # Attendance default
        df_att = pd.DataFrame({"student_id": ["S01"], "course_id": ["C01"], "date": ["2026-03-01"], "status": ["Present"]})
        assert resolve_business_keys(df_att, "attendance") == ["student_id", "course_id", "date"]

        # Submissions default
        df_sub = pd.DataFrame({"assignment_id": ["A01"], "student_id": ["S01"], "score": [90]})
        assert resolve_business_keys(df_sub, "submissions") == ["assignment_id", "student_id"]

        # Exams fallback when exam_id is absent
        df_exams_fallback = pd.DataFrame({
            "student_id": ["S01"],
            "course_id": ["C01"],
            "exam_type": ["Midterm"],
            "score": [85],
        })
        assert resolve_business_keys(df_exams_fallback, "exams") == ["student_id", "course_id", "exam_type"]

        # Custom keys
        assert resolve_business_keys(df_students, "students", custom_keys=["name"]) == ["name"]

        # Missing keys raises error
        with pytest.raises(DataValidationError):
            resolve_business_keys(pd.DataFrame({"unrelated": [1]}), "students")

    def test_compute_duplicate_metrics_and_conflicts(self):
        """Test distinction between exact duplicates, business-key duplicates, and conflicting records."""
        # Row 0 and 1: exact duplicates
        # Row 2 and 3: business key duplicate with conflicting status
        data = {
            "student_id": ["S01", "S01", "S02", "S02", "S03"],
            "course_id": ["C01", "C01", "C01", "C01", "C01"],
            "date": ["2026-03-01", "2026-03-01", "2026-03-01", "2026-03-01", "2026-03-01"],
            "status": ["Present", "Present", "Present", "Absent", "Late"],
        }
        df = pd.DataFrame(data)
        keys = ["student_id", "course_id", "date"]

        metrics = compute_duplicate_metrics(df, keys)

        assert metrics.total_rows == 5
        assert metrics.unique_business_keys == 3
        assert metrics.duplicate_groups == 2  # S01 group and S02 group
        assert metrics.key_duplicate_rows == 4  # rows 0, 1, 2, 3
        assert metrics.exact_duplicate_rows == 2  # rows 0 and 1
        assert metrics.conflicting_rows == 2  # rows 2 and 3 (Present vs Absent)

    def test_detect_duplicates_annotations(self):
        """Test annotation columns added by detect_duplicates."""
        df = pd.DataFrame({
            "student_id": ["S01", "S01", "S02"],
            "name": ["Alice", "Alice M.", "Bob"],
            "program": ["CS", "CS", "IT"],
        })
        annotated, metrics = detect_duplicates(df, "students")

        assert "is_duplicate_key" in annotated.columns
        assert "is_exact_duplicate" in annotated.columns
        assert "is_conflicting" in annotated.columns
        assert "duplicate_group_id" in annotated.columns

        assert annotated.loc[0, "is_duplicate_key"] == True
        assert annotated.loc[1, "is_duplicate_key"] == True
        assert annotated.loc[2, "is_duplicate_key"] == False

        assert annotated.loc[0, "is_conflicting"] == True
        assert annotated.loc[0, "duplicate_group_id"].startswith("students_grp_")
        assert metrics.duplicate_groups == 1

    def test_clean_duplicates_empty_and_no_duplicates(self):
        """Test handling of empty DataFrame and DataFrame without duplicates."""
        # Empty
        df_empty = pd.DataFrame(columns=["student_id", "name", "program"])
        cleaned_empty, report_empty = clean_duplicates(df_empty, "students")
        assert len(cleaned_empty) == 0
        assert report_empty.rows_removed == 0
        assert report_empty.total_rows_before == 0

        # No duplicates
        df_clean = pd.DataFrame({
            "student_id": ["S01", "S02"],
            "name": ["Alice", "Bob"],
        })
        cleaned, report = clean_duplicates(df_clean, "students")
        assert len(cleaned) == 2
        assert report.rows_removed == 0
        assert report.after_metrics.key_duplicate_rows == 0

    def test_deduplicate_students_most_complete(self):
        """Test deduplicating students by keeping the more complete record."""
        df = pd.DataFrame({
            "student_id": ["S01", "S01", "S02"],
            "name": ["Alice", "Alice Smith", "Bob"],
            "program": [None, "Computer Science", "Information Technology"],
            "year": [2026, 2026, 2026],
        })

        cleaned, report = deduplicate_students(df, strategy="most_complete")

        assert len(cleaned) == 2
        assert report.rows_removed == 1
        # S01 record kept should be the one with 'Computer Science' (2 non-nulls vs 3 non-nulls)
        alice_row = cleaned[cleaned["student_id"] == "S01"].iloc[0]
        assert alice_row["program"] == "Computer Science"
        assert alice_row["name"] == "Alice Smith"

        # Check quarantine
        assert report.quarantined_records is not None
        assert len(report.quarantined_records) == 1
        assert "Superseded by more complete record" in report.quarantined_records.iloc[0]["quarantine_reason"]

    def test_deduplicate_courses(self):
        """Test deduplicating course records."""
        df = pd.DataFrame({
            "course_id": ["CS101", "CS101"],
            "course_name": ["Intro to CS", "Intro to Computer Science"],
            "faculty": ["Engineering", "Engineering"],
        })
        cleaned, report = deduplicate_courses(df, strategy="most_complete")
        assert len(cleaned) == 1
        assert report.rows_removed == 1
        assert cleaned.iloc[0]["course_id"] == "CS101"

    def test_deduplicate_attendance_chronological_last(self):
        """Test deduplicating attendance keeping the latest audit/record."""
        df = pd.DataFrame({
            "student_id": ["S01", "S01"],
            "course_id": ["CS101", "CS101"],
            "date": ["2026-03-01", "2026-03-01"],
            "status": ["Absent", "Present"],  # Corrected later
        })
        cleaned, report = deduplicate_attendance(df, strategy="last")

        assert len(cleaned) == 1
        assert report.rows_removed == 1
        assert cleaned.iloc[0]["status"] == "Present"
        assert "Superseded by" in report.quarantined_records.iloc[0]["quarantine_reason"]

    def test_deduplicate_submissions_highest_score(self):
        """Test deduplicating submissions by retaining the highest valid score."""
        df = pd.DataFrame({
            "assignment_id": ["A01", "A01", "A02"],
            "student_id": ["S01", "S01", "S01"],
            "submission_date": ["2026-03-01", "2026-03-03", "2026-03-02"],
            "score": [65.0, 92.5, 80.0],
        })
        cleaned, report = deduplicate_submissions(df, strategy="highest_score")

        assert len(cleaned) == 2
        assert report.rows_removed == 1
        s01_a01 = cleaned[cleaned["assignment_id"] == "A01"].iloc[0]
        assert s01_a01["score"] == 92.5
        assert "92.5 vs 65.0" in report.quarantined_records.iloc[0]["quarantine_reason"]

    def test_deduplicate_exams_highest_score_and_fallback_keys(self):
        """Test deduplicating exams using fallback business keys and highest score."""
        df = pd.DataFrame({
            "student_id": ["S01", "S01"],
            "course_id": ["CS101", "CS101"],
            "exam_type": ["Final", "Final"],
            "score": [72.0, 88.0],
        })
        cleaned, report = deduplicate_exams(df, strategy="highest_score")

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["score"] == 88.0
        assert report.rows_removed == 1
        assert report.business_keys == ["student_id", "course_id", "exam_type"]

    def test_deduplicate_assignments_and_merge_non_null(self):
        """Test assignments deduplication and merge_non_null strategy."""
        df = pd.DataFrame({
            "assignment_id": ["A01", "A01"],
            "course_id": ["CS101", "CS101"],
            "title": ["Homework 1", "Homework 1"],
            "due_date": [None, "2026-03-15"],
        })
        cleaned, report = deduplicate_assignments(df, strategy="merge_non_null")

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["due_date"] == "2026-03-15"
        assert report.rows_removed == 1

    def test_action_flag_only_does_not_remove_rows(self):
        """Test that action='flag_only' annotates without dropping any records."""
        df = pd.DataFrame({
            "student_id": ["S01", "S01"],
            "name": ["Alice", "Alice"],
            "program": ["CS", "CS"],
            "year": [2026, 2026],
        })
        flagged, report = clean_duplicates(df, "students", action="flag_only")

        assert len(flagged) == 2  # None removed
        assert report.rows_removed == 0
        assert "is_duplicate_key" in flagged.columns
        assert flagged["is_duplicate_key"].all()

    def test_duplicate_report_markdown_and_dict(self):
        """Test DuplicateReport serialization to dict and markdown table."""
        df = pd.DataFrame({
            "student_id": ["S01", "S01", "S02"],
            "name": ["Alice", "Alice", "Bob"],
            "program": ["CS", "CS", "IT"],
            "year": [2026, 2026, 2026],
        })
        _, report = deduplicate_students(df)

        report_dict = report.to_dict()
        assert report_dict["entity_name"] == "students"
        assert report_dict["rows_removed"] == 1
        assert report_dict["rows_retained"] == 2

        md = report.to_markdown()
        assert "### Duplicate Audit Report: `students`" in md
        assert "| Total Records | 3 | 2 | -1 |" in md

    def test_deduplicate_academic_dataset_batch(self):
        """Test batch deduplication across all 6 core entities."""
        batch = {
            "students": pd.DataFrame({
                "student_id": ["S01", "S01", "S02"],
                "name": ["Alice", "Alice", "Bob"],
                "program": ["CS", "CS", "IT"],
                "year": [2026, 2026, 2026],
            }),
            "courses": pd.DataFrame({
                "course_id": ["CS101", "CS102"],
                "course_name": ["Intro CS", "Data Structures"],
                "faculty": ["Engineering", "Engineering"],
            }),
            "attendance": pd.DataFrame({
                "student_id": ["S01", "S01"],
                "course_id": ["CS101", "CS101"],
                "date": ["2026-03-01", "2026-03-01"],
                "status": ["Absent", "Present"],
            }),
            "assignments": pd.DataFrame({
                "assignment_id": ["A01", "A01"],
                "course_id": ["CS101", "CS101"],
                "title": ["HW 1", "HW 1"],
                "due_date": ["2026-03-10", "2026-03-10"],
            }),
            "submissions": pd.DataFrame({
                "assignment_id": ["A01", "A01"],
                "student_id": ["S01", "S01"],
                "submission_date": ["2026-03-09", "2026-03-10"],
                "score": [75.0, 95.0],
            }),
            "exams": pd.DataFrame({
                "exam_id": ["E01", "E01"],
                "student_id": ["S01", "S01"],
                "score": [60.0, 85.0],
            }),
        }

        cleaned_batch, batch_report = deduplicate_academic_dataset(batch)

        assert isinstance(batch_report, BatchDuplicateReport)
        assert len(cleaned_batch) == 6
        assert len(cleaned_batch["students"]) == 2
        assert len(cleaned_batch["courses"]) == 2
        assert len(cleaned_batch["attendance"]) == 1
        assert len(cleaned_batch["assignments"]) == 1
        assert len(cleaned_batch["submissions"]) == 1
        assert len(cleaned_batch["exams"]) == 1

        assert batch_report.total_removed == 5
        batch_md = batch_report.to_markdown()
        assert "Academic Dataset Duplicate Detection & Deduplication Summary" in batch_md
        assert "`students`" in batch_md
        assert "`attendance`" in batch_md
