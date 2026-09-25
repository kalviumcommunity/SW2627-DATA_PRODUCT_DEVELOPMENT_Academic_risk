"""Unit tests for Concept #14: Academic Data Consistency & Relational Integrity."""

import numpy as np
import pandas as pd
import pytest

from src.data_consistency import (
    APPROVED_ATTENDANCE_STATUSES,
    AcademicConsistencyReport,
    ConsistencyIssue,
    EntityConsistencyReport,
    validate_academic_consistency,
    validate_approved_attendance_statuses,
    validate_attendance_range,
    validate_dates_logical,
    validate_enrollment_relationships,
    validate_entity_consistency,
    validate_id_format,
    validate_referential_integrity,
    validate_scores_range,
    validate_submission_timing_logic,
)
from src.exceptions import DataValidationError


class TestDataConsistency:
    """Test suite for academic consistency rules, referential integrity, and reporting."""

    def test_validate_scores_range_valid_and_invalid(self):
        """Test scores bounded strictly between 0 and 100."""
        # Valid scores
        df_valid = pd.DataFrame({"score": [0.0, 50.0, 75.5, 100.0, None]})
        assert validate_scores_range(df_valid, "score") is None

        # Invalid scores: negative and >100
        df_invalid = pd.DataFrame({"score": [85.0, -10.0, 105.0, 90.0]})
        issue = validate_scores_range(df_invalid, "score", entity_name="exams")
        assert isinstance(issue, ConsistencyIssue)
        assert issue.violating_count == 2
        assert issue.severity == "ERROR"
        assert issue.rule_name == "scores_between_0_and_100"

    def test_validate_attendance_range(self):
        """Test attendance rates and percentages bounded correctly."""
        df_valid = pd.DataFrame({"attendance_pct": [0.0, 50.0, 100.0]})
        assert validate_attendance_range(df_valid) is None

        df_invalid = pd.DataFrame({"attendance_pct": [95.0, -5.0, 120.0]})
        issue = validate_attendance_range(df_invalid)
        assert issue is not None
        assert issue.violating_count == 2
        assert issue.severity == "ERROR"

    def test_validate_approved_attendance_statuses(self):
        """Test approved attendance statuses validation."""
        df_valid = pd.DataFrame({"status": ["Present", "Absent", "Late", "Excused", "Unrecorded"]})
        assert validate_approved_attendance_statuses(df_valid) is None

        df_invalid = pd.DataFrame({"status": ["Present", "Invalid_Status", "TBD"]})
        issue = validate_approved_attendance_statuses(df_invalid)
        assert issue is not None
        assert issue.violating_count == 2
        assert issue.severity == "ERROR"

    def test_validate_dates_logical(self):
        """Test date validation for parseability and calendar epoch bounds."""
        df_valid = pd.DataFrame({"due_date": ["2026-03-01", "2026-04-15 23:59:00"]})
        assert validate_dates_logical(df_valid, "due_date", "assignments", "due_dates_valid") is None

        # Unparseable and out of bounds (year 1900)
        df_invalid = pd.DataFrame({"due_date": ["2026-03-01", "not-a-date", "1900-01-01"]})
        issue = validate_dates_logical(
            df_invalid, "due_date", "assignments", "due_dates_valid", min_date="2020-01-01"
        )
        assert issue is not None
        assert issue.violating_count == 2
        assert issue.severity == "ERROR"

    def test_validate_submission_timing_logic(self):
        """Test submission timing logical plausibility relative to assignment due date."""
        df_sub = pd.DataFrame({
            "assignment_id": ["A01", "A01"],
            "submission_date": ["2026-03-10", "2028-05-01"],  # 2 years late
        })
        df_assign = pd.DataFrame({
            "assignment_id": ["A01"],
            "due_date": ["2026-03-10"],
        })

        issue = validate_submission_timing_logic(df_sub, df_assign)
        assert issue is not None
        assert issue.violating_count == 1
        assert issue.severity == "WARNING"

    def test_validate_id_format(self):
        """Test ID format validation."""
        df_valid = pd.DataFrame({"student_id": ["S001", "STU_102", "STU-303"]})
        assert validate_id_format(df_valid, "student_id", "students") is None

        # IDs with invalid special characters or spaces
        df_invalid = pd.DataFrame({"student_id": ["S001", "stu 102", "STU@303!"]})
        issue = validate_id_format(df_invalid, "student_id", "students")
        assert issue is not None
        assert issue.violating_count == 2
        assert issue.severity == "ERROR"

    def test_validate_referential_integrity(self):
        """Test foreign key referential integrity between parent and child tables."""
        df_students = pd.DataFrame({"student_id": ["S01", "S02"]})
        df_attendance_valid = pd.DataFrame({"student_id": ["S01", "S02", "S01"]})
        assert validate_referential_integrity(
            df_attendance_valid, df_students, "student_id", "attendance", "students"
        ) is None

        # Orphan student_id S99
        df_attendance_orphan = pd.DataFrame({"student_id": ["S01", "S99"]})
        issue = validate_referential_integrity(
            df_attendance_orphan, df_students, "student_id", "attendance", "students"
        )
        assert issue is not None
        assert issue.violating_count == 1
        assert issue.severity == "ERROR"
        assert "orphan records" in issue.description

    def test_validate_enrollment_relationships(self):
        """Test validation of student enrollment in course activities."""
        df_enrollments = pd.DataFrame({
            "student_id": ["S01", "S02"],
            "course_id": ["CS101", "CS101"],
        })
        df_attendance_valid = pd.DataFrame({
            "student_id": ["S01", "S02"],
            "course_id": ["CS101", "CS101"],
        })
        assert validate_enrollment_relationships(df_attendance_valid, df_enrollments) is None

        # S01 attending CS202 without enrollment
        df_attendance_unauthorized = pd.DataFrame({
            "student_id": ["S01", "S01"],
            "course_id": ["CS101", "CS202"],
        })
        issue = validate_enrollment_relationships(df_attendance_unauthorized, df_enrollments)
        assert issue is not None
        assert issue.violating_count == 1
        assert issue.severity == "ERROR"

    def test_validate_entity_consistency(self):
        """Test full entity-level consistency validator."""
        df_exams = pd.DataFrame({
            "student_id": ["S01", "S02"],
            "course_id": ["CS101", "CS101"],
            "score": [85.0, 92.0],
            "exam_date": ["2026-03-15", "2026-03-15"],
        })
        report = validate_entity_consistency(df_exams, "exams")
        assert isinstance(report, EntityConsistencyReport)
        assert report.is_valid == True
        assert report.passed_rules >= 3

    def test_validate_academic_consistency_batch_and_strict(self):
        """Test multi-dataset system-wide consistency audit."""
        batch_valid = {
            "students": pd.DataFrame({"student_id": ["S01", "S02"], "name": ["Alice", "Bob"]}),
            "courses": pd.DataFrame({"course_id": ["CS101"], "course_name": ["Intro CS"]}),
            "enrollments": pd.DataFrame({"student_id": ["S01", "S02"], "course_id": ["CS101", "CS101"]}),
            "attendance": pd.DataFrame({
                "student_id": ["S01", "S02"],
                "course_id": ["CS101", "CS101"],
                "date": ["2026-03-01", "2026-03-01"],
                "status": ["Present", "Absent"],
            }),
            "exams": pd.DataFrame({
                "student_id": ["S01"],
                "course_id": ["CS101"],
                "score": [88.0],
            }),
        }

        report_valid = validate_academic_consistency(batch_valid, strict=False)
        assert isinstance(report_valid, AcademicConsistencyReport)
        assert report_valid.is_valid == True
        assert report_valid.total_errors == 0

        # Inject referential violation and out-of-bounds score
        batch_invalid = dict(batch_valid)
        batch_invalid["exams"] = pd.DataFrame({
            "student_id": ["S999"],  # Orphan student
            "course_id": ["CS101"],
            "score": [125.0],        # Score > 100
        })

        report_invalid = validate_academic_consistency(batch_invalid, strict=False)
        assert report_invalid.is_valid == False
        assert report_invalid.total_errors >= 2

        # Check strict mode raises error
        with pytest.raises(DataValidationError):
            validate_academic_consistency(batch_invalid, strict=True)

        # Markdown report rendering
        md = report_invalid.to_markdown()
        assert "Academic Data Consistency & Relational Integrity Audit: NON-COMPLIANT" in md
        assert "`exams`" in md
