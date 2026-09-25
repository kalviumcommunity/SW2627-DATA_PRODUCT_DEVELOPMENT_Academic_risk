"""Unit tests for Concept #15: Academic Multi-Source Merge & Data Integration."""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.multi_source_merge import (
    MergeAuditReport,
    aggregate_attendance_by_course,
    aggregate_exams_by_course,
    aggregate_submissions_by_course,
    integrate_academic_data,
)


class TestMultiSourceMerge:
    """Test suite for 7-dataset academic integration, duplicate expansion, and orphan tracking."""

    @pytest.fixture
    def sample_academic_ecosystem(self):
        """Standard 7-dataset academic test fixtures."""
        students = pd.DataFrame({
            "student_id": ["S01", "S02", "S03"],
            "name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
            "program": ["Computer Science", "Information Technology", "Business"],
            "year": [2026, 2026, 2025],
        })

        courses = pd.DataFrame({
            "course_id": ["CS101", "CS102", "HIST101"],  # HIST101 will have no enrollments
            "course_name": ["Intro CS", "Data Structures", "World History"],
            "faculty": ["Engineering", "Engineering", "Arts"],
        })

        # S01 in CS101 & CS102; S02 in CS101; S03 has NO enrollments (unmatched student)
        enrollments = pd.DataFrame({
            "student_id": ["S01", "S01", "S02"],
            "course_id": ["CS101", "CS102", "CS101"],
        })

        attendance = pd.DataFrame({
            "student_id": ["S01", "S01", "S01", "S02", "S99"],  # S99 is an orphan student
            "course_id": ["CS101", "CS101", "CS102", "CS101", "CS101"],
            "date": ["2026-03-01", "2026-03-02", "2026-03-01", "2026-03-01", "2026-03-01"],
            "status": ["Present", "Absent", "Present", "Present", "Present"],
        })

        assignments = pd.DataFrame({
            "assignment_id": ["A01", "A02", "A03"],
            "course_id": ["CS101", "CS101", "CS102"],
            "title": ["HW 1", "HW 2", "Project 1"],
            "due_date": ["2026-03-10", "2026-03-20", "2026-03-25"],
        })

        submissions = pd.DataFrame({
            "assignment_id": ["A01", "A02", "A03", "A01"],
            "student_id": ["S01", "S01", "S01", "S99"],  # S99 is orphan; S02 has NO submissions
            "score": [90.0, 85.0, 95.0, 70.0],
            "is_late": [False, True, False, False],
        })

        exams = pd.DataFrame({
            "exam_id": ["E01", "E02"],
            "student_id": ["S01", "S01"],
            "course_id": ["CS101", "CS101"],
            "exam_type": ["Midterm", "Final"],
            "score": [88.0, 92.0],
        })

        return {
            "students": students,
            "courses": courses,
            "enrollments": enrollments,
            "attendance": attendance,
            "assignments": assignments,
            "submissions": submissions,
            "exams": exams,
        }

    def test_aggregate_attendance_by_course(self, sample_academic_ecosystem):
        """Test attendance aggregation avoids row explosion and computes rates."""
        att = sample_academic_ecosystem["attendance"]
        agg = aggregate_attendance_by_course(att)

        assert len(agg) == 4  # (S01, CS101), (S01, CS102), (S02, CS101), (S99, CS101)
        # S01 in CS101 has 2 sessions: 1 present, 1 absent => 50%
        s01_cs101 = agg[(agg["student_id"] == "S01") & (agg["course_id"] == "CS101")].iloc[0]
        assert s01_cs101["attendance_sessions"] == 2
        assert s01_cs101["attendance_present"] == 1
        assert s01_cs101["attendance_absent"] == 1
        assert s01_cs101["attendance_rate"] == 50.0

    def test_aggregate_submissions_by_course(self, sample_academic_ecosystem):
        """Test submissions aggregation calculates statistics joined with assignments."""
        sub = sample_academic_ecosystem["submissions"]
        assign = sample_academic_ecosystem["assignments"]
        agg = aggregate_submissions_by_course(sub, assign)

        # S01 in CS101 submitted A01 (90.0) and A02 (85.0) => avg = 87.5
        s01_cs101 = agg[(agg["student_id"] == "S01") & (agg["course_id"] == "CS101")].iloc[0]
        assert s01_cs101["assignments_submitted_count"] == 2
        assert s01_cs101["assignments_avg_score"] == 87.5
        assert s01_cs101["assignments_min_score"] == 85.0
        assert s01_cs101["assignments_max_score"] == 90.0
        assert s01_cs101["assignments_late_count"] == 1  # A02 was late

    def test_aggregate_exams_by_course(self, sample_academic_ecosystem):
        """Test exams aggregation derives midterm and final scores."""
        exams = sample_academic_ecosystem["exams"]
        agg = aggregate_exams_by_course(exams)

        s01_cs101 = agg[(agg["student_id"] == "S01") & (agg["course_id"] == "CS101")].iloc[0]
        assert s01_cs101["exams_taken_count"] == 2
        assert s01_cs101["exams_avg_score"] == 90.0
        assert s01_cs101["exam_midterm_score"] == 88.0
        assert s01_cs101["exam_final_score"] == 92.0

    def test_integrate_academic_data_no_duplicate_expansion(self, sample_academic_ecosystem):
        """Test that multi-source integration avoids Cartesian product explosion."""
        eco = sample_academic_ecosystem
        integrated, audit = integrate_academic_data(
            students=eco["students"],
            courses=eco["courses"],
            enrollments=eco["enrollments"],
            attendance=eco["attendance"],
            assignments=eco["assignments"],
            submissions=eco["submissions"],
            exams=eco["exams"],
        )

        assert isinstance(audit, MergeAuditReport)
        assert audit.duplicate_expansion_detected == False

        # Enrolled student-courses = 3 (S01 in CS101, S01 in CS102, S02 in CS101)
        # Unenrolled student = 1 (S03)
        # Total rows = 4
        assert len(integrated) == 4
        assert audit.output_counts["enrolled_records"] == 3
        assert audit.output_counts["unenrolled_students"] == 1

    def test_integrate_academic_data_does_not_hide_unmatched_and_orphans(self, sample_academic_ecosystem):
        """Test that unmatched students/courses and orphan activity records are preserved and reported."""
        eco = sample_academic_ecosystem
        integrated, audit = integrate_academic_data(**eco)

        # 1. Unmatched student S03 is NOT deleted or hidden
        assert "S03" in audit.unmatched_students
        s03_row = integrated[integrated["student_id"] == "S03"].iloc[0]
        assert s03_row["enrollment_status"] == "Unenrolled"
        assert s03_row["name"] == "Charlie Brown"

        # 2. Unmatched course HIST101 is tracked
        assert "HIST101" in audit.unmatched_courses

        # 3. Orphan attendance student S99 is captured, not hidden
        assert "S99" in audit.orphan_attendance_students
        assert "S99" in audit.orphan_submission_students
        assert "orphan_attendance" in audit.orphan_records
        assert len(audit.orphan_records["orphan_attendance"]) == 1

        # 4. Missing relationships detected
        # S02 is enrolled in CS101 but has 0 submissions and 0 exams
        assert "S02" in audit.students_with_no_submissions
        assert "S02" in audit.students_with_no_exams

    def test_merge_audit_report_markdown_and_dict(self, sample_academic_ecosystem):
        """Test MergeAuditReport serialization and markdown rendering."""
        eco = sample_academic_ecosystem
        _, audit = integrate_academic_data(**eco)

        d = audit.to_dict()
        assert d["total_input_rows"] > 0
        assert d["has_orphans"] == True
        assert d["unmatched_students_count"] == 1

        md = audit.to_markdown()
        assert "Academic Multi-Source Merge Audit Report" in md
        assert "Dataset Row Counts: Before & After Integration" in md
        assert "Orphan Records Audit" in md
        assert "`orphan_attendance`" in md
