"""Unit tests for Concept #16: Student Feature Engineering."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.feature_engineering import (
    FEATURE_DEFINITIONS,
    calculate_student_assignment_features,
    calculate_student_attendance_features,
    calculate_student_exam_features,
    engineer_student_features,
    generate_feature_dictionary_markdown,
)


class TestFeatureEngineering:
    """Test suite for academic student feature engineering and trend calculations."""

    @pytest.fixture
    def academic_fixtures(self):
        """Mock student academic data for feature verification."""
        students = pd.DataFrame({
            "student_id": ["S01", "S02", "S03"],
            "name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
            "program": ["CS", "IT", "CS"],
            "year": [2026, 2026, 2025],
        })

        courses = pd.DataFrame({
            "course_id": ["CS101", "CS102"],
            "course_name": ["Intro CS", "Data Structures"],
            "faculty": ["Engineering", "Engineering"],
        })

        # S01 in CS101 & CS102 (expected 2 + 1 = 3 assignments); S02 in CS101 (expected 2); S03 unenrolled
        enrollments = pd.DataFrame({
            "student_id": ["S01", "S01", "S02"],
            "course_id": ["CS101", "CS102", "CS101"],
        })

        # S01: 4 sessions (Present, Present, Absent, Absent) => 50% attendance; trend: early 100%, recent 0% => -100.0 (Declining)
        # S02: 2 sessions (Present, Late) => (1 + 0.5)/2 = 75%; trend: early 100%, recent 50% => -50.0 (Declining)
        # S03: 0 sessions => attendance_percentage is NaN (never 0!)
        attendance = pd.DataFrame({
            "student_id": ["S01", "S01", "S01", "S01", "S02", "S02"],
            "course_id": ["CS101", "CS101", "CS101", "CS101", "CS101", "CS101"],
            "date": [
                "2026-02-01", "2026-02-08", "2026-03-01", "2026-03-08",
                "2026-02-01", "2026-02-08",
            ],
            "status": ["Present", "Present", "Absent", "Absent", "Present", "Late"],
        })

        assignments = pd.DataFrame({
            "assignment_id": ["A01", "A02", "A03"],
            "course_id": ["CS101", "CS101", "CS102"],
            "title": ["HW 1", "HW 2", "Project 1"],
            "due_date": ["2026-02-15", "2026-03-01", "2026-03-15"],
        })

        # S01 submitted all 3 assignments: A01 (70.0), A02 (90.0, late), A03 (80.0)
        # S02 submitted 1 of 2: A01 (85.0); missing A02
        # S03 submitted 0 assignments
        submissions = pd.DataFrame({
            "assignment_id": ["A01", "A02", "A03", "A01"],
            "student_id": ["S01", "S01", "S01", "S02"],
            "score": [70.0, 90.0, 80.0, 85.0],
            "submission_date": ["2026-02-14", "2026-03-02", "2026-03-14", "2026-02-15"],
            "is_late": [False, True, False, False],
        })

        # S01: 2 exams: Midterm (75.0, 2026-03-01), Final (85.0, 2026-05-01) => avg 80.0, recent 85.0, trend +10.0
        # S02: 1 exam: Midterm (60.0, 2026-03-01) => avg 60.0, recent 60.0, trend 0.0
        # S03: 0 exams => NaN
        exams = pd.DataFrame({
            "exam_id": ["E01", "E02", "E03"],
            "student_id": ["S01", "S01", "S02"],
            "course_id": ["CS101", "CS101", "CS101"],
            "exam_date": ["2026-03-01", "2026-05-01", "2026-03-01"],
            "score": [75.0, 85.0, 60.0],
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

    def test_all_ten_features_defined_in_feature_definitions(self):
        """Verify all 10 required features are defined in FEATURE_DEFINITIONS metadata."""
        expected_features = [
            "attendance_percentage",
            "assignment_completion_rate",
            "average_assignment_score",
            "missing_submission_count",
            "late_submission_count",
            "average_exam_score",
            "recent_exam_score",
            "attendance_trend",
            "assignment_trend",
            "exam_trend",
        ]
        for feat in expected_features:
            assert feat in FEATURE_DEFINITIONS
            meta = FEATURE_DEFINITIONS[feat]
            assert "formula" in meta
            assert "description" in meta
            assert "risk_interpretation" in meta

    def test_calculate_student_attendance_features(self, academic_fixtures):
        """Test attendance_percentage and attendance_trend calculations."""
        fix = academic_fixtures
        att_feats = calculate_student_attendance_features(fix["students"], fix["attendance"])

        assert len(att_feats) == 3
        # S01: 4 sessions: 2 present, 2 absent => 50.0%
        s01 = att_feats[att_feats["student_id"] == "S01"].iloc[0]
        assert s01["attendance_percentage"] == 50.0
        # Early: 2 present = 100%, Recent: 2 absent = 0% => trend -100.0 (Declining)
        assert s01["attendance_trend"] == -100.0
        assert s01["attendance_trend_direction"] == "Declining"

        # S02: 2 sessions: 1 present, 1 late (0.5) => 1.5 / 2 = 75.0%
        s02 = att_feats[att_feats["student_id"] == "S02"].iloc[0]
        assert s02["attendance_percentage"] == 75.0

        # S03: No attendance sessions => strictly NaN (never 0!)
        s03 = att_feats[att_feats["student_id"] == "S03"].iloc[0]
        assert pd.isna(s03["attendance_percentage"])
        assert pd.isna(s03["attendance_trend"])
        assert s03["attendance_trend_direction"] == "No Data"

    def test_calculate_student_assignment_features(self, academic_fixtures):
        """Test coursework assignment features and trends."""
        fix = academic_fixtures
        assign_feats = calculate_student_assignment_features(
            students_df=fix["students"],
            submissions_df=fix["submissions"],
            assignments_df=fix["assignments"],
            enrollments_df=fix["enrollments"],
        )

        assert len(assign_feats) == 3

        # S01 expected 3 assignments, submitted 3
        s01 = assign_feats[assign_feats["student_id"] == "S01"].iloc[0]
        assert s01["assignment_completion_rate"] == 100.0
        assert s01["average_assignment_score"] == 80.0  # (70 + 90 + 80) / 3
        assert s01["missing_submission_count"] == 0
        assert s01["late_submission_count"] == 1  # A02 was late

        # S02 expected 2 assignments (CS101), submitted 1
        s02 = assign_feats[assign_feats["student_id"] == "S02"].iloc[0]
        assert s02["assignment_completion_rate"] == 50.0
        assert s02["missing_submission_count"] == 1
        assert s02["average_assignment_score"] == 85.0
        assert s02["late_submission_count"] == 0

        # S03 has 0 submissions
        s03 = assign_feats[assign_feats["student_id"] == "S03"].iloc[0]
        assert s03["assignment_completion_rate"] == 0.0
        assert pd.isna(s03["average_assignment_score"])  # NaN, never 0!
        assert s03["missing_submission_count"] == 0  # Unenrolled, 0 expected

    def test_calculate_student_exam_features(self, academic_fixtures):
        """Test exam average, recent score, and trend."""
        fix = academic_fixtures
        exam_feats = calculate_student_exam_features(fix["students"], fix["exams"])

        # S01: 75.0, 85.0
        s01 = exam_feats[exam_feats["student_id"] == "S01"].iloc[0]
        assert s01["average_exam_score"] == 80.0
        assert s01["recent_exam_score"] == 85.0
        assert s01["exam_trend"] == 10.0  # 85 - 75 = +10.0
        assert s01["exam_trend_direction"] == "Improving"

        # S02: Single exam 60.0
        s02 = exam_feats[exam_feats["student_id"] == "S02"].iloc[0]
        assert s02["average_exam_score"] == 60.0
        assert s02["recent_exam_score"] == 60.0
        assert s02["exam_trend"] == 0.0
        assert s02["exam_trend_direction"] == "Stable"

        # S03: 0 exams => NaN
        s03 = exam_feats[exam_feats["student_id"] == "S03"].iloc[0]
        assert pd.isna(s03["average_exam_score"])
        assert pd.isna(s03["recent_exam_score"])
        assert pd.isna(s03["exam_trend"])
        assert s03["exam_trend_direction"] == "No Data"

    def test_engineer_student_features_master_pipeline(self, academic_fixtures):
        """Test master feature engineering pipeline output and column completeness."""
        fix = academic_fixtures
        features_df = engineer_student_features(**fix)

        assert len(features_df) == 3
        # Check that all 10 core features exist
        required_features = [
            "attendance_percentage",
            "assignment_completion_rate",
            "average_assignment_score",
            "missing_submission_count",
            "late_submission_count",
            "average_exam_score",
            "recent_exam_score",
            "attendance_trend",
            "assignment_trend",
            "exam_trend",
        ]
        for feat in required_features:
            assert feat in features_df.columns, f"Feature '{feat}' missing from master features DataFrame"

        # Check demographic columns preserved
        for col in ["student_id", "name", "program", "year", "courses_enrolled_count"]:
            assert col in features_df.columns

    def test_documentation_file_exists_and_covers_all_features(self):
        """Test that docs/FEATURE_ENGINEERING.md exists and documents every feature."""
        doc_path = Path("docs/FEATURE_ENGINEERING.md")
        assert doc_path.exists(), "docs/FEATURE_ENGINEERING.md must exist"

        content = doc_path.read_text(encoding="utf-8")
        for feat in FEATURE_DEFINITIONS.keys():
            assert feat in content, f"Feature '{feat}' must be documented in docs/FEATURE_ENGINEERING.md"

        # Check markdown generator function works
        gen_md = generate_feature_dictionary_markdown()
        assert "# Academic Student Feature Engineering Dictionary" in gen_md
        assert "attendance_percentage" in gen_md
