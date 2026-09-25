"""Unit tests for Concept #12: Academic Date/Time Transformation Pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.datetime_pipeline import (
    AcademicCalendar,
    DEFAULT_ACADEMIC_CALENDAR,
    calculate_submission_latency,
    derive_date_features,
    transform_academic_datetimes,
    transform_assignment_dates,
    transform_attendance_dates,
    transform_exam_dates,
    transform_intervention_dates,
    transform_submission_dates,
)
from src.exceptions import DataValidationError


class TestDatetimePipeline:
    """Test suite for academic date transformations, calendar calculations, and latency."""

    def test_academic_calendar_milestones_and_validation(self):
        """Test academic calendar week and stage derivations."""
        cal = AcademicCalendar(
            term_name="Spring 2026",
            term_start="2026-01-15",
            term_end="2026-05-30",
        )

        # Pre-term
        dt_pre = pd.Timestamp("2026-01-10")
        assert cal.get_academic_week(dt_pre) == 0
        assert cal.get_academic_period(dt_pre) == "Pre-Term"

        # Early Term (Week 1: Jan 15 to Jan 21)
        dt_w1 = pd.Timestamp("2026-01-18")
        assert cal.get_academic_week(dt_w1) == 1
        assert cal.get_academic_period(dt_w1) == "Early Term"

        # Mid Term (e.g. Week 7: around early March)
        dt_w7 = pd.Timestamp("2026-03-02")
        w7 = cal.get_academic_week(dt_w7)
        assert w7 == 7
        assert cal.get_academic_period(dt_w7) == "Mid Term"

        # Late Term (Week 12: April)
        dt_w12 = pd.Timestamp("2026-04-10")
        assert cal.get_academic_period(dt_w12) == "Late Term"

        # Finals (Week 16: mid-May)
        dt_finals = pd.Timestamp("2026-05-15")
        assert cal.get_academic_period(dt_finals) == "Finals / End Term"

        # Academic Term name
        assert cal.get_academic_term(pd.Timestamp("2026-03-01")) == "Spring 2026"
        assert cal.get_academic_term(pd.Timestamp("2026-10-15")) == "Fall 2026"

        # Invalid calendar validation
        with pytest.raises(DataValidationError):
            AcademicCalendar(term_start="2026-06-01", term_end="2026-01-01")

    def test_derive_date_features_comprehensive(self):
        """Test temporal feature derivation from a date column."""
        # 2026-03-06 is a Friday, 2026-03-07 is a Saturday (weekend)
        df = pd.DataFrame({
            "record_id": [1, 2, 3],
            "event_date": ["2026-03-06", "2026-03-07", "invalid-date"],
        })

        out = derive_date_features(df, date_column="event_date", prefix="evt")

        # Column names created
        expected_cols = [
            "evt_day",
            "evt_month",
            "evt_month_name",
            "evt_year",
            "evt_week",
            "evt_weekday",
            "evt_weekday_num",
            "evt_is_weekend",
            "evt_academic_week",
            "evt_academic_period",
            "evt_academic_term",
        ]
        for col in expected_cols:
            assert col in out.columns

        # Verify Friday
        assert out.loc[0, "evt_day"] == 6
        assert out.loc[0, "evt_month"] == 3
        assert out.loc[0, "evt_month_name"] == "March"
        assert out.loc[0, "evt_year"] == 2026
        assert out.loc[0, "evt_weekday"] == "Friday"
        assert out.loc[0, "evt_weekday_num"] == 4
        assert out.loc[0, "evt_is_weekend"] == False

        # Verify Saturday (Weekend)
        assert out.loc[1, "evt_weekday"] == "Saturday"
        assert out.loc[1, "evt_weekday_num"] == 5
        assert out.loc[1, "evt_is_weekend"] == True

        # Verify Invalid date safely handles nulls without crashing
        assert pd.isna(out.loc[2, "evt_day"])
        assert pd.isna(out.loc[2, "evt_weekday"])
        assert pd.isna(out.loc[2, "evt_is_weekend"])

    def test_derive_date_features_empty_df_and_missing_col(self):
        """Test edge cases with empty DataFrame and missing date column."""
        df_empty = pd.DataFrame(columns=["date"])
        out_empty = derive_date_features(df_empty, "date")
        assert len(out_empty) == 0
        assert "date_day" in out_empty.columns

        with pytest.raises(DataValidationError):
            derive_date_features(pd.DataFrame({"x": [1]}), "non_existent_date")

    def test_transform_attendance_dates(self):
        """Test attendance date transformation pipeline."""
        df_att = pd.DataFrame({
            "student_id": ["S01", "S02"],
            "course_id": ["CS101", "CS101"],
            "date": ["2026-02-02", "2026-02-06"],
            "status": ["Present", "Absent"],
        })
        out = transform_attendance_dates(df_att)

        assert "attendance_weekday" in out.columns
        assert "attendance_academic_week" in out.columns
        assert "attendance_academic_period" in out.columns
        assert out.loc[0, "attendance_weekday"] == "Monday"
        assert out.loc[1, "attendance_weekday"] == "Friday"

    def test_transform_assignment_dates(self):
        """Test assignment due dates transformation."""
        df_assign = pd.DataFrame({
            "assignment_id": ["A01"],
            "course_id": ["CS101"],
            "due_date": ["2026-03-15 23:59:00"],
        })
        out = transform_assignment_dates(df_assign)

        assert "due_day" in out.columns
        assert out.loc[0, "due_day"] == 15
        assert out.loc[0, "due_month_name"] == "March"

    def test_transform_submission_dates(self):
        """Test submission timestamps transformation."""
        df_sub = pd.DataFrame({
            "assignment_id": ["A01"],
            "student_id": ["S01"],
            "submission_date": ["2026-03-14 18:30:00"],
        })
        out = transform_submission_dates(df_sub)

        assert "submission_weekday" in out.columns
        assert out.loc[0, "submission_weekday"] == "Saturday"
        assert out.loc[0, "submission_is_weekend"] == True

    def test_transform_exam_and_intervention_dates(self):
        """Test exam and intervention date transformations."""
        df_exam = pd.DataFrame({
            "exam_id": ["E01"],
            "student_id": ["S01"],
            "course_id": ["CS101"],
            "exam_date": ["2026-03-05"],
        })
        out_exam = transform_exam_dates(df_exam)
        assert "exam_academic_period" in out_exam.columns
        assert out_exam.loc[0, "exam_weekday"] == "Thursday"

        df_interv = pd.DataFrame({
            "student_id": ["S01"],
            "intervention_date": ["2026-02-20"],
            "action": ["Advising session"],
        })
        out_interv = transform_intervention_dates(df_interv)
        assert "intervention_day" in out_interv.columns
        assert out_interv.loc[0, "intervention_day"] == 20

    def test_calculate_submission_latency_internal_due_date(self):
        """Test submission latency calculation when due_date is already in submissions."""
        # Due date: 2026-03-10 12:00:00
        # Student 1: submitted 2 days early (2026-03-08 12:00:00)
        # Student 2: submitted on time (2026-03-10 11:30:00)
        # Student 3: submitted 30 mins late (grace period)
        # Student 4: submitted 10 hours late
        # Student 5: submitted 2 days late (severely late)
        # Student 6: unsubmitted
        df_sub = pd.DataFrame({
            "assignment_id": ["A1"] * 6,
            "student_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "due_date": ["2026-03-10 12:00:00"] * 6,
            "submission_date": [
                "2026-03-08 12:00:00",
                "2026-03-10 11:30:00",
                "2026-03-10 12:30:00",
                "2026-03-10 22:00:00",
                "2026-03-12 12:00:00",
                None,
            ],
        })

        out = calculate_submission_latency(df_sub)

        assert "days_late" in out.columns
        assert "hours_late" in out.columns
        assert "is_late" in out.columns
        assert "submission_timeliness" in out.columns

        # S1: Early (>24h)
        assert out.loc[0, "days_late"] == -2.0
        assert out.loc[0, "is_late"] == False
        assert out.loc[0, "submission_timeliness"] == "Early (>24h)"

        # S2: On Time
        assert out.loc[1, "days_late"] < 0
        assert out.loc[1, "is_late"] == False
        assert out.loc[1, "submission_timeliness"] == "On Time"

        # S3: Grace Period (<=1h late)
        assert out.loc[2, "hours_late"] == 0.5
        assert out.loc[2, "is_late"] == True
        assert out.loc[2, "submission_timeliness"] == "Grace Period (<=1h late)"

        # S4: Late (1-24h late)
        assert out.loc[3, "hours_late"] == 10.0
        assert out.loc[3, "is_late"] == True
        assert out.loc[3, "submission_timeliness"] == "Late (1-24h late)"

        # S5: Severely Late (>24h late)
        assert out.loc[4, "days_late"] == 2.0
        assert out.loc[4, "is_late"] == True
        assert out.loc[4, "submission_timeliness"] == "Severely Late (>24h late)"

        # S6: Unsubmitted
        assert pd.isna(out.loc[5, "is_late"])
        assert out.loc[5, "submission_timeliness"] == "Unsubmitted"

    def test_calculate_submission_latency_with_assignments_df_join(self):
        """Test submission latency when joining due_date from assignments_df."""
        df_assign = pd.DataFrame({
            "assignment_id": ["A01"],
            "due_date": ["2026-03-10 23:59:00"],
        })
        df_sub = pd.DataFrame({
            "assignment_id": ["A01"],
            "student_id": ["S01"],
            "submission_date": ["2026-03-11 02:00:00"],
        })

        out = calculate_submission_latency(df_sub, assignments_df=df_assign)

        assert "due_date" in out.columns
        assert out.loc[0, "is_late"] == True
        assert out.loc[0, "submission_timeliness"] == "Grace Period (<=1h late)" or out.loc[0, "hours_late"] > 1

    def test_transform_academic_datetimes_batch(self):
        """Test batch transformation across multiple academic tables."""
        batch = {
            "attendance": pd.DataFrame({
                "student_id": ["S01"],
                "course_id": ["CS101"],
                "date": ["2026-03-02"],
                "status": ["Present"],
            }),
            "assignments": pd.DataFrame({
                "assignment_id": ["A01"],
                "course_id": ["CS101"],
                "due_date": ["2026-03-15"],
            }),
            "submissions": pd.DataFrame({
                "assignment_id": ["A01"],
                "student_id": ["S01"],
                "submission_date": ["2026-03-14"],
            }),
            "exams": pd.DataFrame({
                "exam_id": ["E01"],
                "student_id": ["S01"],
                "course_id": ["CS101"],
                "exam_date": ["2026-03-20"],
            }),
            "interventions": pd.DataFrame({
                "student_id": ["S01"],
                "intervention_date": ["2026-03-25"],
            }),
        }

        transformed = transform_academic_datetimes(batch)

        assert len(transformed) == 5
        assert "attendance_weekday" in transformed["attendance"].columns
        assert "due_month_name" in transformed["assignments"].columns
        assert "submission_weekday" in transformed["submissions"].columns
        assert "days_late" in transformed["submissions"].columns  # joined from assignments
        assert "exam_academic_period" in transformed["exams"].columns
        assert "intervention_day" in transformed["interventions"].columns
