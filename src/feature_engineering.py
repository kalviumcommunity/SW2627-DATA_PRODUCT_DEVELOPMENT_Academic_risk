"""Academic Student Feature Engineering Engine.

Derives student-level analytical features feeding the academic risk engine:
- attendance_percentage
- assignment_completion_rate
- average_assignment_score
- missing_submission_count
- late_submission_count
- average_exam_score
- recent_exam_score
- attendance_trend
- assignment_trend
- exam_trend

Every calculation is strictly documented, preserving missing values where appropriate
and never converting missing attendance or exam evaluations to zero.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.feature_engineering")

# Documentation metadata for all engineered student features
FEATURE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "attendance_percentage": {
        "formula": "(Present + 0.5 * Late) / (Total Sessions - Excused) * 100",
        "description": "Percentage of class sessions attended. Late arrivals contribute 0.5 weighting.",
        "range": "[0.0, 100.0] or NaN if no sessions",
        "risk_interpretation": "Values < 75.0 indicate severe attendance disengagement risk.",
    },
    "assignment_completion_rate": {
        "formula": "(Submitted Assignments / Total Expected Coursework) * 100",
        "description": "Ratio of required coursework submitted on or before deadline.",
        "range": "[0.0, 100.0] or 100.0 if no assignments exist",
        "risk_interpretation": "Values < 70.0 indicate high coursework abandonment risk.",
    },
    "average_assignment_score": {
        "formula": "mean(valid_assignment_scores)",
        "description": "Mean score across all evaluated assignment submissions.",
        "range": "[0.0, 100.0] or NaN if unattempted",
        "risk_interpretation": "Values < 60.0 signal academic difficulty in formative learning.",
    },
    "missing_submission_count": {
        "formula": "max(0, Total Expected Assignments - Submitted Assignments)",
        "description": "Total number of expected course assignments with no submitted record.",
        "range": "Integer >= 0",
        "risk_interpretation": "Count >= 2 represents persistent submission neglect.",
    },
    "late_submission_count": {
        "formula": "sum(is_late == True or submission_date > due_date)",
        "description": "Number of assignments submitted after the formal deadline.",
        "range": "Integer >= 0",
        "risk_interpretation": "Frequent late submissions indicate time management difficulties.",
    },
    "average_exam_score": {
        "formula": "mean(valid_exam_scores)",
        "description": "Mean score across all summative examinations taken by the student.",
        "range": "[0.0, 100.0] or NaN if unattempted",
        "risk_interpretation": "Values < 50.0 indicate high failure risk on summative assessments.",
    },
    "recent_exam_score": {
        "formula": "score of the chronologically newest exam taken",
        "description": "Score achieved on the most recently completed exam sitting.",
        "range": "[0.0, 100.0] or NaN if unattempted",
        "risk_interpretation": "Low recent scores reveal ongoing academic struggles requiring immediate intervention.",
    },
    "attendance_trend": {
        "formula": "recent_half_attendance_pct - early_half_attendance_pct",
        "description": "Difference in attendance percentage between the second half and first half of sessions.",
        "range": "[-100.0, +100.0] or NaN if < 2 sessions",
        "risk_interpretation": "Negative values (< -5.0 pp) reveal disengagement deterioration over the term.",
    },
    "assignment_trend": {
        "formula": "recent_half_avg_score - early_half_avg_score",
        "description": "Difference in assignment scores between recent coursework and early coursework.",
        "range": "[-100.0, +100.0] or NaN if < 2 submissions",
        "risk_interpretation": "Negative values (< -5.0 pp) reveal declining comprehension or growing coursework difficulty.",
    },
    "exam_trend": {
        "formula": "recent_exam_score - first_exam_score",
        "description": "Difference in score between the most recent exam and the initial exam.",
        "range": "[-100.0, +100.0] or 0.0 if single exam, NaN if no exams",
        "risk_interpretation": "Negative delta indicates performance drops between midterms and subsequent exams.",
    },
}


# ---------------------------------------------------------------------------
# Individual Component Calculators
# ---------------------------------------------------------------------------


def calculate_student_attendance_features(
    students_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate attendance_percentage and attendance_trend per student.

    Calculation Rules:
    - Present counts as 1.0, Late counts as 0.5, Excused is deducted from denominator.
    - If student has 0 recorded sessions, returns NaN.
    - Attendance trend compares the attendance rate in the second half of sessions vs first half.
    """
    if "student_id" not in students_df.columns:
        raise DataValidationError("students_df must contain 'student_id'")

    student_ids = students_df[["student_id"]].drop_duplicates()

    if attendance_df.empty or "student_id" not in attendance_df.columns:
        res = student_ids.copy()
        res["attendance_percentage"] = np.nan
        res["attendance_trend"] = np.nan
        res["attendance_trend_direction"] = "No Data"
        return res

    att = attendance_df.copy()
    status_col = "status" if "status" in att.columns else None
    date_col = "date" if "date" in att.columns else None

    # Sort chronologically for trend analysis
    if date_col:
        att["_parsed_dt"] = pd.to_datetime(att[date_col], errors="coerce")
        att = att.sort_values(by=["student_id", "_parsed_dt"])

    features_list: List[Dict[str, Any]] = []

    for sid, group in att.groupby("student_id"):
        total_sessions = len(group)
        if total_sessions == 0:
            continue

        if status_col:
            statuses = group[status_col].astype(str).str.lower()
            present_cnt = int((statuses == "present").sum())
            late_cnt = int((statuses == "late").sum())
            excused_cnt = int((statuses == "excused").sum())

            effective_sessions = total_sessions - excused_cnt
            if effective_sessions > 0:
                att_pct = ((present_cnt + 0.5 * late_cnt) / effective_sessions) * 100.0
                att_pct = round(min(100.0, max(0.0, att_pct)), 2)
            else:
                att_pct = np.nan

            # Longitudinal Trend (split sessions in half chronologically)
            if total_sessions >= 2 and effective_sessions > 0:
                mid = total_sessions // 2
                first_half = group.iloc[:mid]
                second_half = group.iloc[mid:]

                def calc_half_rate(h_df: pd.DataFrame) -> Optional[float]:
                    h_stat = h_df[status_col].astype(str).str.lower()
                    h_eff = len(h_df) - int((h_stat == "excused").sum())
                    if h_eff > 0:
                        return (
                            (int((h_stat == "present").sum()) + 0.5 * int((h_stat == "late").sum()))
                            / h_eff
                        ) * 100.0
                    return None

                rate1 = calc_half_rate(first_half)
                rate2 = calc_half_rate(second_half)
                if rate1 is not None and rate2 is not None:
                    att_trend = round(rate2 - rate1, 2)
                else:
                    att_trend = 0.0
            else:
                att_trend = 0.0 if total_sessions == 1 else np.nan
        else:
            att_pct = np.nan
            att_trend = np.nan

        # Direction classification
        if pd.isna(att_trend):
            direction = "No Data"
        elif att_trend > 5.0:
            direction = "Improving"
        elif att_trend < -5.0:
            direction = "Declining"
        else:
            direction = "Stable"

        features_list.append({
            "student_id": sid,
            "attendance_percentage": att_pct,
            "attendance_trend": att_trend,
            "attendance_trend_direction": direction,
        })

    features_df = pd.DataFrame(features_list)
    out = student_ids.merge(features_df, on="student_id", how="left")
    out["attendance_trend_direction"] = out["attendance_trend_direction"].fillna("No Data")
    return out


def calculate_student_assignment_features(
    students_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    enrollments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate assignment coursework features:

    - assignment_completion_rate
    - average_assignment_score
    - missing_submission_count
    - late_submission_count
    - assignment_trend
    """
    student_ids = students_df[["student_id"]].drop_duplicates()

    # Determine total expected assignments per student based on enrolled courses
    expected_assignments_map: Dict[str, int] = {}
    if not enrollments_df.empty and not assignments_df.empty and "course_id" in assignments_df.columns:
        # Count assignments per course
        course_assign_counts = assignments_df.groupby("course_id").size().to_dict()
        for sid, group in enrollments_df.groupby("student_id"):
            courses = group["course_id"].dropna().unique()
            expected_total = sum(course_assign_counts.get(cid, 0) for cid in courses)
            expected_assignments_map[sid] = expected_total
    else:
        # Fallback: total unique assignments in assignment catalog
        fallback_total = len(assignments_df["assignment_id"].unique()) if "assignment_id" in assignments_df.columns else 0
        for sid in student_ids["student_id"]:
            expected_assignments_map[sid] = fallback_total

    if submissions_df.empty or "student_id" not in submissions_df.columns:
        res = student_ids.copy()
        res["assignment_completion_rate"] = 0.0
        res["average_assignment_score"] = np.nan
        res["missing_submission_count"] = res["student_id"].map(expected_assignments_map).fillna(0).astype("Int64")
        res["late_submission_count"] = 0
        res["assignment_trend"] = np.nan
        res["assignment_trend_direction"] = "No Data"
        return res

    sub = submissions_df.copy()
    score_col = "score" if "score" in sub.columns else None
    sub["_score_num"] = pd.to_numeric(sub[score_col], errors="coerce") if score_col else np.nan

    # Check is_late
    if "is_late" in sub.columns:
        sub["_is_late_flag"] = sub["is_late"].astype(bool)
    elif "days_late" in sub.columns:
        sub["_is_late_flag"] = pd.to_numeric(sub["days_late"], errors="coerce") > 0
    else:
        sub["_is_late_flag"] = False

    # Check date for trend
    date_col = "submission_date" if "submission_date" in sub.columns else None
    if date_col:
        sub["_parsed_dt"] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.sort_values(by=["student_id", "_parsed_dt"])

    features_list: List[Dict[str, Any]] = []

    for sid, group in sub.groupby("student_id"):
        expected_total = expected_assignments_map.get(sid, len(group))
        submitted_count = len(group)

        # Completion rate
        if expected_total > 0:
            completion_rate = round(min(100.0, (submitted_count / expected_total) * 100.0), 2)
            missing_count = max(0, expected_total - submitted_count)
        else:
            completion_rate = 100.0 if submitted_count > 0 else np.nan
            missing_count = 0

        # Score stats
        valid_scores = group["_score_num"].dropna()
        if len(valid_scores) > 0:
            avg_score = round(float(valid_scores.mean()), 2)
        else:
            avg_score = np.nan

        # Late count
        late_count = int(group["_is_late_flag"].sum())

        # Assignment trend (chronological half-split)
        if len(valid_scores) >= 2:
            mid = len(valid_scores) // 2
            first_half_avg = valid_scores.iloc[:mid].mean()
            second_half_avg = valid_scores.iloc[mid:].mean()
            assign_trend = round(float(second_half_avg - first_half_avg), 2)
        else:
            assign_trend = 0.0 if len(valid_scores) == 1 else np.nan

        if pd.isna(assign_trend):
            direction = "No Data"
        elif assign_trend > 5.0:
            direction = "Improving"
        elif assign_trend < -5.0:
            direction = "Declining"
        else:
            direction = "Stable"

        features_list.append({
            "student_id": sid,
            "assignment_completion_rate": completion_rate,
            "average_assignment_score": avg_score,
            "missing_submission_count": missing_count,
            "late_submission_count": late_count,
            "assignment_trend": assign_trend,
            "assignment_trend_direction": direction,
        })

    features_df = pd.DataFrame(features_list)
    out = student_ids.merge(features_df, on="student_id", how="left")

    # Fill defaults for students with 0 submissions
    out["assignment_completion_rate"] = out["assignment_completion_rate"].fillna(0.0)
    out["missing_submission_count"] = out["missing_submission_count"].fillna(
        out["student_id"].map(expected_assignments_map).fillna(0)
    ).astype("Int64")
    out["late_submission_count"] = out["late_submission_count"].fillna(0).astype("Int64")
    out["assignment_trend_direction"] = out["assignment_trend_direction"].fillna("No Data")

    return out


def calculate_student_exam_features(
    students_df: pd.DataFrame,
    exams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate exam assessment features:

    - average_exam_score
    - recent_exam_score
    - exam_trend
    """
    student_ids = students_df[["student_id"]].drop_duplicates()

    if exams_df.empty or "student_id" not in exams_df.columns:
        res = student_ids.copy()
        res["average_exam_score"] = np.nan
        res["recent_exam_score"] = np.nan
        res["exam_trend"] = np.nan
        res["exam_trend_direction"] = "No Data"
        return res

    ex = exams_df.copy()
    score_col = "score" if "score" in ex.columns else None
    ex["_score_num"] = pd.to_numeric(ex[score_col], errors="coerce") if score_col else np.nan

    # Sort chronologically if date present, else by index
    date_col = None
    for candidate in ["exam_date", "date"]:
        if candidate in ex.columns:
            date_col = candidate
            break

    if date_col:
        ex["_parsed_dt"] = pd.to_datetime(ex[date_col], errors="coerce")
        ex = ex.sort_values(by=["student_id", "_parsed_dt"])

    features_list: List[Dict[str, Any]] = []

    for sid, group in ex.groupby("student_id"):
        valid_scores = group["_score_num"].dropna()
        if len(valid_scores) == 0:
            features_list.append({
                "student_id": sid,
                "average_exam_score": np.nan,
                "recent_exam_score": np.nan,
                "exam_trend": np.nan,
                "exam_trend_direction": "No Data",
            })
            continue

        avg_score = round(float(valid_scores.mean()), 2)
        # Recent exam score is last chronological valid score
        recent_score = round(float(valid_scores.iloc[-1]), 2)

        # Exam trend: recent score vs first score
        if len(valid_scores) >= 2:
            first_score = float(valid_scores.iloc[0])
            trend_val = round(recent_score - first_score, 2)
        else:
            trend_val = 0.0

        if trend_val > 5.0:
            direction = "Improving"
        elif trend_val < -5.0:
            direction = "Declining"
        else:
            direction = "Stable"

        features_list.append({
            "student_id": sid,
            "average_exam_score": avg_score,
            "recent_exam_score": recent_score,
            "exam_trend": trend_val,
            "exam_trend_direction": direction,
        })

    features_df = pd.DataFrame(features_list)
    out = student_ids.merge(features_df, on="student_id", how="left")
    out["exam_trend_direction"] = out["exam_trend_direction"].fillna("No Data")
    return out


# ---------------------------------------------------------------------------
# Master Student Feature Pipeline
# ---------------------------------------------------------------------------


def engineer_student_features(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    enrollments: pd.DataFrame,
    attendance: pd.DataFrame,
    assignments: pd.DataFrame,
    submissions: pd.DataFrame,
    exams: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the complete set of engineered student-level academic features.

    Merges demographic metadata with derived attendance, assignment, exam, and trend metrics.

    Args:
        students: Student records DataFrame.
        courses: Courses catalog DataFrame.
        enrollments: Student-course enrollments DataFrame.
        attendance: Attendance logs DataFrame.
        assignments: Coursework metadata DataFrame.
        submissions: Student submissions DataFrame.
        exams: Exam sittings DataFrame.

    Returns:
        DataFrame indexed by student_id with all 10 core engineered features.
    """
    if not isinstance(students, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame for 'students', got: {type(students).__name__}")
    if "student_id" not in students.columns:
        raise DataValidationError("Students DataFrame missing required 'student_id' column")

    logger.info("Beginning student feature engineering across %d students...", len(students))

    # Base student demographics
    core_cols = [c for c in ["student_id", "name", "program", "year"] if c in students.columns]
    master = students[core_cols].drop_duplicates(subset=["student_id"]).copy()

    # Course enrollment count
    if not enrollments.empty and "student_id" in enrollments.columns:
        enroll_counts = enrollments.groupby("student_id").size().rename("courses_enrolled_count")
        master = master.merge(enroll_counts, on="student_id", how="left")
        master["courses_enrolled_count"] = master["courses_enrolled_count"].fillna(0).astype("Int64")
    else:
        master["courses_enrolled_count"] = 0

    # 1. Attendance Features (attendance_percentage, attendance_trend)
    att_feats = calculate_student_attendance_features(students, attendance)
    master = master.merge(att_feats, on="student_id", how="left")

    # 2. Assignment Features (completion_rate, avg_score, missing_count, late_count, assignment_trend)
    assign_feats = calculate_student_assignment_features(
        students_df=students,
        submissions_df=submissions,
        assignments_df=assignments,
        enrollments_df=enrollments,
    )
    master = master.merge(assign_feats, on="student_id", how="left")

    # 3. Exam Features (average_exam_score, recent_exam_score, exam_trend)
    exam_feats = calculate_student_exam_features(students, exams)
    master = master.merge(exam_feats, on="student_id", how="left")

    logger.info(
        "Successfully engineered %d features for %d students.",
        len(master.columns),
        len(master),
    )

    return master


def generate_feature_dictionary_markdown() -> str:
    """Generate formatted markdown documentation of all engineered features."""
    lines = [
        "# Academic Student Feature Engineering Dictionary",
        "",
        "This document defines the mathematical formulation, operational logic, and academic risk usage "
        "of all student-level features engineered to feed the explainable risk identification engine.",
        "",
        "| Feature Name | Mathematical Formula | Valid Range | Academic Risk Significance |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for name, meta in FEATURE_DEFINITIONS.items():
        lines.append(
            f"| `{name}` | `{meta['formula']}` | `{meta['range']}` | {meta['risk_interpretation']} |"
        )

    lines.append("")
    lines.append("## Feature Formulation Details")
    for name, meta in FEATURE_DEFINITIONS.items():
        lines.append(f"### `{name}`")
        lines.append(f"- **Description**: {meta['description']}")
        lines.append(f"- **Formula**: `{meta['formula']}`")
        lines.append(f"- **Expected Scale**: {meta['range']}")
        lines.append(f"- **Risk Diagnostic**: {meta['risk_interpretation']}")
        lines.append("")

    return "\n".join(lines)
