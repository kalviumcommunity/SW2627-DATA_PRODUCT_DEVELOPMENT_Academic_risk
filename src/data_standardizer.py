"""Academic data type standardization utilities for Pandas and SQL conformity."""

import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_standardizer")

# Canonical attendance mapping dictionary
ATTENDANCE_STATUS_MAP: Dict[str, str] = {
    "p": "Present",
    "present": "Present",
    "1": "Present",
    "1.0": "Present",
    "y": "Present",
    "yes": "Present",
    "a": "Absent",
    "absent": "Absent",
    "0": "Absent",
    "0.0": "Absent",
    "n": "Absent",
    "no": "Absent",
    "l": "Late",
    "late": "Late",
    "tardy": "Late",
    "e": "Excused",
    "excused": "Excused",
    "ex": "Excused",
    "medical": "Excused",
    "unrecorded": "Unrecorded",
    "unknown": "Unrecorded",
}

# Canonical exam type mapping
EXAM_TYPE_MAP: Dict[str, str] = {
    "midterm": "Midterm",
    "mid-term": "Midterm",
    "mid term": "Midterm",
    "final": "Final",
    "finals": "Final",
    "quiz": "Quiz",
    "practical": "Practical",
    "lab": "Practical",
    "assignment": "Assignment",
}


def standardize_id_series(series: pd.Series) -> pd.Series:
    """Standardize an ID column (strip whitespace, uppercase, treat empty as NaN).

    Args:
        series: Pandas Series containing identifier codes.

    Returns:
        Standardized string Series.
    """
    def clean_id(val: Any) -> Optional[str]:
        if pd.isna(val):
            return None
        s = str(val).strip().upper()
        # Remove trailing .0 from floating point numeric conversions (e.g. 1001.0 -> 1001)
        if s.endswith(".0"):
            s = s[:-2]
        return s if s else None

    return series.apply(clean_id).astype("string")


def standardize_date_series(series: pd.Series, target_format: str = "%Y-%m-%d") -> pd.Series:
    """Standardize date column into ISO YYYY-MM-DD string format safely.

    Invalid date entries are safely coerced to NaT / None without crashing.

    Args:
        series: Pandas Series with date strings, timestamps, or mixed types.
        target_format: Output date format string. Defaults to '%Y-%m-%d'.

    Returns:
        Series of standardized date strings with NaT preserved.
    """
    dt_series = pd.to_datetime(series, errors="coerce", format="mixed")
    # Format non-null dates as string for SQL compatibility
    return dt_series.dt.strftime(target_format)


def standardize_score_series(series: pd.Series) -> pd.Series:
    """Standardize numerical score values (strip text/symbols, cast to float).

    Handles strings like '85.5%', '92/100', ' 74 pts', and coerces invalid entries to NaN.

    Args:
        series: Pandas Series containing score entries.

    Returns:
        Float64 Series with NaNs preserved for missing values.
    """
    def clean_score(val: Any) -> Optional[float]:
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)

        s = str(val).strip()
        # Remove common non-numeric suffixes (/100, %, pts, marks)
        s = re.sub(r"\/100", "", s)
        s = re.sub(r"[%\sptsmarksPTSMARKS]", "", s)

        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    return series.apply(clean_score).astype("float64")


def standardize_percentage_series(series: pd.Series, scale_to_100: bool = True) -> pd.Series:
    """Standardize percentage metrics to a standard 0.0 - 100.0 float scale.

    Intelligently handles decimal ratios (0.85 -> 85.0) and formatted strings ('85%').

    Args:
        series: Pandas Series of percentage values.
        scale_to_100: If True, values in range [0.0, 1.0] are multiplied by 100.

    Returns:
        Float64 Series representing percentage [0.0, 100.0].
    """
    cleaned = standardize_score_series(series)
    if scale_to_100:
        # If all non-null values are between 0 and 1.0, scale up to 100
        valid = cleaned.dropna()
        if not valid.empty and (valid >= 0).all() and (valid <= 1.0).all():
            cleaned = cleaned * 100.0
    return cleaned.round(2)


def standardize_year_series(series: pd.Series) -> pd.Series:
    """Standardize academic study year into an integer in range 1-4.

    Handles strings like 'Year 2', '2nd Year', '2', 2.0. Invalid values become NaN.

    Args:
        series: Pandas Series containing academic year entries.

    Returns:
        Int64 (nullable integer) Series with values 1-4.
    """
    def clean_year(val: Any) -> Optional[int]:
        if pd.isna(val):
            return None
        s = str(val).strip()
        # Extract first digit found
        digits = re.findall(r"[1-4]", s)
        if digits:
            return int(digits[0])
        return None

    return series.apply(clean_year).astype("Int64")


def standardize_attendance_status_series(series: pd.Series) -> pd.Series:
    """Standardize attendance status into canonical categories:

    ['Present', 'Absent', 'Late', 'Excused', 'Unrecorded']

    Args:
        series: Pandas Series containing attendance status entries.

    Returns:
        Standardized string Series.
    """
    def clean_status(val: Any) -> str:
        if pd.isna(val):
            return "Unrecorded"
        key = str(val).strip().lower()
        return ATTENDANCE_STATUS_MAP.get(key, "Unrecorded")

    return series.apply(clean_status).astype("string")


def standardize_categorical_series(series: pd.Series, title_case: bool = True) -> pd.Series:
    """Standardize categorical text fields (strip extra whitespace, title-case).

    Args:
        series: Pandas Series of text categories.
        title_case: Whether to convert text to Title Case. Defaults to True.

    Returns:
        Clean string Series.
    """
    def clean_text(val: Any) -> Optional[str]:
        if pd.isna(val):
            return None
        s = str(val).strip()
        # Collapse multiple internal whitespaces
        s = re.sub(r"\s+", " ", s)
        if not s:
            return None
        return s.title() if title_case else s

    return series.apply(clean_text).astype("string")


# ---------------------------------------------------------------------------
# Entity-Specific Standardization Pipelines
# ---------------------------------------------------------------------------

def standardize_students(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize students dataset columns for SQL and Pandas compliance."""
    out = df.copy()
    if "student_id" in out.columns:
        out["student_id"] = standardize_id_series(out["student_id"])
    if "name" in out.columns:
        out["name"] = standardize_categorical_series(out["name"], title_case=True)
    if "program" in out.columns:
        out["program"] = standardize_categorical_series(out["program"], title_case=True)
    if "year" in out.columns:
        out["year"] = standardize_year_series(out["year"])
    return out


def standardize_courses(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize courses dataset columns."""
    out = df.copy()
    if "course_id" in out.columns:
        out["course_id"] = standardize_id_series(out["course_id"])
    if "course_name" in out.columns:
        out["course_name"] = standardize_categorical_series(out["course_name"], title_case=True)
    if "faculty" in out.columns:
        out["faculty"] = standardize_categorical_series(out["faculty"], title_case=True)
    return out


def standardize_enrollments(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize enrollments dataset columns."""
    out = df.copy()
    if "student_id" in out.columns:
        out["student_id"] = standardize_id_series(out["student_id"])
    if "course_id" in out.columns:
        out["course_id"] = standardize_id_series(out["course_id"])
    return out


def standardize_attendance(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize attendance dataset columns."""
    out = df.copy()
    if "student_id" in out.columns:
        out["student_id"] = standardize_id_series(out["student_id"])
    if "course_id" in out.columns:
        out["course_id"] = standardize_id_series(out["course_id"])
    if "date" in out.columns:
        out["date"] = standardize_date_series(out["date"])
    if "status" in out.columns:
        out["status"] = standardize_attendance_status_series(out["status"])
    return out


def standardize_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize assignments dataset columns."""
    out = df.copy()
    if "assignment_id" in out.columns:
        out["assignment_id"] = standardize_id_series(out["assignment_id"])
    if "course_id" in out.columns:
        out["course_id"] = standardize_id_series(out["course_id"])
    if "title" in out.columns:
        out["title"] = standardize_categorical_series(out["title"], title_case=False)
    if "due_date" in out.columns:
        out["due_date"] = standardize_date_series(out["due_date"])
    return out


def standardize_submissions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize submissions dataset columns."""
    out = df.copy()
    if "assignment_id" in out.columns:
        out["assignment_id"] = standardize_id_series(out["assignment_id"])
    if "student_id" in out.columns:
        out["student_id"] = standardize_id_series(out["student_id"])
    if "submission_date" in out.columns:
        out["submission_date"] = standardize_date_series(out["submission_date"])
    if "score" in out.columns:
        out["score"] = standardize_score_series(out["score"])
    return out


def standardize_exams(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize exams dataset columns."""
    out = df.copy()
    if "exam_id" in out.columns:
        out["exam_id"] = standardize_id_series(out["exam_id"])
    if "student_id" in out.columns:
        out["student_id"] = standardize_id_series(out["student_id"])
    if "course_id" in out.columns:
        out["course_id"] = standardize_id_series(out["course_id"])
    if "exam_type" in out.columns:
        def clean_exam_type(val: Any) -> str:
            if pd.isna(val):
                return "Assessment"
            key = str(val).strip().lower()
            return EXAM_TYPE_MAP.get(key, str(val).strip().title())
        out["exam_type"] = out["exam_type"].apply(clean_exam_type).astype("string")
    if "score" in out.columns:
        out["score"] = standardize_score_series(out["score"])
    return out


def standardize_interventions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize interventions dataset columns."""
    out = df.copy()
    if "student_id" in out.columns:
        out["student_id"] = standardize_id_series(out["student_id"])
    if "date" in out.columns:
        out["date"] = standardize_date_series(out["date"])
    if "type" in out.columns:
        out["type"] = standardize_categorical_series(out["type"], title_case=True)
    if "status" in out.columns:
        out["status"] = standardize_categorical_series(out["status"], title_case=True)
    if "notes" in out.columns:
        out["notes"] = out["notes"].astype(str).str.strip()
    return out


def standardize_academic_dataset(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Standardize all academic entity datasets to compliant Pandas and SQL types.

    Args:
        datasets: Dict mapping entity_name -> DataFrame.

    Returns:
        Dict mapping entity_name -> standardized DataFrame.
    """
    standardized: Dict[str, pd.DataFrame] = {}
    entity_handlers = {
        "students": standardize_students,
        "courses": standardize_courses,
        "enrollments": standardize_enrollments,
        "attendance": standardize_attendance,
        "assignments": standardize_assignments,
        "submissions": standardize_submissions,
        "exams": standardize_exams,
        "interventions": standardize_interventions,
    }

    for name, df in datasets.items():
        clean_name = name.strip().lower()
        handler = entity_handlers.get(clean_name)
        if handler:
            std_df = handler(df)
            logger.info("Standardized entity '%s' (%d rows, %d cols)", clean_name, len(std_df), len(std_df.columns))
        else:
            std_df = df.copy()
        standardized[clean_name] = std_df

    return standardized
