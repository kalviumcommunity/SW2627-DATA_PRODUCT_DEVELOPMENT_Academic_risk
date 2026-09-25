"""Academic Date/Time Transformation Pipeline for trend analysis and risk monitoring.

Derives temporal and academic calendar features (day, week, month, weekday,
academic period, semester stage, submission latency) across core academic entities:
- attendance dates
- assignment due dates
- submission dates
- exam dates
- intervention dates
"""

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.datetime_pipeline")


@dataclass
class AcademicCalendar:
    """Configurable academic calendar defining term boundaries and progress milestones."""

    term_name: str = "Spring 2026"
    term_start: str = "2026-01-15"
    term_end: str = "2026-05-30"
    # Stage thresholds by academic week
    early_term_max_week: int = 4
    mid_term_max_week: int = 9
    late_term_max_week: int = 14

    def __post_init__(self) -> None:
        self.start_dt = pd.to_datetime(self.term_start)
        self.end_dt = pd.to_datetime(self.term_end)
        if self.start_dt >= self.end_dt:
            raise DataValidationError(f"term_start ({self.term_start}) must be before term_end ({self.term_end})")

    def get_academic_week(self, dt: pd.Timestamp) -> Optional[int]:
        """Compute 1-based academic week of the semester."""
        if pd.isna(dt):
            return None
        # Difference in days from term start
        delta_days = (dt.normalize() - self.start_dt.normalize()).days
        if delta_days < 0:
            # Pre-term / orientation
            return 0
        week_num = (delta_days // 7) + 1
        return int(week_num)

    def get_academic_period(self, dt: pd.Timestamp) -> Optional[str]:
        """Determine the academic period stage based on term calendar."""
        if pd.isna(dt):
            return None
        week = self.get_academic_week(dt)
        if week is None:
            return None
        if week <= 0:
            return "Pre-Term"
        elif week <= self.early_term_max_week:
            return "Early Term"
        elif week <= self.mid_term_max_week:
            return "Mid Term"
        elif week <= self.late_term_max_week:
            return "Late Term"
        else:
            return "Finals / End Term"

    def get_academic_term(self, dt: pd.Timestamp) -> Optional[str]:
        """Infer academic semester from calendar month."""
        if pd.isna(dt):
            return None
        month = dt.month
        # Standard collegiate terms:
        # Jan - May: Spring
        # Jun - Jul: Summer
        # Aug - Dec: Fall
        if 1 <= month <= 5:
            term = "Spring"
        elif 6 <= month <= 7:
            term = "Summer"
        else:
            term = "Fall"
        return f"{term} {dt.year}"


# Global default academic calendar
DEFAULT_ACADEMIC_CALENDAR = AcademicCalendar()


def derive_date_features(
    df: pd.DataFrame,
    date_column: str,
    prefix: Optional[str] = None,
    calendar: Optional[AcademicCalendar] = None,
    keep_parsed_column: bool = False,
) -> pd.DataFrame:
    """Derive comprehensive temporal and academic period features from a date column.

    Derives:
    - `{prefix}day`: Day of month (1-31, Int64)
    - `{prefix}month`: Month of year (1-12, Int64)
    - `{prefix}month_name`: Full month name (string)
    - `{prefix}year`: Calendar year (Int64)
    - `{prefix}week`: ISO week of year (1-53, Int64)
    - `{prefix}weekday`: Day of week name (e.g. 'Monday', 'Friday')
    - `{prefix}weekday_num`: Day of week number (0=Monday, 6=Sunday, Int64)
    - `{prefix}is_weekend`: Boolean weekend flag (boolean)
    - `{prefix}academic_week`: 1-based week since term start (Int64)
    - `{prefix}academic_period`: Academic term stage ('Early Term', 'Mid Term', ...)
    - `{prefix}academic_term`: Academic semester (e.g. 'Spring 2026')

    Args:
        df: Input DataFrame.
        date_column: Name of the date column to transform.
        prefix: Column prefix for derived features. If None, uses '{date_column}_'.
        calendar: AcademicCalendar instance. Defaults to DEFAULT_ACADEMIC_CALENDAR.
        keep_parsed_column: Whether to keep the parsed datetime64 column.

    Returns:
        New DataFrame with derived temporal features appended.

    Raises:
        DataValidationError: If input is not a DataFrame or date_column is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")
    if date_column not in df.columns:
        raise DataValidationError(f"Date column '{date_column}' not found in DataFrame columns: {list(df.columns)}")

    out = df.copy()
    cal = calendar or DEFAULT_ACADEMIC_CALENDAR

    if prefix is None:
        col_prefix = f"{date_column}_"
    else:
        col_prefix = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix

    # Edge case: Empty DataFrame
    if len(out) == 0:
        out[f"{col_prefix}day"] = pd.Series(dtype="Int64")
        out[f"{col_prefix}month"] = pd.Series(dtype="Int64")
        out[f"{col_prefix}month_name"] = pd.Series(dtype="string")
        out[f"{col_prefix}year"] = pd.Series(dtype="Int64")
        out[f"{col_prefix}week"] = pd.Series(dtype="Int64")
        out[f"{col_prefix}weekday"] = pd.Series(dtype="string")
        out[f"{col_prefix}weekday_num"] = pd.Series(dtype="Int64")
        out[f"{col_prefix}is_weekend"] = pd.Series(dtype="boolean")
        out[f"{col_prefix}academic_week"] = pd.Series(dtype="Int64")
        out[f"{col_prefix}academic_period"] = pd.Series(dtype="string")
        out[f"{col_prefix}academic_term"] = pd.Series(dtype="string")
        return out

    # Parse date column safely using format="mixed"
    parsed_dt = pd.to_datetime(out[date_column], errors="coerce", format="mixed")

    # 1. Calendar Day, Month, Year
    out[f"{col_prefix}day"] = parsed_dt.dt.day.astype("Int64")
    out[f"{col_prefix}month"] = parsed_dt.dt.month.astype("Int64")
    out[f"{col_prefix}month_name"] = parsed_dt.dt.month_name().astype("string")
    out[f"{col_prefix}year"] = parsed_dt.dt.year.astype("Int64")

    # 2. Week and Weekday features
    out[f"{col_prefix}week"] = parsed_dt.dt.isocalendar().week.astype("Int64")
    out[f"{col_prefix}weekday"] = parsed_dt.dt.day_name().astype("string")
    out[f"{col_prefix}weekday_num"] = parsed_dt.dt.dayofweek.astype("Int64")
    # Weekend is Saturday (5) or Sunday (6)
    out[f"{col_prefix}is_weekend"] = parsed_dt.dt.dayofweek.isin([5, 6]).astype("boolean")
    # Mask is_weekend to pd.NA where date was NaT
    out.loc[parsed_dt.isna(), f"{col_prefix}is_weekend"] = pd.NA

    # 3. Academic Calendar Features
    academic_weeks = parsed_dt.apply(cal.get_academic_week)
    out[f"{col_prefix}academic_week"] = academic_weeks.astype("Int64")

    academic_periods = parsed_dt.apply(cal.get_academic_period)
    out[f"{col_prefix}academic_period"] = academic_periods.astype("string")

    academic_terms = parsed_dt.apply(cal.get_academic_term)
    out[f"{col_prefix}academic_term"] = academic_terms.astype("string")

    if keep_parsed_column:
        out[f"{col_prefix}datetime"] = parsed_dt

    logger.debug(
        "Derived temporal features for column '%s' with prefix '%s' across %d rows.",
        date_column,
        col_prefix,
        len(out),
    )

    return out


# ---------------------------------------------------------------------------
# Dedicated Academic Entity Date Pipelines
# ---------------------------------------------------------------------------


def transform_attendance_dates(
    df: pd.DataFrame,
    date_col: str = "date",
    calendar: Optional[AcademicCalendar] = None,
) -> pd.DataFrame:
    """Transform attendance records with temporal and academic period fields.

    Derives day-of-week, academic week, and period indicators to facilitate
    identifying recurring attendance drop patterns (e.g. Friday absenteeism, mid-term declines).
    """
    if date_col not in df.columns:
        raise DataValidationError(f"Attendance DataFrame missing required date column '{date_col}'")
    return derive_date_features(df, date_column=date_col, prefix="attendance", calendar=calendar)


def transform_assignment_dates(
    df: pd.DataFrame,
    date_col: str = "due_date",
    calendar: Optional[AcademicCalendar] = None,
) -> pd.DataFrame:
    """Transform assignment due dates with week and academic stage fields.

    Helps track assignment workload pacing across the semester.
    """
    if date_col not in df.columns:
        raise DataValidationError(f"Assignments DataFrame missing required due date column '{date_col}'")
    return derive_date_features(df, date_column=date_col, prefix="due", calendar=calendar)


def transform_submission_dates(
    df: pd.DataFrame,
    date_col: str = "submission_date",
    calendar: Optional[AcademicCalendar] = None,
) -> pd.DataFrame:
    """Transform student submission timestamps with weekday, week, and period features."""
    if date_col not in df.columns:
        raise DataValidationError(f"Submissions DataFrame missing required date column '{date_col}'")
    return derive_date_features(df, date_column=date_col, prefix="submission", calendar=calendar)


def transform_exam_dates(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    calendar: Optional[AcademicCalendar] = None,
) -> pd.DataFrame:
    """Transform exam dates with academic week and period features.

    Supports candidate column names: 'exam_date', 'date', 'scheduled_date'.
    """
    col = date_col
    if col is None:
        for candidate in ["exam_date", "date", "scheduled_date"]:
            if candidate in df.columns:
                col = candidate
                break

    if col is None or col not in df.columns:
        raise DataValidationError(
            f"Exam DataFrame missing date column. Available columns: {list(df.columns)}"
        )

    return derive_date_features(df, date_column=col, prefix="exam", calendar=calendar)


def transform_intervention_dates(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    calendar: Optional[AcademicCalendar] = None,
) -> pd.DataFrame:
    """Transform academic advisor follow-up and intervention dates.

    Supports candidate column names: 'intervention_date', 'date', 'followup_date'.
    """
    col = date_col
    if col is None:
        for candidate in ["intervention_date", "date", "followup_date"]:
            if candidate in df.columns:
                col = candidate
                break

    if col is None or col not in df.columns:
        raise DataValidationError(
            f"Intervention DataFrame missing date column. Available columns: {list(df.columns)}"
        )

    return derive_date_features(df, date_column=col, prefix="intervention", calendar=calendar)


def calculate_submission_latency(
    submissions_df: pd.DataFrame,
    assignments_df: Optional[pd.DataFrame] = None,
    sub_date_col: str = "submission_date",
    due_date_col: str = "due_date",
    assignment_id_col: str = "assignment_id",
) -> pd.DataFrame:
    """Calculate submission timeliness and latency relative to assignment due date.

    If assignments_df is provided and due_date_col is not in submissions_df,
    merges due_date from assignments_df.

    Derives:
    - `days_late`: Float difference in days (negative = early, 0 = on-time, positive = late)
    - `hours_late`: Float difference in hours
    - `is_late`: Boolean flag (True if submitted past due date, False if on-time/early)
    - `submission_timeliness`: Categorical status:
      - 'Early (>24h)': Submitted over 24 hours prior to deadline
      - 'On Time': Submitted within 24 hours before deadline
      - 'Grace Period (<=1h late)': Submitted within 1 hour after deadline
      - 'Late (1-24h late)': Submitted between 1 and 24 hours after deadline
      - 'Severely Late (>24h late)': Submitted over 24 hours after deadline
      - 'Unsubmitted': Submission date missing

    Args:
        submissions_df: Submissions DataFrame.
        assignments_df: Optional assignments DataFrame containing due_date.
        sub_date_col: Submission date column name.
        due_date_col: Due date column name.
        assignment_id_col: Foreign key column for joining assignments.

    Returns:
        Submissions DataFrame with derived latency columns.
    """
    if not isinstance(submissions_df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(submissions_df).__name__}")

    out = submissions_df.copy()

    # If due_date_col not in submissions, attempt merge from assignments_df
    if due_date_col not in out.columns:
        if assignments_df is None or not isinstance(assignments_df, pd.DataFrame):
            raise DataValidationError(
                f"Due date column '{due_date_col}' not found in submissions and no assignments_df provided."
            )
        if assignment_id_col not in out.columns or assignment_id_col not in assignments_df.columns:
            raise DataValidationError(f"Cannot join on '{assignment_id_col}' between submissions and assignments.")

        # Merge due_date
        due_lookup = assignments_df[[assignment_id_col, due_date_col]].drop_duplicates(subset=[assignment_id_col])
        out = out.merge(due_lookup, on=assignment_id_col, how="left")

    sub_dt = pd.to_datetime(out[sub_date_col], errors="coerce", format="mixed")
    due_dt = pd.to_datetime(out[due_date_col], errors="coerce", format="mixed")

    # Time delta
    delta = sub_dt - due_dt
    delta_seconds = delta.dt.total_seconds()

    out["days_late"] = (delta_seconds / 86400.0).round(2)
    out["hours_late"] = (delta_seconds / 3600.0).round(2)

    # Boolean is_late flag (with tolerance of 0 seconds)
    out["is_late"] = (delta_seconds > 0).astype("boolean")
    out.loc[sub_dt.isna() | due_dt.isna(), "is_late"] = pd.NA

    # Categorical timeliness classification
    def classify_timeliness(row: pd.Series) -> str:
        s_date = row[sub_date_col]
        if pd.isna(s_date):
            return "Unsubmitted"
        secs = row["_secs"]
        if pd.isna(secs):
            return "Unknown"
        if secs < -86400:
            return "Early (>24h)"
        elif secs <= 0:
            return "On Time"
        elif secs <= 3600:
            return "Grace Period (<=1h late)"
        elif secs <= 86400:
            return "Late (1-24h late)"
        else:
            return "Severely Late (>24h late)"

    out["_secs"] = delta_seconds
    out["submission_timeliness"] = out.apply(classify_timeliness, axis=1).astype("string")
    out = out.drop(columns=["_secs"])

    return out


def transform_academic_datetimes(
    datasets: Dict[str, pd.DataFrame],
    calendar: Optional[AcademicCalendar] = None,
) -> Dict[str, pd.DataFrame]:
    """Apply temporal transformations across all relevant academic datasets.

    Args:
        datasets: Dictionary of DataFrames (attendance, assignments, submissions, exams, interventions).
        calendar: Optional AcademicCalendar instance.

    Returns:
        Dictionary of enriched DataFrames.
    """
    cal = calendar or DEFAULT_ACADEMIC_CALENDAR
    out: Dict[str, pd.DataFrame] = {}

    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            out[name] = df.copy() if isinstance(df, pd.DataFrame) else df
            continue

        clean_name = name.lower()
        try:
            if clean_name == "attendance" and "date" in df.columns:
                out[name] = transform_attendance_dates(df, calendar=cal)
            elif clean_name == "assignments" and "due_date" in df.columns:
                out[name] = transform_assignment_dates(df, calendar=cal)
            elif clean_name == "submissions" and "submission_date" in df.columns:
                # If assignments is also present, calculate submission latency
                transformed_sub = transform_submission_dates(df, calendar=cal)
                if "assignments" in datasets and "due_date" in datasets["assignments"].columns:
                    transformed_sub = calculate_submission_latency(
                        transformed_sub, assignments_df=datasets["assignments"]
                    )
                out[name] = transformed_sub
            elif clean_name == "exams":
                out[name] = transform_exam_dates(df, calendar=cal)
            elif clean_name in {"interventions", "advisor_notes", "advising"}:
                out[name] = transform_intervention_dates(df, calendar=cal)
            else:
                out[name] = df.copy()
        except Exception as exc:
            logger.warning("Could not apply datetime transformation to '%s': %s", name, exc)
            out[name] = df.copy()

    return out
