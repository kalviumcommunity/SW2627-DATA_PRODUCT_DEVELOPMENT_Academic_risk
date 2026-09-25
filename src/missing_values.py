"""Academic missing value detection, domain-aware handling, and before/after reporting."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.missing_values")


@dataclass
class MissingValueReport:
    """Detailed before-and-after audit report for missing value handling."""

    entity_name: str
    total_rows: int
    before_missing_counts: Dict[str, int]
    after_missing_counts: Dict[str, int]
    strategies_applied: List[str] = field(default_factory=list)
    preserved_indicators: List[str] = field(default_factory=list)

    @property
    def total_before_missing(self) -> int:
        """Total missing cells before processing."""
        return sum(self.before_missing_counts.values())

    @property
    def total_after_missing(self) -> int:
        """Total missing cells after processing."""
        return sum(self.after_missing_counts.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate structured markdown summary of before/after quality."""
        lines = [
            f"### Missing Value Quality Report: `{self.entity_name}`",
            f"- **Total Records**: {self.total_rows:,}",
            f"- **Total Missing Cells Before**: {self.total_before_missing:,}",
            f"- **Total Missing Cells After**: {self.total_after_missing:,}",
            "",
            "| Column | Missing Before | Missing After | Change |",
            "| :--- | :--- | :--- | :--- |",
        ]

        all_cols = sorted(set(list(self.before_missing_counts.keys()) + list(self.after_missing_counts.keys())))
        for col in all_cols:
            before = self.before_missing_counts.get(col, 0)
            after = self.after_missing_counts.get(col, 0)
            diff = after - before
            change_str = f"{diff:+d}" if diff != 0 else "0"
            lines.append(f"| `{col}` | {before} | {after} | {change_str} |")

        lines.append("")
        lines.append("**Domain Strategies Applied:**")
        for strategy in self.strategies_applied:
            lines.append(f"- {strategy}")

        if self.preserved_indicators:
            lines.append("")
            lines.append("**Preserved Missingness Indicators:**")
            for ind in self.preserved_indicators:
                lines.append(f"- Added indicator column: `{ind}`")

        return "\n".join(lines)


def detect_missing_values(df: pd.DataFrame, entity_name: str = "dataset") -> Dict[str, Any]:
    """Detect and quantify missing values across all columns of a dataset.

    Args:
        df: Pandas DataFrame to inspect.
        entity_name: Label for the entity.

    Returns:
        Dictionary containing missing counts, percentages, and affected rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    total_rows = len(df)
    missing_counts = df.isna().sum().to_dict()
    missing_percentages = {
        col: round((cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        for col, cnt in missing_counts.items()
    }
    rows_with_any_missing = int(df.isna().any(axis=1).sum())

    return {
        "entity_name": entity_name,
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "rows_with_any_missing": rows_with_any_missing,
        "pct_rows_with_missing": round((rows_with_any_missing / total_rows) * 100.0, 2) if total_rows > 0 else 0.0,
    }


def handle_attendance_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, MissingValueReport]:
    """Handle missing attendance values according to academic domain rules.

    Rules:
    - Never impute missing attendance status as 0 or Absent!
    - Explicitly set missing status to 'Unrecorded'.
    - Add boolean indicator 'attendance_recorded' (True if valid status, False if unrecorded).

    Args:
        df: Attendance DataFrame.

    Returns:
        Tuple of (Processed DataFrame, MissingValueReport).
    """
    out = df.copy()
    before_missing = out.isna().sum().to_dict()
    strategies = []
    indicators = []

    if "status" in out.columns:
        missing_status_count = int(out["status"].isna().sum())
        # Add indicator preserving whether attendance was actually recorded
        out["attendance_recorded"] = ~out["status"].isna()
        indicators.append("attendance_recorded")

        if missing_status_count > 0:
            out["status"] = out["status"].fillna("Unrecorded")
            strategies.append(
                f"Preserved {missing_status_count} missing attendance records as 'Unrecorded' (not converted to 0/Absent)."
            )
        else:
            strategies.append("All attendance records have recorded status.")

    after_missing = out.isna().sum().to_dict()
    report = MissingValueReport(
        entity_name="attendance",
        total_rows=len(out),
        before_missing_counts=before_missing,
        after_missing_counts=after_missing,
        strategies_applied=strategies,
        preserved_indicators=indicators,
    )
    return out, report


def handle_exams_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, MissingValueReport]:
    """Handle missing exam scores according to academic domain rules.

    Rules:
    - Never impute missing exam scores as 0!
    - Preserve NaN scores for unrecorded/deferred exams so they don't corrupt GPAs.
    - Add boolean indicator 'exam_score_recorded' (True if score present, False if unrecorded).
    - Add 'exam_status' classifying record as 'Graded' vs 'Pending / Absent / Deferred'.

    Args:
        df: Exams DataFrame.

    Returns:
        Tuple of (Processed DataFrame, MissingValueReport).
    """
    out = df.copy()
    before_missing = out.isna().sum().to_dict()
    strategies = []
    indicators = []

    if "score" in out.columns:
        missing_scores = int(out["score"].isna().sum())
        out["exam_score_recorded"] = ~out["score"].isna()
        indicators.append("exam_score_recorded")

        out["exam_status"] = np.where(out["score"].isna(), "Pending / Absent / Deferred", "Graded")
        indicators.append("exam_status")

        if missing_scores > 0:
            strategies.append(
                f"Preserved {missing_scores} missing exam scores as NaN to prevent falsifying performance to 0. Flagged as 'Pending / Absent / Deferred'."
            )
        else:
            strategies.append("All exam records contain recorded scores.")

    after_missing = out.isna().sum().to_dict()
    report = MissingValueReport(
        entity_name="exams",
        total_rows=len(out),
        before_missing_counts=before_missing,
        after_missing_counts=after_missing,
        strategies_applied=strategies,
        preserved_indicators=indicators,
    )
    return out, report


def handle_submissions_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, MissingValueReport]:
    """Handle missing assignment submission values explicitly.

    Rules:
    - Differentiate between unsubmitted vs ungraded submissions.
    - If submission_date is missing: classify as 'Not Submitted'.
    - If submission_date is present but score is missing: classify as 'Ungraded' (preserve score as NaN).
    - If both present: classify as 'Submitted & Graded'.
    - Add boolean indicator 'is_missing_submission'.

    Args:
        df: Submissions DataFrame.

    Returns:
        Tuple of (Processed DataFrame, MissingValueReport).
    """
    out = df.copy()
    before_missing = out.isna().sum().to_dict()
    strategies = []
    indicators = []

    has_sub_date = "submission_date" in out.columns
    has_score = "score" in out.columns

    if has_sub_date:
        missing_sub_dates = int(out["submission_date"].isna().sum())
        out["is_missing_submission"] = out["submission_date"].isna()
        indicators.append("is_missing_submission")

        if has_score:
            # Classification conditions
            conditions = [
                out["submission_date"].isna(),
                (~out["submission_date"].isna()) & (out["score"].isna()),
                (~out["submission_date"].isna()) & (~out["score"].isna()),
            ]
            choices = ["Not Submitted", "Submitted - Ungraded", "Graded"]
            out["submission_status"] = np.select(conditions, choices, default="Unknown")
            indicators.append("submission_status")

            missing_scores = int(out["score"].isna().sum())
            strategies.append(
                f"Identified {missing_sub_dates} missing submissions and {missing_scores} unrecorded scores without zero-imputation."
            )
    elif has_score:
        out["is_score_missing"] = out["score"].isna()
        indicators.append("is_score_missing")

    after_missing = out.isna().sum().to_dict()
    report = MissingValueReport(
        entity_name="submissions",
        total_rows=len(out),
        before_missing_counts=before_missing,
        after_missing_counts=after_missing,
        strategies_applied=strategies,
        preserved_indicators=indicators,
    )
    return out, report


def handle_demographics_missing_values(
    df: pd.DataFrame,
    entity_name: str,
) -> Tuple[pd.DataFrame, MissingValueReport]:
    """Handle missing demographic/categorical values for students or courses.

    Args:
        df: Students or Courses DataFrame.
        entity_name: 'students' or 'courses'.

    Returns:
        Tuple of (Processed DataFrame, MissingValueReport).
    """
    out = df.copy()
    before_missing = out.isna().sum().to_dict()
    strategies = []
    indicators = []

    clean_entity = entity_name.strip().lower()

    if clean_entity == "students":
        if "program" in out.columns and out["program"].isna().any():
            cnt = int(out["program"].isna().sum())
            out["program_imputed"] = out["program"].isna()
            out["program"] = out["program"].fillna("Undeclared")
            indicators.append("program_imputed")
            strategies.append(f"Filled {cnt} missing program values with 'Undeclared'.")

    elif clean_entity == "courses":
        if "faculty" in out.columns and out["faculty"].isna().any():
            cnt = int(out["faculty"].isna().sum())
            out["faculty_imputed"] = out["faculty"].isna()
            out["faculty"] = out["faculty"].fillna("Unassigned")
            indicators.append("faculty_imputed")
            strategies.append(f"Filled {cnt} missing faculty values with 'Unassigned'.")

    after_missing = out.isna().sum().to_dict()
    report = MissingValueReport(
        entity_name=clean_entity,
        total_rows=len(out),
        before_missing_counts=before_missing,
        after_missing_counts=after_missing,
        strategies_applied=strategies,
        preserved_indicators=indicators,
    )
    return out, report


def handle_academic_missing_values(
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, MissingValueReport]]:
    """Apply domain-aware missing value treatment across all academic datasets.

    Enforces project guidelines:
    - Missing attendance is NOT converted to 0.
    - Missing exam scores are NOT converted to 0.
    - Missing assignment submissions are treated explicitly with indicator flags.
    - Generates before-and-after audit reports for all entities.

    Args:
        datasets: Dict of entity_name -> DataFrame.

    Returns:
        Tuple of (Dict of processed DataFrames, Dict of MissingValueReports).
    """
    processed_datasets: Dict[str, pd.DataFrame] = {}
    reports: Dict[str, MissingValueReport] = {}

    for entity, df in datasets.items():
        clean_entity = entity.strip().lower()

        if clean_entity == "attendance":
            proc_df, rep = handle_attendance_missing_values(df)
        elif clean_entity == "exams":
            proc_df, rep = handle_exams_missing_values(df)
        elif clean_entity == "submissions":
            proc_df, rep = handle_submissions_missing_values(df)
        elif clean_entity in {"students", "courses"}:
            proc_df, rep = handle_demographics_missing_values(df, entity_name=clean_entity)
        else:
            # Pass-through with before/after tracking
            before = df.isna().sum().to_dict()
            proc_df = df.copy()
            rep = MissingValueReport(
                entity_name=clean_entity,
                total_rows=len(df),
                before_missing_counts=before,
                after_missing_counts=before,
                strategies_applied=["No entity-specific missing value transformation required."],
            )

        processed_datasets[clean_entity] = proc_df
        reports[clean_entity] = rep
        logger.info(
            "Processed missing values for '%s' (before: %d missing, after: %d missing)",
            clean_entity,
            rep.total_before_missing,
            rep.total_after_missing,
        )

    return processed_datasets, reports
