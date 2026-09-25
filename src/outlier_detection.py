"""Academic Outlier Detection, Statistical Evaluation, and Treatment Engine.

Provides multi-method outlier detection (IQR, Z-score, Modified Z-score/MAD, and Domain Bounds)
tailored for academic metrics (attendance, assignment scores, exam scores, submission delays).
Ensures statistical outliers representing legitimate academic risk signals are not blindly deleted.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.outlier_detection")


class OutlierMethod(str, Enum):
    """Supported statistical outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    DOMAIN = "domain"


class OutlierTreatment(str, Enum):
    """Supported treatments for detected outliers."""

    FLAG_ONLY = "flag_only"  # Preserves rows; adds outlier indicators for risk models
    CAP = "cap"              # Winsorizes/caps values to fence boundaries or domain limits
    QUARANTINE = "quarantine"  # Quarantines impossible/corrupted values into audit table
    KEEP = "keep"            # Keeps original values unchanged without modifying rows


@dataclass
class ColumnOutlierSummary:
    """Statistical summary of outlier detection and treatment for a single numerical column."""

    column_name: str
    method_used: str
    treatment_applied: str
    total_records: int
    non_null_records: int
    outliers_count: int
    outliers_pct: float
    lower_bound: float
    upper_bound: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    median_value: Optional[float] = None
    iqr_value: Optional[float] = None
    low_outliers_count: int = 0
    high_outliers_count: int = 0
    domain_invalid_count: int = 0
    sample_outlier_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


@dataclass
class AcademicOutlierReport:
    """Outlier audit report for a specific academic entity."""

    entity_name: str
    column_summaries: Dict[str, ColumnOutlierSummary] = field(default_factory=dict)
    total_rows: int = 0
    outlier_rows_count: int = 0
    quarantined_records: Optional[pd.DataFrame] = None

    @property
    def total_outliers_found(self) -> int:
        """Total outlier occurrences across all analyzed columns."""
        return sum(s.outliers_count for s in self.column_summaries.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format, excluding raw DataFrame."""
        return {
            "entity_name": self.entity_name,
            "total_rows": self.total_rows,
            "outlier_rows_count": self.outlier_rows_count,
            "total_outliers_found": self.total_outliers_found,
            "quarantined_count": len(self.quarantined_records) if self.quarantined_records is not None else 0,
            "columns": {col: s.to_dict() for col, s in self.column_summaries.items()},
        }

    def to_markdown(self) -> str:
        """Generate markdown summary for the outlier report."""
        lines = [
            f"### Academic Outlier Audit Report: `{self.entity_name}`",
            f"- **Total Records Analyzed**: {self.total_rows:,}",
            f"- **Records with Outliers**: {self.outlier_rows_count:,}",
            f"- **Total Outlier Points Detected**: {self.total_outliers_found:,}",
            "",
            "| Column | Method | Lower Bound | Upper Bound | Outliers | % Outliers | Low / High | Treatment |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for col, s in self.column_summaries.items():
            lines.append(
                f"| `{col}` | {s.method_used.upper()} | {s.lower_bound:.2f} | {s.upper_bound:.2f} | "
                f"{s.outliers_count:,} | {s.outliers_pct:.1f}% | {s.low_outliers_count} / {s.high_outliers_count} | "
                f"`{s.treatment_applied}` |"
            )

        lines.append("")
        if self.quarantined_records is not None and not self.quarantined_records.empty:
            lines.append(f"**Quarantined Invalid Records ({len(self.quarantined_records):,} rows):**")
            cols_to_show = [c for c in self.quarantined_records.columns if c in self.column_summaries]
            if "outlier_quarantine_reason" in self.quarantined_records.columns:
                cols_to_show.append("outlier_quarantine_reason")
            sample_df = self.quarantined_records[cols_to_show].head(5)
            # Pure markdown table formatting
            headers = [str(c) for c in sample_df.columns]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for _, r in sample_df.iterrows():
                vals = [str(v) if pd.notna(v) else "" for v in r]
                lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)


@dataclass
class BatchOutlierReport:
    """Outlier report aggregated across multiple academic datasets."""

    reports: Dict[str, AcademicOutlierReport] = field(default_factory=dict)

    @property
    def total_outliers(self) -> int:
        """Total outliers detected across all entities."""
        return sum(r.total_outliers_found for r in self.reports.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch report to dictionary."""
        return {
            "total_outliers": self.total_outliers,
            "entities": {name: r.to_dict() for name, r in self.reports.items()},
        }

    def to_markdown(self) -> str:
        """Generate markdown summary for the batch outlier report."""
        lines = [
            "## Academic Outlier Detection & Treatment Summary",
            f"- **Datasets Processed**: {len(self.reports)}",
            f"- **Total Outliers Detected**: {self.total_outliers:,}",
            "",
            "| Entity | Total Rows | Rows with Outliers | Total Outliers | Columns Evaluated |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for name, r in self.reports.items():
            cols = ", ".join(f"`{c}`" for c in r.column_summaries.keys()) or "None"
            lines.append(
                f"| `{name}` | {r.total_rows:,} | {r.outlier_rows_count:,} | {r.total_outliers_found:,} | {cols} |"
            )

        lines.append("")
        for name, r in self.reports.items():
            lines.append(r.to_markdown())
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical Outlier Detection Algorithms
# ---------------------------------------------------------------------------


def compute_iqr_bounds(
    series: pd.Series,
    k: float = 1.5,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Compute Tukey's IQR fences for outlier detection.

    Args:
        series: Numerical series.
        k: IQR multiplier (default 1.5 for standard outliers, 3.0 for extreme).
        domain_min: Optional hard minimum constraint.
        domain_max: Optional hard maximum constraint.

    Returns:
        Tuple of (lower_bound, upper_bound, iqr).
    """
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0, 0.0, 0.0

    q1 = float(np.percentile(valid, 25))
    q3 = float(np.percentile(valid, 75))
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    # Apply domain boundary clipping if specified
    if domain_min is not None:
        lower_bound = max(lower_bound, domain_min)
    if domain_max is not None:
        upper_bound = min(upper_bound, domain_max)

    return lower_bound, upper_bound, iqr


def compute_zscore_bounds(
    series: pd.Series,
    threshold: float = 3.0,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Compute parametric Z-score cutoff bounds.

    Returns:
        Tuple of (lower_bound, upper_bound, mean, std).
    """
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0

    if std == 0.0:
        lower_bound = mean
        upper_bound = mean
    else:
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

    if domain_min is not None:
        lower_bound = max(lower_bound, domain_min)
    if domain_max is not None:
        upper_bound = min(upper_bound, domain_max)

    return lower_bound, upper_bound, mean, std


def compute_modified_zscore_bounds(
    series: pd.Series,
    threshold: float = 3.5,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Compute robust Modified Z-score bounds using Median and MAD.

    Returns:
        Tuple of (lower_bound, upper_bound, median, mad).
    """
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0

    median = float(valid.median())
    abs_deviations = (valid - median).abs()
    mad = float(abs_deviations.median())

    if mad == 0.0:
        # If MAD is 0 (more than 50% identical values), fall back to mean absolute deviation
        mad = float(abs_deviations.mean())

    if mad == 0.0:
        lower_bound = median
        upper_bound = median
    else:
        # M_i = 0.6745 * (x - median) / MAD => x = median +- threshold * MAD / 0.6745
        scale = (threshold * mad) / 0.6745
        lower_bound = median - scale
        upper_bound = median + scale

    if domain_min is not None:
        lower_bound = max(lower_bound, domain_min)
    if domain_max is not None:
        upper_bound = min(upper_bound, domain_max)

    return lower_bound, upper_bound, median, mad


# ---------------------------------------------------------------------------
# Column-Level Detection & Treatment
# ---------------------------------------------------------------------------


def detect_column_outliers(
    series: pd.Series,
    method: Union[str, OutlierMethod] = "iqr",
    threshold: Optional[float] = None,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
) -> Tuple[pd.Series, ColumnOutlierSummary]:
    """Detect statistical and domain outliers in a numeric pandas Series.

    Args:
        series: Input numerical Series.
        method: Detection method ('iqr', 'zscore', 'modified_zscore', 'domain').
        threshold: Sensitivity multiplier (e.g. 1.5 for IQR, 3.0 for Z-score).
        domain_min: Hard academic minimum limit (e.g. 0.0 for scores).
        domain_max: Hard academic maximum limit (e.g. 100.0 for scores).

    Returns:
        Tuple of:
          - Boolean Series (True for outlier rows, False otherwise, NaN for null rows).
          - ColumnOutlierSummary with statistical cutoffs.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid = numeric_series.dropna()
    total_records = len(series)
    non_null_records = len(valid)

    method_str = str(method).lower()
    col_name = str(series.name) if series.name is not None else "metric"

    # Default thresholds by method
    if threshold is None:
        if method_str in {"iqr", "outliermethod.iqr"}:
            threshold = 1.5
        elif method_str in {"zscore", "outliermethod.zscore"}:
            threshold = 3.0
        elif method_str in {"modified_zscore", "outliermethod.modified_zscore"}:
            threshold = 3.5
        else:
            threshold = 1.0

    iqr_val = None
    mean_val = float(valid.mean()) if non_null_records > 0 else None
    std_val = float(valid.std(ddof=1)) if non_null_records > 1 else None
    median_val = float(valid.median()) if non_null_records > 0 else None
    min_val = float(valid.min()) if non_null_records > 0 else None
    max_val = float(valid.max()) if non_null_records > 0 else None

    # Compute bounds based on selected method
    if method_str in {"iqr", "outliermethod.iqr"}:
        lower_bound, upper_bound, iqr_val = compute_iqr_bounds(
            numeric_series, k=threshold, domain_min=domain_min, domain_max=domain_max
        )
    elif method_str in {"zscore", "outliermethod.zscore"}:
        lower_bound, upper_bound, mean_val, std_val = compute_zscore_bounds(
            numeric_series, threshold=threshold, domain_min=domain_min, domain_max=domain_max
        )
    elif method_str in {"modified_zscore", "outliermethod.modified_zscore"}:
        lower_bound, upper_bound, median_val, _ = compute_modified_zscore_bounds(
            numeric_series, threshold=threshold, domain_min=domain_min, domain_max=domain_max
        )
    elif method_str in {"domain", "outliermethod.domain"}:
        lower_bound = domain_min if domain_min is not None else -np.inf
        upper_bound = domain_max if domain_max is not None else np.inf
    else:
        raise DataValidationError(f"Unknown outlier detection method: '{method}'")

    # Domain invalid count (strictly outside realistic domain bounds)
    domain_invalid_mask = pd.Series(False, index=series.index)
    if domain_min is not None:
        domain_invalid_mask |= numeric_series < domain_min
    if domain_max is not None:
        domain_invalid_mask |= numeric_series > domain_max

    domain_invalid_count = int(domain_invalid_mask.sum())

    # Detect outliers
    low_outliers_mask = numeric_series < lower_bound
    high_outliers_mask = numeric_series > upper_bound
    outlier_mask = (low_outliers_mask | high_outliers_mask) & numeric_series.notna()

    outliers_count = int(outlier_mask.sum())
    low_outliers_count = int(low_outliers_mask.sum())
    high_outliers_count = int(high_outliers_mask.sum())
    outliers_pct = (outliers_count / non_null_records * 100.0) if non_null_records > 0 else 0.0

    sample_outlier_values = [
        float(x) for x in numeric_series[outlier_mask].dropna().head(5).tolist()
    ]

    summary = ColumnOutlierSummary(
        column_name=col_name,
        method_used=method_str,
        treatment_applied=OutlierTreatment.FLAG_ONLY.value,
        total_records=total_records,
        non_null_records=non_null_records,
        outliers_count=outliers_count,
        outliers_pct=round(outliers_pct, 2),
        lower_bound=round(lower_bound, 4),
        upper_bound=round(upper_bound, 4),
        min_value=round(min_val, 4) if min_val is not None else None,
        max_value=round(max_val, 4) if max_val is not None else None,
        mean_value=round(mean_val, 4) if mean_val is not None else None,
        std_value=round(std_val, 4) if std_val is not None else None,
        median_value=round(median_val, 4) if median_val is not None else None,
        iqr_value=round(iqr_val, 4) if iqr_val is not None else None,
        low_outliers_count=low_outliers_count,
        high_outliers_count=high_outliers_count,
        domain_invalid_count=domain_invalid_count,
        sample_outlier_values=sample_outlier_values,
    )

    return outlier_mask, summary


# ---------------------------------------------------------------------------
# Entity-Level Outlier Handlers
# ---------------------------------------------------------------------------


def detect_attendance_outliers(
    df: pd.DataFrame,
    col: Optional[str] = None,
    method: str = "iqr",
    treatment: str = "flag_only",
) -> Tuple[pd.DataFrame, AcademicOutlierReport]:
    """Detect statistical and domain outliers in attendance metrics.

    Evaluates attendance rates (domain bounds: 0.0 to 100.0%).
    Low attendance outliers are marked as critical academic risk factors.
    """
    return _process_entity_outliers(
        df=df,
        entity_name="attendance",
        candidate_cols=[col] if col else ["attendance_pct", "attendance_rate", "rate", "percentage"],
        domain_min=0.0,
        domain_max=100.0,
        method=method,
        treatment=treatment,
    )


def detect_assignment_score_outliers(
    df: pd.DataFrame,
    col: Optional[str] = None,
    method: str = "iqr",
    treatment: str = "flag_only",
) -> Tuple[pd.DataFrame, AcademicOutlierReport]:
    """Detect statistical and domain outliers in assignment scores.

    Evaluates assignment scores (domain bounds: 0.0 to 100.0%).
    """
    return _process_entity_outliers(
        df=df,
        entity_name="submissions",
        candidate_cols=[col] if col else ["score", "marks", "grade", "points"],
        domain_min=0.0,
        domain_max=100.0,
        method=method,
        treatment=treatment,
    )


def detect_exam_score_outliers(
    df: pd.DataFrame,
    col: Optional[str] = None,
    method: str = "iqr",
    treatment: str = "flag_only",
) -> Tuple[pd.DataFrame, AcademicOutlierReport]:
    """Detect statistical outliers in exam scores.

    Evaluates exam scores (domain bounds: 0.0 to 100.0%).
    """
    return _process_entity_outliers(
        df=df,
        entity_name="exams",
        candidate_cols=[col] if col else ["score", "marks", "grade", "exam_score"],
        domain_min=0.0,
        domain_max=100.0,
        method=method,
        treatment=treatment,
    )


def detect_submission_delay_outliers(
    df: pd.DataFrame,
    col: Optional[str] = None,
    method: str = "iqr",
    treatment: str = "flag_only",
) -> Tuple[pd.DataFrame, AcademicOutlierReport]:
    """Detect statistical outliers in student submission latency / delays.

    Evaluates days late or hours late. Values below 0 represent early submissions.
    Values exceeding extreme statistical limits (e.g. >30 days) indicate severe delay.
    """
    return _process_entity_outliers(
        df=df,
        entity_name="submission_delays",
        candidate_cols=[col] if col else ["days_late", "hours_late", "submission_delay"],
        domain_min=-100.0,  # Max 100 days early
        domain_max=120.0,   # Max 120 days late (full semester)
        method=method,
        treatment=treatment,
    )


# ---------------------------------------------------------------------------
# Generic DataFrame Outlier Pipeline
# ---------------------------------------------------------------------------


def analyze_and_treat_outliers(
    df: pd.DataFrame,
    columns: List[str],
    entity_name: str = "academic_dataset",
    method: str = "iqr",
    treatment: str = "flag_only",
    domain_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Tuple[pd.DataFrame, AcademicOutlierReport]:
    """Analyze numerical columns for outliers and apply explainable treatment.

    Treatments:
    - 'flag_only': Appends '{col}_is_outlier', '{col}_outlier_type' ('low_risk', 'high_achiever', 'normal'),
                   preserving all records for risk models.
    - 'cap': Winsorizes / caps values to lower and upper bounds, storing raw value in 'raw_{col}'.
    - 'quarantine': Separates rows with domain-invalid values into report.quarantined_records.

    Args:
        df: Input DataFrame.
        columns: List of numerical column names to inspect.
        entity_name: Entity label for the report.
        method: Statistical method ('iqr', 'zscore', 'modified_zscore').
        treatment: Treatment strategy ('flag_only', 'cap', 'quarantine', 'keep').
        domain_bounds: Optional mapping of column name to (min, max) domain limits.

    Returns:
        Tuple of:
          - Processed DataFrame.
          - AcademicOutlierReport with full statistical audit.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    domain_bounds = domain_bounds or {}
    report = AcademicOutlierReport(entity_name=entity_name, total_rows=len(out))

    if len(out) == 0:
        return out, report

    outlier_indices: Set[Any] = set()
    quarantine_indices: Set[Any] = set()

    for col in columns:
        if col not in out.columns:
            logger.debug("Column '%s' not present in DataFrame for entity '%s'", col, entity_name)
            continue

        d_min, d_max = domain_bounds.get(col, (None, None))
        mask, summary = detect_column_outliers(
            out[col], method=method, domain_min=d_min, domain_max=d_max
        )
        summary.treatment_applied = treatment
        report.column_summaries[col] = summary

        if summary.outliers_count > 0:
            outlier_indices.update(out[mask].index.tolist())

        # 1. Flag-only treatment
        if treatment == "flag_only":
            out[f"{col}_is_outlier"] = mask.astype("boolean")
            # Classify outlier type: 'low_risk', 'high_achiever', 'domain_invalid', 'normal'
            numeric_col = pd.to_numeric(out[col], errors="coerce")
            type_series = pd.Series("normal", index=out.index, dtype="string")

            low_mask = mask & (numeric_col < summary.lower_bound)
            high_mask = mask & (numeric_col > summary.upper_bound)

            type_series[low_mask] = "low_risk"
            type_series[high_mask] = "high_achiever"

            # Domain invalid tag
            if d_min is not None or d_max is not None:
                invalid_mask = pd.Series(False, index=out.index)
                if d_min is not None:
                    invalid_mask |= numeric_col < d_min
                if d_max is not None:
                    invalid_mask |= numeric_col > d_max
                type_series[invalid_mask] = "domain_invalid"

            type_series[numeric_col.isna()] = pd.NA
            out[f"{col}_outlier_type"] = type_series

        # 2. Cap / Winsorize treatment
        elif treatment == "cap":
            out[f"raw_{col}"] = out[col]
            out[col] = out[col].clip(lower=summary.lower_bound, upper=summary.upper_bound)
            out[f"{col}_is_capped"] = mask.astype("boolean")

        # 3. Quarantine treatment (quarantine domain invalid entries)
        elif treatment == "quarantine":
            if summary.domain_invalid_count > 0:
                numeric_col = pd.to_numeric(out[col], errors="coerce")
                invalid_mask = pd.Series(False, index=out.index)
                if d_min is not None:
                    invalid_mask |= numeric_col < d_min
                if d_max is not None:
                    invalid_mask |= numeric_col > d_max
                quarantine_indices.update(out[invalid_mask].index.tolist())

    report.outlier_rows_count = len(outlier_indices)

    # If quarantine selected and invalid rows detected, separate them
    if treatment == "quarantine" and quarantine_indices:
        quarantine_df = out.loc[sorted(quarantine_indices)].copy()
        quarantine_df["outlier_quarantine_reason"] = "Value outside valid academic domain bounds"
        report.quarantined_records = quarantine_df
        out = out.drop(index=list(quarantine_indices)).reset_index(drop=True)

    logger.info(
        "Outlier evaluation for '%s': %d total rows, %d rows with outliers across %d evaluated columns.",
        entity_name,
        report.total_rows,
        report.outlier_rows_count,
        len(report.column_summaries),
    )

    return out, report


def detect_academic_outliers(
    datasets: Dict[str, pd.DataFrame],
    method: str = "iqr",
    treatment: str = "flag_only",
) -> Tuple[Dict[str, pd.DataFrame], BatchOutlierReport]:
    """Detect statistical and domain outliers across an entire batch of academic datasets.

    Evaluates:
    - attendance: 'attendance_pct', 'attendance_rate'
    - submissions: 'score', 'days_late', 'hours_late'
    - exams: 'score'
    - assignments: 'max_points'

    Args:
        datasets: Dictionary of DataFrames.
        method: Statistical method ('iqr', 'zscore', 'modified_zscore').
        treatment: Treatment strategy ('flag_only', 'cap', 'quarantine', 'keep').

    Returns:
        Tuple of:
          - Dictionary of treated DataFrames.
          - BatchOutlierReport summarizing outliers across all datasets.
    """
    col_mapping = {
        "attendance": (["attendance_pct", "attendance_rate", "rate"], {"attendance_pct": (0.0, 100.0), "attendance_rate": (0.0, 1.0)}),
        "submissions": (["score", "days_late", "hours_late"], {"score": (0.0, 100.0), "days_late": (-100.0, 120.0), "hours_late": (-2400.0, 2880.0)}),
        "exams": (["score"], {"score": (0.0, 100.0)}),
        "assignments": (["points", "max_points"], {"max_points": (0.0, 500.0)}),
    }

    treated_datasets: Dict[str, pd.DataFrame] = {}
    batch_report = BatchOutlierReport()

    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            treated_datasets[name] = df.copy() if isinstance(df, pd.DataFrame) else df
            continue

        clean_name = name.lower()
        if clean_name in col_mapping:
            candidates, domain_bounds = col_mapping[clean_name]
            # Identify columns actually present in DataFrame
            target_cols = [c for c in candidates if c in df.columns]
            if target_cols:
                treated_df, report = analyze_and_treat_outliers(
                    df=df,
                    columns=target_cols,
                    entity_name=name,
                    method=method,
                    treatment=treatment,
                    domain_bounds=domain_bounds,
                )
                treated_datasets[name] = treated_df
                batch_report.reports[name] = report
                continue

        # If no specific mapping or target columns found, keep as-is
        treated_datasets[name] = df.copy()

    return treated_datasets, batch_report


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _process_entity_outliers(
    df: pd.DataFrame,
    entity_name: str,
    candidate_cols: List[Optional[str]],
    domain_min: Optional[float],
    domain_max: Optional[float],
    method: str,
    treatment: str,
) -> Tuple[pd.DataFrame, AcademicOutlierReport]:
    """Helper to detect outliers for an entity given candidate column names."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    target_col = None
    for c in candidate_cols:
        if c and c in df.columns:
            target_col = c
            break

    if target_col is None:
        raise DataValidationError(
            f"No matching numerical column found for '{entity_name}'. Candidates: {candidate_cols}"
        )

    return analyze_and_treat_outliers(
        df=df,
        columns=[target_col],
        entity_name=entity_name,
        method=method,
        treatment=treatment,
        domain_bounds={target_col: (domain_min, domain_max)},
    )
