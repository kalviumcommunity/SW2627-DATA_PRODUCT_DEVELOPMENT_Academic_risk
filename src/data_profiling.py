"""Academic data profiling and structured data-quality reporting system."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_profiling")


@dataclass
class ColumnProfile:
    """Detailed statistical and structural profile of a single column."""

    column_name: str
    data_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_numerical: bool
    numerical_stats: Optional[Dict[str, float]] = None
    top_values: Optional[Dict[str, int]] = None


@dataclass
class DatasetProfile:
    """Summary profile of a tabular dataset."""

    dataset_name: str
    row_count: int
    column_count: int
    duplicate_rows: int
    duplicate_percentage: float
    memory_usage_bytes: int
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    quality_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset profile to serializable dictionary."""
        return asdict(self)


@dataclass
class AcademicQualityReport:
    """Specialized data quality report for academic entities."""

    entity_name: str
    row_count: int
    column_count: int
    quality_score: float  # Percentage (0 - 100) based on completeness and validity
    anomalies_detected: List[str] = field(default_factory=list)
    academic_metrics: Dict[str, Any] = field(default_factory=dict)
    base_profile: Optional[DatasetProfile] = None

    def to_markdown(self) -> str:
        """Format the quality report as a clean markdown document."""
        lines = [
            f"## Academic Data Quality Report: `{self.entity_name}`",
            f"- **Total Records**: {self.row_count:,}",
            f"- **Total Columns**: {self.column_count}",
            f"- **Data Quality Score**: {self.quality_score:.1f}%",
            "",
            "### Academic Domain Findings",
        ]

        if self.academic_metrics:
            for k, v in self.academic_metrics.items():
                formatted_key = k.replace("_", " ").title()
                lines.append(f"- **{formatted_key}**: {v}")
        else:
            lines.append("- No entity-specific metrics computed.")

        lines.append("")
        lines.append("### Quality Alerts & Anomalies")
        if self.anomalies_detected:
            for alert in self.anomalies_detected:
                lines.append(f"- ⚠️ {alert}")
        else:
            lines.append("- ✅ No structural or domain anomalies detected.")

        return "\n".join(lines)


def profile_column(series: pd.Series) -> ColumnProfile:
    """Compute complete statistical profile for a single pandas Series.

    Args:
        series: Pandas Series to profile.

    Returns:
        ColumnProfile dataclass with statistics.
    """
    total_len = len(series)
    missing = int(series.isna().sum())
    missing_pct = round((missing / total_len) * 100.0, 2) if total_len > 0 else 0.0

    # Non-null values for calculation
    valid_series = series.dropna()
    unique_cnt = int(valid_series.nunique())
    unique_pct = round((unique_cnt / total_len) * 100.0, 2) if total_len > 0 else 0.0

    is_num = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    num_stats = None

    if is_num and not valid_series.empty:
        # Calculate clean numerical stats
        std_val = float(valid_series.std()) if len(valid_series) > 1 else 0.0
        num_stats = {
            "min": round(float(valid_series.min()), 2),
            "max": round(float(valid_series.max()), 2),
            "mean": round(float(valid_series.mean()), 2),
            "median": round(float(valid_series.median()), 2),
            "std": round(std_val if not math.isnan(std_val) else 0.0, 2),
            "q25": round(float(valid_series.quantile(0.25)), 2),
            "q75": round(float(valid_series.quantile(0.75)), 2),
        }

    # Top value counts for categorical or low-cardinality columns
    top_vals = None
    if not is_num or unique_cnt <= 10:
        val_counts = valid_series.value_counts().head(5)
        top_vals = {str(k): int(v) for k, v in val_counts.items()}

    return ColumnProfile(
        column_name=str(series.name),
        data_type=str(series.dtype),
        missing_count=missing,
        missing_percentage=missing_pct,
        unique_count=unique_cnt,
        unique_percentage=unique_pct,
        is_numerical=is_num,
        numerical_stats=num_stats,
        top_values=top_vals,
    )


def profile_dataset(df: pd.DataFrame, dataset_name: str = "dataset") -> DatasetProfile:
    """Generate a comprehensive statistical profile of a dataset.

    Args:
        df: Pandas DataFrame to profile.
        dataset_name: Name of the dataset for labeling.

    Returns:
        DatasetProfile dataclass.

    Raises:
        DataValidationError: If input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    row_count = len(df)
    col_count = len(df.columns)
    dup_rows = int(df.duplicated().sum()) if row_count > 0 else 0
    dup_pct = round((dup_rows / row_count) * 100.0, 2) if row_count > 0 else 0.0
    mem_usage = int(df.memory_usage(deep=True).sum())

    col_profiles = {col: profile_column(df[col]) for col in df.columns}

    # Quality alerts
    alerts: List[str] = []
    if dup_rows > 0:
        alerts.append(f"Contains {dup_rows} duplicate rows ({dup_pct}%).")

    for col, cp in col_profiles.items():
        if cp.missing_percentage > 20.0:
            alerts.append(f"Column '{col}' has high missingness ({cp.missing_percentage}%).")
        if cp.unique_count == 1 and row_count > 1:
            alerts.append(f"Column '{col}' has constant value across all rows.")

    logger.debug("Profiled dataset '%s' (%d rows, %d cols, %d alerts)", dataset_name, row_count, col_count, len(alerts))
    return DatasetProfile(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=col_count,
        duplicate_rows=dup_rows,
        duplicate_percentage=dup_pct,
        memory_usage_bytes=mem_usage,
        column_profiles=col_profiles,
        quality_alerts=alerts,
    )


def profile_academic_dataset(
    df: pd.DataFrame,
    entity_name: str,
) -> AcademicQualityReport:
    """Generate specialized domain quality assessment for an academic entity.

    Performs entity-specific sanity checks:
    - Score ranges [0 - 100] for exams and submissions
    - Attendance status categories and presence rates
    - Student ID uniqueness in student directory
    - Primary key integrity

    Args:
        df: Pandas DataFrame of the academic entity.
        entity_name: Entity label ('students', 'attendance', 'exams', etc.).

    Returns:
        AcademicQualityReport dataclass.
    """
    base_prof = profile_dataset(df, dataset_name=entity_name)
    clean_name = entity_name.strip().lower()
    anomalies: List[str] = list(base_prof.quality_alerts)
    metrics: Dict[str, Any] = {}

    quality_score = 100.0

    # 1. Students Entity Profiling
    if clean_name == "students":
        if "student_id" in df.columns:
            unique_ids = df["student_id"].nunique()
            dup_ids = len(df) - unique_ids
            metrics["distinct_students"] = unique_ids
            if dup_ids > 0:
                anomalies.append(f"Primary key violation: {dup_ids} duplicate student_id records found.")
                quality_score -= 20.0
        if "program" in df.columns:
            metrics["program_breakdown"] = df["program"].value_counts().to_dict()
        if "year" in df.columns:
            metrics["year_distribution"] = df["year"].value_counts().to_dict()

    # 2. Courses Entity Profiling
    elif clean_name == "courses":
        if "course_id" in df.columns:
            metrics["distinct_courses"] = df["course_id"].nunique()
        if "faculty" in df.columns:
            metrics["distinct_faculty"] = df["faculty"].nunique()

    # 3. Attendance Entity Profiling
    elif clean_name == "attendance":
        if "status" in df.columns:
            status_counts = df["status"].astype(str).str.strip().str.title().value_counts().to_dict()
            metrics["status_distribution"] = status_counts

            present_count = status_counts.get("Present", 0)
            total_records = len(df)
            presence_rate = round((present_count / total_records) * 100.0, 2) if total_records > 0 else 0.0
            metrics["overall_presence_rate"] = f"{presence_rate}%"

            # Check for non-standard statuses
            standard_statuses = {"Present", "Absent", "Late", "Excused"}
            unrecognized = [s for s in status_counts if s not in standard_statuses]
            if unrecognized:
                anomalies.append(f"Non-standard attendance status values detected: {unrecognized}")
                quality_score -= 10.0

        if "student_id" in df.columns:
            metrics["students_with_attendance"] = df["student_id"].nunique()

    # 4. Assignments & Submissions Entity Profiling
    elif clean_name in {"assignments", "submissions"}:
        if "score" in df.columns:
            valid_scores = df["score"].dropna()
            if not valid_scores.empty:
                out_of_bounds = valid_scores[(valid_scores < 0) | (valid_scores > 100)]
                metrics["average_score"] = round(float(valid_scores.mean()), 2)
                metrics["median_score"] = round(float(valid_scores.median()), 2)
                metrics["graded_count"] = len(valid_scores)
                metrics["unrecorded_count"] = int(df["score"].isna().sum())

                if not out_of_bounds.empty:
                    anomalies.append(f"{len(out_of_bounds)} score(s) outside standard [0, 100] range.")
                    quality_score -= 25.0

    # 5. Exams Entity Profiling
    elif clean_name == "exams":
        if "exam_type" in df.columns:
            metrics["exam_type_breakdown"] = df["exam_type"].value_counts().to_dict()
        if "score" in df.columns:
            valid_scores = df["score"].dropna()
            if not valid_scores.empty:
                out_of_bounds = valid_scores[(valid_scores < 0) | (valid_scores > 100)]
                metrics["exam_score_mean"] = round(float(valid_scores.mean()), 2)
                metrics["exam_score_min"] = round(float(valid_scores.min()), 2)
                metrics["exam_score_max"] = round(float(valid_scores.max()), 2)

                if not out_of_bounds.empty:
                    anomalies.append(f"{len(out_of_bounds)} exam score(s) outside [0, 100] boundary.")
                    quality_score -= 25.0

    # Deduct score for missing rows or missing columns
    if base_prof.row_count == 0:
        quality_score = 0.0
    else:
        quality_score = max(0.0, min(100.0, quality_score - (base_prof.duplicate_percentage * 0.5)))

    logger.info("Generated academic quality report for '%s' (score: %.1f%%)", clean_name, quality_score)
    return AcademicQualityReport(
        entity_name=clean_name,
        row_count=base_prof.row_count,
        column_count=base_prof.column_count,
        quality_score=round(quality_score, 1),
        anomalies_detected=anomalies,
        academic_metrics=metrics,
        base_profile=base_prof,
    )


def profile_academic_database(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, AcademicQualityReport]:
    """Generate quality reports for all academic entity datasets in the database.

    Args:
        datasets: Mapping of entity_name -> DataFrame.

    Returns:
        Dictionary mapping entity_name -> AcademicQualityReport.
    """
    reports: Dict[str, AcademicQualityReport] = {}
    for entity, df in datasets.items():
        reports[entity] = profile_academic_dataset(df, entity_name=entity)
    return reports
