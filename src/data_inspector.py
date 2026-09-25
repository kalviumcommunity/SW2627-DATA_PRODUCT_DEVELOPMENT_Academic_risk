"""Data inspection and structural diagnostic utilities for academic datasets."""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_inspector")


def inspect_dataframe(df: pd.DataFrame, name: Optional[str] = None) -> Dict[str, Any]:
    """Inspect a pandas DataFrame and extract structural summary metrics.

    Args:
        df: Pandas DataFrame to inspect.
        name: Optional label for the dataset (e.g. 'attendance_raw').

    Returns:
        Dictionary containing row counts, missing values, dtypes, and duplicates.

    Raises:
        DataValidationError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame for inspection, got: {type(df).__name__}"
        logger.error(msg)
        raise DataValidationError(msg)

    total_rows = len(df)
    missing_counts = df.isna().sum().to_dict()
    missing_percentages = {
        col: round((count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        for col, count in missing_counts.items()
    }

    metrics = {
        "dataset_name": name or "unnamed_dataset",
        "row_count": total_rows,
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }

    logger.debug(
        "Inspected dataset '%s': %d rows, %d cols, %d duplicates",
        metrics["dataset_name"],
        total_rows,
        metrics["column_count"],
        metrics["duplicate_rows"],
    )
    return metrics


def generate_summary_report(df: pd.DataFrame, name: Optional[str] = None) -> str:
    """Generate a clean, readable text summary report of a DataFrame.

    Args:
        df: Pandas DataFrame to summarize.
        name: Optional dataset name.

    Returns:
        Formatted multi-line text summary report.
    """
    info = inspect_dataframe(df, name=name)

    lines = [
        f"--- Dataset Summary: {info['dataset_name']} ---",
        f"Rows: {info['row_count']:,} | Columns: {info['column_count']} | Duplicate Rows: {info['duplicate_rows']:,}",
        f"Memory Usage: {info['memory_usage_bytes'] / 1024:.2f} KB",
        "Columns & Quality:",
    ]

    for col in info["columns"]:
        dtype = info["dtypes"][col]
        missing = info["missing_counts"][col]
        pct = info["missing_percentages"][col]
        lines.append(f"  - {col} ({dtype}): {missing} missing ({pct}%)")

    return "\n".join(lines)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Check that all required column names exist in the DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: List of expected column names.

    Raises:
        DataValidationError: If any required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}. Present columns: {list(df.columns)}"
        logger.error(msg)
        raise DataValidationError(msg)


def check_not_empty(df: pd.DataFrame, name: str = "dataset") -> None:
    """Validate that a DataFrame is not empty (contains at least one row).

    Args:
        df: DataFrame to check.
        name: Label for descriptive error messaging.

    Raises:
        DataValidationError: If DataFrame has 0 rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    if df.empty:
        msg = f"Dataset '{name}' is empty (0 rows)."
        logger.error(msg)
        raise DataValidationError(msg)
