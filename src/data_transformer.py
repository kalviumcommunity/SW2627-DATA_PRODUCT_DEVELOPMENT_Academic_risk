"""Reusable data transformation utilities for academic datasets."""

import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.exceptions import DataTransformationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_transformer")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lower_snake_case and strip special characters.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized column names (new DataFrame copy).

    Raises:
        DataTransformationError: If input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataTransformationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()

    def clean_name(col: Any) -> str:
        s = str(col).strip().lower()
        # Replace spaces, hyphens, and slashes with underscores
        s = re.sub(r"[\s\-\/\.]+", "_", s)
        # Remove any remaining characters that are not alphanumeric or underscore
        s = re.sub(r"[^\w]", "", s)
        # Collapse multiple underscores
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    new_columns = [clean_name(col) for col in out.columns]
    logger.debug("Standardized columns: %s -> %s", list(out.columns), new_columns)
    out.columns = new_columns
    return out


def strip_whitespace(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Strip leading and trailing whitespace from string/object columns.

    Args:
        df: Input DataFrame.
        columns: Specific columns to strip. If None, applies to all object/string columns.

    Returns:
        DataFrame with stripped strings (new DataFrame copy).

    Raises:
        DataTransformationError: If input is not a DataFrame or columns not in df.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataTransformationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    target_cols = columns if columns is not None else [
        c for c in out.columns if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object
    ]

    for col in target_cols:
        if col not in out.columns:
            raise DataTransformationError(f"Column '{col}' not found in DataFrame.")
        # Only strip string values, preserve non-strings / NaNs
        out[col] = out[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    logger.debug("Stripped whitespace on columns: %s", target_cols)
    return out


def convert_to_datetime(
    df: pd.DataFrame,
    columns: List[str],
    date_format: Optional[str] = None,
    errors: str = "coerce",
) -> pd.DataFrame:
    """Safely convert specified columns to datetime.

    Args:
        df: Input DataFrame.
        columns: List of column names to convert.
        date_format: Optional explicit strptime format string.
        errors: Error handling mode ('coerce', 'raise'). Defaults to 'coerce'.

    Returns:
        DataFrame with converted datetime columns.

    Raises:
        DataTransformationError: If column missing or conversion fails with errors='raise'.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataTransformationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise DataTransformationError(f"Column '{col}' not found for datetime conversion.")
        try:
            out[col] = pd.to_datetime(out[col], format=date_format, errors=errors)
            logger.debug("Converted column '%s' to datetime", col)
        except Exception as exc:
            msg = f"Failed to convert column '{col}' to datetime: {exc}"
            logger.error(msg)
            raise DataTransformationError(msg) from exc

    return out


def cast_column_types(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    """Safely cast DataFrame columns to target data types.

    Args:
        df: Input DataFrame.
        type_map: Mapping of column name to target type (e.g. {'student_id': 'str', 'score': 'float'}).

    Returns:
        DataFrame with casted column types.

    Raises:
        DataTransformationError: If casting fails or column missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataTransformationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    for col, target_type in type_map.items():
        if col not in out.columns:
            raise DataTransformationError(f"Column '{col}' not found for type casting.")
        try:
            out[col] = out[col].astype(target_type)
            logger.debug("Casted column '%s' to %s", col, target_type)
        except Exception as exc:
            msg = f"Failed to cast column '{col}' to '{target_type}': {exc}"
            logger.error(msg)
            raise DataTransformationError(msg) from exc

    return out


def drop_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop exact duplicate rows from DataFrame and log the count.

    Args:
        df: Input DataFrame.
        subset: Optional list of columns to consider when identifying duplicates.

    Returns:
        DataFrame with duplicates removed.

    Raises:
        DataTransformationError: If input is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataTransformationError(f"Expected DataFrame, got: {type(df).__name__}")

    initial_len = len(df)
    out = df.drop_duplicates(subset=subset).copy()
    dropped_count = initial_len - len(out)

    if dropped_count > 0:
        logger.info("Dropped %d duplicate rows (from %d to %d rows)", dropped_count, initial_len, len(out))
    else:
        logger.debug("No duplicate rows found.")

    return out
