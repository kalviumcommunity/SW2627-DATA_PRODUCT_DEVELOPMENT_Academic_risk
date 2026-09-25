"""Data input/output utilities for loading and persisting academic datasets."""

from pathlib import Path
import sqlite3
from typing import Any, Union
import pandas as pd

from src.exceptions import DataLoadError, DataSaveError
from src.logger import get_logger

logger = get_logger("academic_risk.data_io")


def load_csv(file_path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame with error handling and logging.

    Args:
        file_path: Path to the target CSV file.
        **kwargs: Additional keyword arguments passed to pd.read_csv.

    Returns:
        pd.DataFrame containing the loaded data.

    Raises:
        DataLoadError: If the file does not exist, is unreadable, or parsing fails.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"CSV file not found at path: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    if not path.is_file():
        msg = f"Path is not a regular file: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    try:
        # Try default encoding first, fall back to latin1 if UnicodeDecodeError occurs
        try:
            df = pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed for %s. Retrying with latin1 encoding.", path)
            df = pd.read_csv(path, encoding="latin1", **kwargs)

        logger.info("Loaded CSV '%s' successfully (%d rows, %d columns)", path.name, len(df), len(df.columns))
        return df
    except Exception as exc:
        msg = f"Failed to load CSV file '{path}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc


def save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False, **kwargs: Any) -> Path:
    """Save a pandas DataFrame to a CSV file, ensuring directories exist.

    Args:
        df: Pandas DataFrame to persist.
        file_path: Target destination path.
        index: Whether to write row index. Defaults to False.
        **kwargs: Additional keyword arguments passed to df.to_csv.

    Returns:
        Path to the saved CSV file.

    Raises:
        DataSaveError: If saving the DataFrame fails.
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame to save, got: {type(df).__name__}"
        logger.error(msg)
        raise DataSaveError(msg)

    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index, **kwargs)
        logger.info("Saved CSV to '%s' successfully (%d rows, %d columns)", path, len(df), len(df.columns))
        return path
    except Exception as exc:
        msg = f"Failed to save CSV file to '{path}': {exc}"
        logger.error(msg)
        raise DataSaveError(msg) from exc


def save_to_sqlite(
    df: pd.DataFrame,
    table_name: str,
    db_path: Union[str, Path],
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    """Save a DataFrame into an SQLite table.

    Args:
        df: Pandas DataFrame to save.
        table_name: Target SQLite table name.
        db_path: Path to the SQLite database file.
        if_exists: How to behave if table exists ('fail', 'replace', 'append'). Defaults to 'replace'.
        index: Whether to write row index as a column. Defaults to False.

    Raises:
        DataSaveError: If database write operation fails.
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame to save, got: {type(df).__name__}"
        logger.error(msg)
        raise DataSaveError(msg)

    path = Path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=index)
        finally:
            conn.close()
        logger.info("Saved %d rows into SQLite table '%s' at '%s'", len(df), table_name, path)
    except Exception as exc:
        msg = f"Failed to save table '{table_name}' to SQLite DB '{path}': {exc}"
        logger.error(msg)
        raise DataSaveError(msg) from exc


def load_from_sqlite(query: str, db_path: Union[str, Path]) -> pd.DataFrame:
    """Execute a SQL query against an SQLite database and return a DataFrame.

    Args:
        query: SQL SELECT query string.
        db_path: Path to the SQLite database file.

    Returns:
        pd.DataFrame containing the query results.

    Raises:
        DataLoadError: If database does not exist or query execution fails.
    """
    path = Path(db_path)
    if not path.exists():
        msg = f"SQLite database not found at path: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    try:
        conn = sqlite3.connect(path)
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
        logger.info("Executed query on '%s', returned %d rows", path.name, len(df))
        return df
    except Exception as exc:
        msg = f"Failed to execute query on SQLite DB '{path}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc


def load_json(file_path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load a JSON file into a pandas DataFrame with error handling and logging.

    Args:
        file_path: Path to the target JSON file.
        **kwargs: Additional keyword arguments passed to pd.read_json.

    Returns:
        pd.DataFrame containing the loaded data.

    Raises:
        DataLoadError: If the file does not exist, is unreadable, or parsing fails.
    """
    import json

    path = Path(file_path)
    if not path.exists():
        msg = f"JSON file not found at path: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    if not path.is_file():
        msg = f"Path is not a regular file: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    try:
        try:
            df = pd.read_json(path, **kwargs)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                try:
                    df = pd.DataFrame(data)
                except ValueError:
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")

        logger.info("Loaded JSON '%s' successfully (%d rows, %d columns)", path.name, len(df), len(df.columns))
        return df
    except Exception as exc:
        msg = f"Failed to load JSON file '{path}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc


def save_json(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    orient: str = "records",
    indent: int = 2,
    **kwargs: Any,
) -> Path:
    """Save a pandas DataFrame to a formatted JSON file.

    Args:
        df: Pandas DataFrame to persist.
        file_path: Target destination path.
        orient: Format of JSON output ('records', 'split', 'index', etc.). Defaults to 'records'.
        indent: Indentation spaces for pretty printing. Defaults to 2.
        **kwargs: Additional keyword arguments passed to df.to_json.

    Returns:
        Path to the saved JSON file.

    Raises:
        DataSaveError: If saving the DataFrame fails.
    """
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected pandas DataFrame to save, got: {type(df).__name__}"
        logger.error(msg)
        raise DataSaveError(msg)

    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient=orient, indent=indent, **kwargs)
        logger.info("Saved JSON to '%s' successfully (%d rows, %d columns)", path, len(df), len(df.columns))
        return path
    except Exception as exc:
        msg = f"Failed to save JSON file to '{path}': {exc}"
        logger.error(msg)
        raise DataSaveError(msg) from exc
