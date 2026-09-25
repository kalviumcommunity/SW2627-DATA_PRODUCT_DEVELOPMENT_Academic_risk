"""Academic Engagement Risk Dashboard - Core Source Package."""

from src.exceptions import (
    DataProcessingError,
    DataLoadError,
    DataValidationError,
    DataTransformationError,
    DataSaveError,
)
from src.logger import get_logger
from src.data_io import (
    load_csv,
    save_csv,
    save_to_sqlite,
    load_from_sqlite,
)
from src.data_inspector import (
    inspect_dataframe,
    generate_summary_report,
    validate_required_columns,
    check_not_empty,
)
from src.data_transformer import (
    standardize_column_names,
    strip_whitespace,
    convert_to_datetime,
    cast_column_types,
    drop_exact_duplicates,
)

__all__ = [
    "DataProcessingError",
    "DataLoadError",
    "DataValidationError",
    "DataTransformationError",
    "DataSaveError",
    "get_logger",
    "load_csv",
    "save_csv",
    "save_to_sqlite",
    "load_from_sqlite",
    "inspect_dataframe",
    "generate_summary_report",
    "validate_required_columns",
    "check_not_empty",
    "standardize_column_names",
    "strip_whitespace",
    "convert_to_datetime",
    "cast_column_types",
    "drop_exact_duplicates",
]
