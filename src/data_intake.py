"""Academic dataset intake and validation layer for core entities (CSV & JSON)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.data_io import load_csv, load_json
from src.data_transformer import standardize_column_names
from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_intake")

# Supported file extensions for dataset intake
SUPPORTED_FORMATS: Set[str] = {".csv", ".json"}

# Canonical schemas and allowed column aliases for the 7 core entities
ENTITY_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "students": {
        "required_columns": ["student_id", "name", "program", "year"],
        "aliases": {},
    },
    "courses": {
        "required_columns": ["course_id", "course_name", "faculty"],
        "aliases": {"instructor": "faculty", "teacher": "faculty"},
    },
    "enrollments": {
        "required_columns": ["student_id", "course_id"],
        "aliases": {},
    },
    "attendance": {
        "required_columns": ["student_id", "course_id", "date", "status"],
        "aliases": {"attendance_status": "status", "present": "status"},
    },
    "assignments": {
        "required_columns": ["assignment_id", "course_id", "title", "due_date"],
        "aliases": {"assignment_title": "title", "deadline": "due_date"},
    },
    "submissions": {
        "required_columns": ["assignment_id", "student_id", "submission_date", "score"],
        "aliases": {"submitted_date": "submission_date", "marks": "score"},
    },
    "exams": {
        "required_columns": ["exam_id", "student_id", "course_id", "exam_type", "score"],
        "aliases": {"type": "exam_type", "marks": "score"},
    },
}


@dataclass
class ValidationResult:
    """Result of intake validation for a single dataset entity."""

    entity_name: str
    is_valid: bool
    file_path: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    columns_found: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a single-line summary of the validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        error_msg = f" - Errors: {'; '.join(self.errors)}" if self.errors else ""
        return f"[{status}] {self.entity_name} ({self.row_count} rows, {self.column_count} cols){error_msg}"


def validate_dataset_file(
    file_path: Union[str, Path, pd.DataFrame],
    entity_name: str,
    supported_formats: Optional[Set[str]] = None,
) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """Validate an academic dataset (file or DataFrame) for structural conformity.

    Performs:
    1. Entity schema check
    2. File existence and supported format check (.csv, .json)
    3. Non-empty file check (>0 bytes)
    4. Dataset parsing (CSV or JSON)
    5. Row and column presence checks
    6. Column name normalization and alias resolution against canonical schema

    Args:
        file_path: Path to dataset file (CSV/JSON) or existing pandas DataFrame.
        entity_name: Target entity name (students, courses, attendance, etc.).
        supported_formats: Allowed file extensions. Defaults to {'.csv', '.json'}.

    Returns:
        Tuple of (Loaded DataFrame or None, ValidationResult).
    """
    clean_entity = entity_name.strip().lower()
    formats = supported_formats or SUPPORTED_FORMATS

    result = ValidationResult(
        entity_name=clean_entity,
        is_valid=False,
        file_path=str(file_path) if not isinstance(file_path, pd.DataFrame) else "in_memory_dataframe",
    )

    # 1. Verify entity schema is defined
    if clean_entity not in ENTITY_SCHEMAS:
        result.errors.append(
            f"Unknown academic entity '{clean_entity}'. Expected one of: {list(ENTITY_SCHEMAS.keys())}"
        )
        logger.error(result.errors[-1])
        return None, result

    # 2. Extract DataFrame from file or in-memory object
    if isinstance(file_path, pd.DataFrame):
        df = file_path.copy()
    else:
        path = Path(file_path)
        if not path.exists():
            result.errors.append(f"File not found: '{path}'")
            logger.error(result.errors[-1])
            return None, result

        if not path.is_file():
            result.errors.append(f"Path is not a regular file: '{path}'")
            logger.error(result.errors[-1])
            return None, result

        if path.suffix.lower() not in formats:
            result.errors.append(
                f"Unsupported file format '{path.suffix}'. Supported formats are: {sorted(list(formats))}"
            )
            logger.error(result.errors[-1])
            return None, result

        if path.stat().st_size == 0:
            result.errors.append(f"File is completely empty (0 bytes): '{path}'")
            logger.error(result.errors[-1])
            return None, result

        try:
            if path.suffix.lower() == ".csv":
                df = load_csv(path)
            elif path.suffix.lower() == ".json":
                df = load_json(path)
            else:
                result.errors.append(f"No parser available for format: '{path.suffix}'")
                return None, result
        except Exception as exc:
            result.errors.append(f"Failed to parse dataset file: {exc}")
            logger.error(result.errors[-1])
            return None, result

    # 3. Verify row and column structure
    result.row_count = len(df)
    result.column_count = len(df.columns)

    if result.row_count == 0:
        result.errors.append(f"Dataset has no data rows (empty dataset): '{file_path}'")
        logger.error(result.errors[-1])
        return None, result

    if result.column_count == 0:
        result.errors.append(f"Dataset has no columns: '{file_path}'")
        logger.error(result.errors[-1])
        return None, result

    # 4. Normalize columns and check aliases
    df_std = standardize_column_names(df)
    schema = ENTITY_SCHEMAS[clean_entity]
    required_cols = schema["required_columns"]
    aliases = schema.get("aliases", {})

    rename_map = {}
    for alias, canonical in aliases.items():
        if alias in df_std.columns and canonical not in df_std.columns:
            rename_map[alias] = canonical
            result.warnings.append(f"Mapped column alias '{alias}' to canonical '{canonical}'")

    if rename_map:
        df_std = df_std.rename(columns=rename_map)

    result.columns_found = list(df_std.columns)
    missing = [col for col in required_cols if col not in df_std.columns]
    result.missing_columns = missing

    if missing:
        result.errors.append(
            f"Missing required columns for entity '{clean_entity}': {missing}. Found: {result.columns_found}"
        )
        logger.error(result.errors[-1])
        return None, result

    result.is_valid = True
    logger.info("Successfully validated entity '%s' (%d rows, %d cols)", clean_entity, result.row_count, result.column_count)
    return df_std, result


def load_and_validate_entity(
    source: Optional[Union[str, Path, pd.DataFrame]] = None,
    entity_name: str = "",
    file_path: Optional[Union[str, Path, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, ValidationResult]:
    """Validate and load a single academic entity dataset, raising if invalid.

    Args:
        source: Path to dataset file or DataFrame.
        entity_name: Entity name (students, courses, etc.).
        file_path: Alternative alias for source parameter.

    Returns:
        Tuple of (Loaded DataFrame, ValidationResult).

    Raises:
        DataValidationError: If validation fails.
    """
    target = file_path if file_path is not None else source
    if target is None:
        raise DataValidationError("Either source or file_path must be provided.")

    df, result = validate_dataset_file(file_path=target, entity_name=entity_name)
    if not result.is_valid or df is None:
        raise DataValidationError(f"Intake validation failed for '{entity_name}': {'; '.join(result.errors)}")
    return df, result


# ---------------------------------------------------------------------------
# Dedicated entity loading functions for students, courses, enrollments, etc.
# ---------------------------------------------------------------------------

def load_students(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate student records from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="students")


def load_courses(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate course records from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="courses")


def load_enrollments(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate student course enrollments from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="enrollments")


def load_attendance(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate student attendance records from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="attendance")


def load_assignments(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate course assignments from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="assignments")


def load_submissions(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate assignment submissions from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="submissions")


def load_exams(source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, ValidationResult]:
    """Load and validate student exam scores from CSV, JSON, or DataFrame."""
    return load_and_validate_entity(source=source, entity_name="exams")


# ---------------------------------------------------------------------------
# Batch Dataset Ingestion Function
# ---------------------------------------------------------------------------

def load_and_validate_academic_dataset(
    data_source: Union[str, Path, Dict[str, Union[str, Path, pd.DataFrame]]],
    strict: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationResult]]:
    """Ingest and validate all 7 academic entities from CSV or JSON sources.

    Args:
        data_source: Directory path or dict of entity -> (filepath or DataFrame).
        strict: If True, raises DataValidationError if ANY entity is missing or invalid.

    Returns:
        Tuple of (Dict[entity_name, DataFrame], Dict[entity_name, ValidationResult]).

    Raises:
        DataValidationError: If strict=True and any entity fails validation.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    results: Dict[str, ValidationResult] = {}
    required_entities = list(ENTITY_SCHEMAS.keys())

    # Build entity -> source map
    entity_sources: Dict[str, Union[Path, pd.DataFrame]] = {}
    if isinstance(data_source, (str, Path)):
        source_dir = Path(data_source)
        for entity in required_entities:
            # Check for entity.csv, then entity.json
            csv_path = source_dir / f"{entity}.csv"
            json_path = source_dir / f"{entity}.json"
            if csv_path.exists():
                entity_sources[entity] = csv_path
            elif json_path.exists():
                entity_sources[entity] = json_path
            else:
                entity_sources[entity] = csv_path  # default expected for reporting
    elif isinstance(data_source, dict):
        entity_sources = {
            k.strip().lower(): v if isinstance(v, pd.DataFrame) else Path(v)
            for k, v in data_source.items()
        }
    else:
        raise DataValidationError(f"Expected directory path or dictionary, got: {type(data_source).__name__}")

    # Validate each entity
    for entity in required_entities:
        source_item = entity_sources.get(entity)
        if source_item is None or (isinstance(source_item, Path) and not source_item.exists()):
            res = ValidationResult(
                entity_name=entity,
                is_valid=False,
                file_path=str(source_item) if source_item is not None else None,
                errors=[f"Dataset file missing for entity '{entity}': expected '{source_item}'"],
            )
            results[entity] = res
            continue

        df, res = validate_dataset_file(file_path=source_item, entity_name=entity)
        results[entity] = res
        if df is not None and res.is_valid:
            datasets[entity] = df

    invalid_entities = [k for k, v in results.items() if not v.is_valid]
    if strict and invalid_entities:
        error_details = "\n".join([f"  - {results[e].summary()}" for e in invalid_entities])
        msg = f"Academic dataset intake failed for {len(invalid_entities)} entities:\n{error_details}"
        logger.error(msg)
        raise DataValidationError(msg)

    return datasets, results
