"""Academic dataset intake and validation layer for core entities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.data_io import load_csv
from src.data_transformer import standardize_column_names
from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_intake")

# Supported file extensions for dataset intake
SUPPORTED_FORMATS: Set[str] = {".csv"}

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
    file_path: Union[str, Path],
    entity_name: str,
    supported_formats: Optional[Set[str]] = None,
) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """Validate a single academic dataset file for structural conformity.

    Performs:
    1. File existence and type check
    2. Supported extension format check
    3. Non-empty file check (>0 bytes)
    4. CSV parsing and row/column count check
    5. Required columns check against the entity schema (including alias mapping)

    Args:
        file_path: Path to dataset file.
        entity_name: Target entity name (students, courses, attendance, etc.).
        supported_formats: Allowed file extensions. Defaults to {'.csv'}.

    Returns:
        Tuple of (Loaded DataFrame or None, ValidationResult).
    """
    path = Path(file_path)
    formats = supported_formats or SUPPORTED_FORMATS
    clean_entity = entity_name.strip().lower()

    result = ValidationResult(
        entity_name=clean_entity,
        is_valid=False,
        file_path=str(path),
    )

    # 1. Verify entity schema is defined
    if clean_entity not in ENTITY_SCHEMAS:
        result.errors.append(
            f"Unknown academic entity '{clean_entity}'. Expected one of: {list(ENTITY_SCHEMAS.keys())}"
        )
        logger.error(result.errors[-1])
        return None, result

    # 2. Verify file existence
    if not path.exists():
        result.errors.append(f"File not found: '{path}'")
        logger.error(result.errors[-1])
        return None, result

    if not path.is_file():
        result.errors.append(f"Path is not a regular file: '{path}'")
        logger.error(result.errors[-1])
        return None, result

    # 3. Verify supported format
    if path.suffix.lower() not in formats:
        result.errors.append(
            f"Unsupported file format '{path.suffix}'. Supported formats are: {sorted(list(formats))}"
        )
        logger.error(result.errors[-1])
        return None, result

    # 4. Verify file is not empty (0 bytes)
    if path.stat().st_size == 0:
        result.errors.append(f"File is completely empty (0 bytes): '{path}'")
        logger.error(result.errors[-1])
        return None, result

    # 5. Parse dataset file
    try:
        df = load_csv(path)
    except Exception as exc:
        result.errors.append(f"Failed to parse CSV file: {exc}")
        logger.error(result.errors[-1])
        return None, result

    # 6. Verify row and column structure
    result.row_count = len(df)
    result.column_count = len(df.columns)

    if result.row_count == 0:
        result.errors.append(f"Dataset has no data rows (empty dataset): '{path}'")
        logger.error(result.errors[-1])
        return None, result

    if result.column_count == 0:
        result.errors.append(f"Dataset has no columns: '{path}'")
        logger.error(result.errors[-1])
        return None, result

    # 7. Normalize columns and check aliases
    df_std = standardize_column_names(df)
    schema = ENTITY_SCHEMAS[clean_entity]
    required_cols = schema["required_columns"]
    aliases = schema.get("aliases", {})

    # Apply aliases if original required column is absent but alias is present
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

    # Validation passed
    result.is_valid = True
    logger.info("Successfully validated entity '%s' (%d rows, %d cols)", clean_entity, result.row_count, result.column_count)
    return df_std, result


def load_and_validate_entity(
    file_path: Union[str, Path],
    entity_name: str,
) -> Tuple[pd.DataFrame, ValidationResult]:
    """Validate and load a single academic entity dataset, raising if invalid.

    Args:
        file_path: Path to dataset file.
        entity_name: Entity name (students, courses, etc.).

    Returns:
        Tuple of (Loaded DataFrame, ValidationResult).

    Raises:
        DataValidationError: If validation fails.
    """
    df, result = validate_dataset_file(file_path=file_path, entity_name=entity_name)
    if not result.is_valid:
        raise DataValidationError(f"Intake validation failed for '{entity_name}': {'; '.join(result.errors)}")
    return df, result


def load_and_validate_academic_dataset(
    data_source: Union[str, Path, Dict[str, Union[str, Path]]],
    strict: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationResult]]:
    """Ingest and validate all 7 academic entities.

    Accepts either:
    - A directory path containing CSV files (e.g. data/raw/ containing students.csv, courses.csv, etc.)
    - A dictionary mapping entity names to file paths.

    Args:
        data_source: Directory path or dict of entity -> filepath.
        strict: If True, raises DataValidationError if ANY entity is missing or invalid.

    Returns:
        Tuple of (Dict[entity_name, DataFrame], Dict[entity_name, ValidationResult]).

    Raises:
        DataValidationError: If strict=True and any entity fails validation.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    results: Dict[str, ValidationResult] = {}
    required_entities = list(ENTITY_SCHEMAS.keys())

    # Build entity -> filepath map
    entity_paths: Dict[str, Path] = {}
    if isinstance(data_source, (str, Path)):
        source_dir = Path(data_source)
        for entity in required_entities:
            # Look for exact entity.csv or entitys.csv
            possible_path = source_dir / f"{entity}.csv"
            entity_paths[entity] = possible_path
    elif isinstance(data_source, dict):
        entity_paths = {k.strip().lower(): Path(v) for k, v in data_source.items()}
    else:
        raise DataValidationError(f"Expected directory path or dictionary, got: {type(data_source).__name__}")

    # Validate each entity
    for entity in required_entities:
        path = entity_paths.get(entity)
        if not path or not path.exists():
            res = ValidationResult(
                entity_name=entity,
                is_valid=False,
                file_path=str(path) if path else None,
                errors=[f"Dataset file missing for entity '{entity}': expected '{path}'"],
            )
            results[entity] = res
            continue

        df, res = validate_dataset_file(file_path=path, entity_name=entity)
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
