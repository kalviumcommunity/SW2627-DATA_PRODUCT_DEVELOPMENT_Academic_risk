"""Academic SQLite Relational Database Management Module.

Provides database initialization, loading of cleaned datasets,
foreign key relationship integrity verification, and row count auditing across
the 8 canonical academic entities:
1. students
2. courses
3. enrollments
4. attendance
5. assignments
6. submissions
7. exams
8. interventions
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataLoadError, DataSaveError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.database")

# Path to the canonical SQL DDL schema file
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

# The 8 canonical academic tables in alphabetical order
CANONICAL_TABLES: List[str] = [
    "students",
    "courses",
    "enrollments",
    "attendance",
    "assignments",
    "submissions",
    "exams",
    "interventions",
]

# Ingestion order honoring foreign key dependencies (parents before children)
TOPOLOGICAL_INGESTION_ORDER: List[str] = [
    "students",
    "courses",
    "enrollments",
    "assignments",
    "attendance",
    "submissions",
    "exams",
    "interventions",
]


@dataclass
class ForeignKeyViolation:
    """Represents a foreign key constraint or referential integrity violation."""

    table_name: str
    rowid: Optional[int]
    parent_table: str
    foreign_key_id: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return asdict(self)


@dataclass
class DatabaseValidationReport:
    """Audit report validating schema, foreign key integrity, and row counts."""

    is_valid: bool
    table_counts: Dict[str, int]
    expected_counts: Dict[str, int]
    row_counts_match: bool
    row_count_differences: Dict[str, int]
    foreign_key_violations: List[ForeignKeyViolation]
    missing_tables: List[str]
    validation_timestamp: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "table_counts": self.table_counts,
            "expected_counts": self.expected_counts,
            "row_counts_match": self.row_counts_match,
            "row_count_differences": self.row_count_differences,
            "foreign_key_violations": [v.to_dict() for v in self.foreign_key_violations],
            "missing_tables": self.missing_tables,
            "validation_timestamp": self.validation_timestamp,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Database Connection & Lifecycle
# ---------------------------------------------------------------------------


def get_connection(
    db_path: Union[str, Path],
    foreign_keys: bool = True,
) -> sqlite3.Connection:
    """Open an SQLite database connection with row factory and foreign keys enabled.

    Args:
        db_path: Path to SQLite database file or ":memory:".
        foreign_keys: Whether to execute PRAGMA foreign_keys = ON. Defaults to True.

    Returns:
        sqlite3.Connection configured for academic relational operations.

    Raises:
        DataLoadError: If database connection fails.
    """
    path_str = str(db_path)
    try:
        if path_str != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path_str)
        conn.row_factory = sqlite3.Row
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except Exception as exc:
        msg = f"Failed to connect to SQLite database at '{db_path}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc


@contextmanager
def db_transaction(
    db_path: Union[str, Path],
    foreign_keys: bool = True,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing an atomic database transaction.

    Commits on success; rolls back and re-raises on exception.

    Args:
        db_path: Path to database.
        foreign_keys: Whether to enforce foreign keys.

    Yields:
        sqlite3.Connection within active transaction.
    """
    conn = get_connection(db_path, foreign_keys=foreign_keys)
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Transaction rolled back due to error: %s", exc)
        raise
    finally:
        conn.close()


def init_database(
    db_path: Union[str, Path],
    schema_path: Optional[Union[str, Path]] = None,
    drop_existing: bool = False,
) -> sqlite3.Connection:
    """Initialize SQLite database with relational tables and indexes from schema.sql.

    Args:
        db_path: Path to target SQLite database.
        schema_path: Path to custom schema DDL file. Defaults to sql/schema.sql.
        drop_existing: If True, drops canonical tables in reverse dependency order before applying schema.

    Returns:
        Open sqlite3.Connection to the initialized database.

    Raises:
        DataSaveError: If schema execution fails.
        DataLoadError: If schema file cannot be read.
    """
    resolved_schema = Path(schema_path or DEFAULT_SCHEMA_PATH)
    if not resolved_schema.exists():
        msg = f"Schema DDL file not found at: {resolved_schema}"
        logger.error(msg)
        raise DataLoadError(msg)

    try:
        ddl_script = resolved_schema.read_text(encoding="utf-8")
    except Exception as exc:
        msg = f"Failed to read schema DDL file '{resolved_schema}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc

    conn = get_connection(db_path, foreign_keys=True)
    try:
        if drop_existing:
            # Drop tables in reverse topological order to satisfy foreign keys
            reverse_order = list(reversed(TOPOLOGICAL_INGESTION_ORDER))
            cursor = conn.cursor()
            for tbl in reverse_order:
                cursor.execute(f"DROP TABLE IF EXISTS {tbl};")
            conn.commit()
            logger.info("Dropped existing tables in reverse dependency order.")

        conn.executescript(ddl_script)
        conn.commit()
        logger.info("Successfully executed schema DDL from '%s' into '%s'", resolved_schema.name, db_path)
        return conn
    except Exception as exc:
        conn.close()
        msg = f"Failed to initialize database schema from '{resolved_schema}': {exc}"
        logger.error(msg)
        raise DataSaveError(msg) from exc


def verify_schema(
    db_path_or_conn: Union[str, Path, sqlite3.Connection],
) -> Dict[str, List[str]]:
    """Verify that all 8 canonical tables exist in the database and return their columns.

    Args:
        db_path_or_conn: Path to SQLite database or active sqlite3.Connection.

    Returns:
        Dictionary mapping table name to list of column names.

    Raises:
        DataValidationError: If any canonical table is missing.
    """
    should_close = False
    if isinstance(db_path_or_conn, sqlite3.Connection):
        conn = db_path_or_conn
    else:
        conn = get_connection(db_path_or_conn)
        should_close = True

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing = [tbl for tbl in CANONICAL_TABLES if tbl not in existing_tables]
        if missing:
            msg = f"Missing required canonical tables in SQLite database: {missing}"
            logger.error(msg)
            raise DataValidationError(msg)

        schema_info: Dict[str, List[str]] = {}
        for tbl in CANONICAL_TABLES:
            cursor.execute(f"PRAGMA table_info({tbl});")
            columns = [row[1] for row in cursor.fetchall()]
            schema_info[tbl] = columns

        logger.info("Schema verified successfully: all %d canonical tables present.", len(CANONICAL_TABLES))
        return schema_info
    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------


def sanitize_dataframe_for_table(
    df: pd.DataFrame,
    table_name: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Align DataFrame columns with SQLite table schema, handling missing and extra columns.

    Args:
        df: Input pandas DataFrame.
        table_name: Target SQLite table.
        conn: Open SQLite connection.

    Returns:
        Sanitized DataFrame with only columns that exist in the target table.
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    table_cols = [row[1] for row in cursor.fetchall()]

    # Case-insensitive column matching
    lower_to_target = {col.lower().strip(): col for col in table_cols}
    df_aligned = df.copy()

    rename_map = {}
    for c in df_aligned.columns:
        c_clean = str(c).lower().strip()
        if c_clean in lower_to_target:
            rename_map[c] = lower_to_target[c_clean]

    df_aligned = df_aligned.rename(columns=rename_map)

    # Keep only target table columns
    cols_to_keep = [col for col in table_cols if col in df_aligned.columns]
    return df_aligned[cols_to_keep]


def load_table_data(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
    if_exists: str = "append",
) -> int:
    """Load a DataFrame into an existing SQLite table.

    Args:
        conn: Open SQLite connection with transaction.
        table_name: Target SQLite table name.
        df: Pandas DataFrame containing rows to insert.
        if_exists: 'append' or 'replace'. Defaults to 'append'.

    Returns:
        Number of rows inserted.

    Raises:
        DataSaveError: If table insert fails.
    """
    if df is None or df.empty:
        logger.info("DataFrame for table '%s' is empty; 0 rows inserted.", table_name)
        return 0

    sanitized_df = sanitize_dataframe_for_table(df, table_name, conn)

    try:
        sanitized_df.to_sql(
            name=table_name,
            con=conn,
            if_exists=if_exists,
            index=False,
        )
        row_count = len(sanitized_df)
        logger.info("Loaded %d rows into table '%s'", row_count, table_name)
        return row_count
    except Exception as exc:
        msg = f"Failed to load data into table '{table_name}': {exc}"
        logger.error(msg)
        raise DataSaveError(msg) from exc


def load_cleaned_datasets_to_sqlite(
    db_path: Union[str, Path],
    datasets: Dict[str, pd.DataFrame],
    init_schema: bool = True,
    validate: bool = True,
    schema_path: Optional[Union[str, Path]] = None,
    drop_existing: bool = True,
) -> DatabaseValidationReport:
    """Load multiple cleaned academic DataFrames into SQLite honoring topological dependencies.

    Args:
        db_path: Target SQLite database file.
        datasets: Dictionary mapping table name to cleaned DataFrame.
        init_schema: If True, initializes schema before loading. Defaults to True.
        validate: If True, runs foreign key and row count validation after loading. Defaults to True.
        schema_path: Path to custom schema DDL file.
        drop_existing: If True and init_schema is True, drops existing tables. Defaults to True.

    Returns:
        DatabaseValidationReport detailing load and integrity audit.

    Raises:
        DataSaveError: If database write operation fails.
        DataValidationError: If integrity validation fails when validate=True.
    """
    if not isinstance(datasets, dict):
        raise DataValidationError(f"datasets must be a dictionary, got {type(datasets).__name__}")

    # Track expected counts from input datasets
    expected_counts: Dict[str, int] = {}
    for tbl in CANONICAL_TABLES:
        if tbl in datasets and datasets[tbl] is not None:
            expected_counts[tbl] = len(datasets[tbl])
        else:
            expected_counts[tbl] = 0

    # Initialize schema if requested
    if init_schema:
        conn = init_database(db_path, schema_path=schema_path, drop_existing=drop_existing)
        conn.close()

    # Load tables in topological order inside a transaction
    with db_transaction(db_path, foreign_keys=True) as conn:
        for table_name in TOPOLOGICAL_INGESTION_ORDER:
            if table_name in datasets and datasets[table_name] is not None:
                load_table_data(conn, table_name, datasets[table_name], if_exists="append")

    logger.info("All datasets loaded into '%s' in topological order.", db_path)

    # Perform validation
    if validate:
        return validate_academic_database(db_path, expected_counts=expected_counts)

    actual_counts = get_table_row_counts(db_path)
    return DatabaseValidationReport(
        is_valid=True,
        table_counts=actual_counts,
        expected_counts=expected_counts,
        row_counts_match=actual_counts == expected_counts,
        row_count_differences={k: actual_counts.get(k, 0) - expected_counts.get(k, 0) for k in CANONICAL_TABLES},
        foreign_key_violations=[],
        missing_tables=[],
        validation_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary="Datasets loaded without post-load validation check.",
    )


# ---------------------------------------------------------------------------
# Validation Utilities: Relationships and Row Counts
# ---------------------------------------------------------------------------


def validate_foreign_keys(
    db_path_or_conn: Union[str, Path, sqlite3.Connection],
) -> List[ForeignKeyViolation]:
    """Execute SQLite PRAGMA foreign_key_check and return all referential integrity violations.

    Args:
        db_path_or_conn: Path to SQLite database or open connection.

    Returns:
        List of ForeignKeyViolation objects (empty list if database is referentially consistent).
    """
    should_close = False
    if isinstance(db_path_or_conn, sqlite3.Connection):
        conn = db_path_or_conn
    else:
        conn = get_connection(db_path_or_conn, foreign_keys=True)
        should_close = True

    violations: List[ForeignKeyViolation] = []
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_key_check;")
        rows = cursor.fetchall()
        for row in rows:
            # PRAGMA foreign_key_check returns: table, rowid, parent, fkid
            tbl = row[0]
            rowid = row[1]
            parent = row[2]
            fkid = row[3]
            v = ForeignKeyViolation(
                table_name=tbl,
                rowid=rowid,
                parent_table=parent,
                foreign_key_id=fkid,
                description=f"Rowid {rowid} in child table '{tbl}' has orphan foreign key referencing parent '{parent}'.",
            )
            violations.append(v)

        if violations:
            logger.warning("Found %d foreign key violations in database.", len(violations))
        else:
            logger.info("PRAGMA foreign_key_check passed: 0 foreign key violations.")
        return violations
    finally:
        if should_close:
            conn.close()


def get_table_row_counts(
    db_path_or_conn: Union[str, Path, sqlite3.Connection],
) -> Dict[str, int]:
    """Retrieve row counts for all canonical academic tables.

    Args:
        db_path_or_conn: Path to SQLite database or open connection.

    Returns:
        Dictionary mapping table name to integer row count.
    """
    should_close = False
    if isinstance(db_path_or_conn, sqlite3.Connection):
        conn = db_path_or_conn
    else:
        conn = get_connection(db_path_or_conn)
        should_close = True

    counts: Dict[str, int] = {}
    try:
        cursor = conn.cursor()
        for tbl in CANONICAL_TABLES:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
                row = cursor.fetchone()
                counts[tbl] = int(row[0]) if row else 0
            except sqlite3.OperationalError:
                counts[tbl] = 0
        return counts
    finally:
        if should_close:
            conn.close()


def validate_row_counts(
    db_path_or_conn: Union[str, Path, sqlite3.Connection],
    expected_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Validate that actual database table counts match expected row counts.

    Args:
        db_path_or_conn: Path to SQLite database or open connection.
        expected_counts: Mapping of table names to expected integer count.

    Returns:
        Dictionary with comparison results:
        - matches: bool
        - table_counts: Dict[str, int]
        - expected_counts: Dict[str, int]
        - differences: Dict[str, int]
    """
    actual = get_table_row_counts(db_path_or_conn)
    differences: Dict[str, int] = {}
    matches = True

    for tbl, exp_cnt in expected_counts.items():
        act_cnt = actual.get(tbl, 0)
        diff = act_cnt - exp_cnt
        differences[tbl] = diff
        if diff != 0:
            matches = False

    return {
        "matches": matches,
        "table_counts": actual,
        "expected_counts": expected_counts,
        "differences": differences,
    }


def validate_database_relationships(
    db_path_or_conn: Union[str, Path, sqlite3.Connection],
) -> List[ForeignKeyViolation]:
    """Comprehensive validation of relational foreign keys across all academic entity pairs.

    Runs both SQLite PRAGMA foreign_key_check and explicit SQL anti-join checks
    to detect any referential integrity inconsistencies.

    Args:
        db_path_or_conn: Path to SQLite database or open connection.

    Returns:
        List of any detected ForeignKeyViolation objects.
    """
    violations = validate_foreign_keys(db_path_or_conn)

    # Perform explicit relational audits
    should_close = False
    if isinstance(db_path_or_conn, sqlite3.Connection):
        conn = db_path_or_conn
    else:
        conn = get_connection(db_path_or_conn)
        should_close = True

    try:
        cursor = conn.cursor()
        relational_checks = [
            ("enrollments", "student_id", "students", "student_id"),
            ("enrollments", "course_id", "courses", "course_id"),
            ("attendance", "student_id", "students", "student_id"),
            ("attendance", "course_id", "courses", "course_id"),
            ("assignments", "course_id", "courses", "course_id"),
            ("submissions", "assignment_id", "assignments", "assignment_id"),
            ("submissions", "student_id", "students", "student_id"),
            ("exams", "student_id", "students", "student_id"),
            ("exams", "course_id", "courses", "course_id"),
            ("interventions", "student_id", "students", "student_id"),
        ]

        for child_tbl, child_fk, parent_tbl, parent_pk in relational_checks:
            try:
                query = f"""
                    SELECT c.{child_fk}
                    FROM {child_tbl} c
                    LEFT JOIN {parent_tbl} p ON c.{child_fk} = p.{parent_pk}
                    WHERE c.{child_fk} IS NOT NULL AND p.{parent_pk} IS NULL
                    LIMIT 5;
                """
                cursor.execute(query)
                orphans = cursor.fetchall()
                for orphan in orphans:
                    val = orphan[0]
                    v = ForeignKeyViolation(
                        table_name=child_tbl,
                        rowid=None,
                        parent_table=parent_tbl,
                        description=f"Key '{val}' in '{child_tbl}.{child_fk}' has no parent in '{parent_tbl}.{parent_pk}'.",
                    )
                    violations.append(v)
            except sqlite3.OperationalError as exc:
                logger.debug("Skipped explicit check for %s -> %s: %s", child_tbl, parent_tbl, exc)

        return violations
    finally:
        if should_close:
            conn.close()


def validate_academic_database(
    db_path_or_conn: Union[str, Path, sqlite3.Connection],
    expected_counts: Optional[Dict[str, int]] = None,
) -> DatabaseValidationReport:
    """Run comprehensive audit on SQLite database: schema, row counts, and relationships.

    Args:
        db_path_or_conn: Path to SQLite database or open connection.
        expected_counts: Optional expected row counts for tables.

    Returns:
        DatabaseValidationReport with complete validation status.

    Raises:
        DataValidationError: If schema is missing or invalid.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Verify schema
    missing_tables: List[str] = []
    try:
        verify_schema(db_path_or_conn)
    except DataValidationError:
        # Check specifically which tables are missing
        actual_counts = get_table_row_counts(db_path_or_conn)
        missing_tables = [tbl for tbl in CANONICAL_TABLES if tbl not in actual_counts]

    # 2. Get table row counts
    actual_counts = get_table_row_counts(db_path_or_conn)

    # 3. Check row counts match expected
    exp_counts = expected_counts or actual_counts.copy()
    row_check = validate_row_counts(db_path_or_conn, exp_counts)
    counts_match = row_check["matches"]
    differences = row_check["differences"]

    # 4. Check foreign key relationships
    fk_violations = validate_database_relationships(db_path_or_conn)

    is_valid = (
        len(missing_tables) == 0
        and counts_match
        and len(fk_violations) == 0
    )

    summary_parts = []
    if is_valid:
        summary_parts.append(
            f"Database validation PASSED. All {len(CANONICAL_TABLES)} tables intact, "
            f"row counts match expectations, and 0 foreign key violations detected."
        )
    else:
        if missing_tables:
            summary_parts.append(f"Missing tables: {missing_tables}.")
        if not counts_match:
            mismatches = {k: v for k, v in differences.items() if v != 0}
            summary_parts.append(f"Row count mismatches: {mismatches}.")
        if fk_violations:
            summary_parts.append(f"{len(fk_violations)} foreign key violations detected.")

    summary = " ".join(summary_parts)
    logger.info("Database validation result: is_valid=%s", is_valid)

    return DatabaseValidationReport(
        is_valid=is_valid,
        table_counts=actual_counts,
        expected_counts=exp_counts,
        row_counts_match=counts_match,
        row_count_differences=differences,
        foreign_key_violations=fk_violations,
        missing_tables=missing_tables,
        validation_timestamp=timestamp,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Object-Oriented Database Manager
# ---------------------------------------------------------------------------


class AcademicDatabase:
    """High-level relational manager for the SQLite Academic Database."""

    def __init__(
        self,
        db_path: Union[str, Path],
        schema_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize AcademicDatabase instance.

        Args:
            db_path: Path to SQLite database file.
            schema_path: Path to custom schema DDL file.
        """
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else ":memory:"
        self.schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self, drop_existing: bool = False) -> "AcademicDatabase":
        """Initialize schema in database.

        Args:
            drop_existing: If True, drop existing tables first.

        Returns:
            Self instance.
        """
        conn = init_database(self.db_path, schema_path=self.schema_path, drop_existing=drop_existing)
        conn.close()
        return self

    def load_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        validate: bool = True,
    ) -> DatabaseValidationReport:
        """Load cleaned datasets into database.

        Args:
            datasets: Dictionary mapping table name to DataFrame.
            validate: Whether to validate foreign keys and counts.

        Returns:
            DatabaseValidationReport.
        """
        return load_cleaned_datasets_to_sqlite(
            db_path=self.db_path,
            datasets=datasets,
            init_schema=False,
            validate=validate,
        )

    def query(self, sql_query: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
        """Execute a SELECT query and return results as a DataFrame.

        Args:
            sql_query: SQL SELECT query string.
            params: Optional query parameters.

        Returns:
            DataFrame containing query results.
        """
        conn = get_connection(self.db_path)
        try:
            return pd.read_sql_query(sql_query, conn, params=params)
        finally:
            conn.close()

    def get_row_counts(self) -> Dict[str, int]:
        """Get row counts for all canonical tables."""
        return get_table_row_counts(self.db_path)

    def validate(self, expected_counts: Optional[Dict[str, int]] = None) -> DatabaseValidationReport:
        """Validate database relationships and row counts."""
        return validate_academic_database(self.db_path, expected_counts=expected_counts)

    def __enter__(self) -> "AcademicDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._conn:
            self._conn.close()
            self._conn = None
