"""Academic string cleaning, categorical normalization, and text standardization.

This module normalizes text fields across academic entities (programs, courses,
faculty, attendance status, assignment status, exam types) while strictly preserving
and protecting alphanumeric business and system IDs from unwanted alteration.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.string_cleaner")

# Acronyms and domain terms that must remain fully uppercase
ACADEMIC_ACRONYMS: Set[str] = {
    "AI",
    "ML",
    "CS",
    "IT",
    "CSE",
    "ECE",
    "EEE",
    "ME",
    "CE",
    "B.TECH",
    "M.TECH",
    "BTECH",
    "MTECH",
    "BSC",
    "MSC",
    "BBA",
    "MBA",
    "PHD",
    "PH.D.",
    "SQL",
    "NLP",
    "DBMS",
    "OS",
    "DSA",
    "IOT",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
}

# Stopwords / minor words that should remain lowercase in titles unless capitalized at start
TITLE_MINOR_WORDS: Set[str] = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    "&",
}

# Canonical program mappings
CANONICAL_PROGRAMS: Dict[str, str] = {
    "btech cse": "B.Tech Computer Science & Engineering",
    "b.tech cse": "B.Tech Computer Science & Engineering",
    "btech cs": "B.Tech Computer Science & Engineering",
    "b.tech cs": "B.Tech Computer Science & Engineering",
    "computer science": "B.Tech Computer Science & Engineering",
    "computer science and engineering": "B.Tech Computer Science & Engineering",
    "computer science & engineering": "B.Tech Computer Science & Engineering",
    "cs": "B.Tech Computer Science & Engineering",
    "cse": "B.Tech Computer Science & Engineering",
    "btech it": "B.Tech Information Technology",
    "b.tech it": "B.Tech Information Technology",
    "information technology": "B.Tech Information Technology",
    "it": "B.Tech Information Technology",
    "btech ece": "B.Tech Electronics & Communication Engineering",
    "b.tech ece": "B.Tech Electronics & Communication Engineering",
    "ece": "B.Tech Electronics & Communication Engineering",
    "bba": "Bachelor of Business Administration",
    "mba": "Master of Business Administration",
    "bsc data science": "B.Sc Data Science",
    "b.sc data science": "B.Sc Data Science",
    "data science": "B.Sc Data Science",
    "ds": "B.Sc Data Science",
}

# Canonical faculty / department mappings
CANONICAL_FACULTY: Dict[str, str] = {
    "eng": "Engineering",
    "engg": "Engineering",
    "engineering": "Engineering",
    "faculty of engineering": "Engineering",
    "school of engineering": "Engineering",
    "department of engineering": "Engineering",
    "cs": "Computer Science",
    "computer science": "Computer Science",
    "school of computer science": "Computer Science",
    "sci": "Sciences",
    "science": "Sciences",
    "sciences": "Sciences",
    "faculty of science": "Sciences",
    "school of sciences": "Sciences",
    "business": "Business Administration",
    "business administration": "Business Administration",
    "school of business": "Business Administration",
    "management": "Management",
    "dept of management": "Management",
    "humanities": "Humanities",
    "arts": "Arts & Humanities",
}

# Canonical attendance status
CANONICAL_ATTENDANCE_STATUS: Dict[str, str] = {
    "p": "Present",
    "present": "Present",
    "1": "Present",
    "1.0": "Present",
    "y": "Present",
    "yes": "Present",
    "here": "Present",
    "attended": "Present",
    "a": "Absent",
    "absent": "Absent",
    "0": "Absent",
    "0.0": "Absent",
    "n": "Absent",
    "no": "Absent",
    "missed": "Absent",
    "l": "Late",
    "late": "Late",
    "tardy": "Late",
    "e": "Excused",
    "excused": "Excused",
    "ex": "Excused",
    "medical": "Excused",
    "authorized": "Excused",
    "unrecorded": "Unrecorded",
    "unknown": "Unrecorded",
    "na": "Unrecorded",
    "n/a": "Unrecorded",
}

# Canonical assignment status
CANONICAL_ASSIGNMENT_STATUS: Dict[str, str] = {
    "submitted": "Submitted",
    "turned in": "Submitted",
    "turnedin": "Submitted",
    "turned_in": "Submitted",
    "done": "Submitted",
    "completed": "Submitted",
    "yes": "Submitted",
    "1": "Submitted",
    "late": "Late",
    "submitted late": "Late",
    "late submission": "Late",
    "turned in late": "Late",
    "pending": "Pending",
    "in progress": "Pending",
    "in-progress": "Pending",
    "in_progress": "Pending",
    "draft": "Pending",
    "not submitted": "Not Submitted",
    "unsubmitted": "Not Submitted",
    "missing": "Not Submitted",
    "no": "Not Submitted",
    "0": "Not Submitted",
    "excused": "Excused",
    "exempt": "Excused",
    "waived": "Excused",
    "graded": "Graded",
    "evaluated": "Graded",
    "scored": "Graded",
}

# Canonical exam types
CANONICAL_EXAM_TYPES: Dict[str, str] = {
    "midterm": "Midterm",
    "mid-term": "Midterm",
    "mid term": "Midterm",
    "midterm 1": "Midterm 1",
    "midterm 2": "Midterm 2",
    "mid term 1": "Midterm 1",
    "mid term 2": "Midterm 2",
    "midterm i": "Midterm I",
    "midterm ii": "Midterm II",
    "mid term i": "Midterm I",
    "mid term ii": "Midterm II",
    "mid-term 1": "Midterm 1",
    "mid-term 2": "Midterm 2",
    "mid-term i": "Midterm I",
    "mid-term ii": "Midterm II",
    "final": "Final",
    "finals": "Final",
    "final exam": "Final",
    "end term": "Final",
    "endterm": "Final",
    "quiz": "Quiz",
    "quiz 1": "Quiz 1",
    "quiz 2": "Quiz 2",
    "practical": "Practical",
    "lab": "Practical",
    "lab exam": "Practical",
    "lab test": "Practical",
    "assignment": "Assignment",
    "project": "Project",
}


@dataclass
class StringFieldCleaningResult:
    """Summary of cleaning changes applied to a single text column."""

    column_name: str
    total_records: int
    unique_before: int
    unique_after: int
    values_modified: int
    sample_mappings: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class StringCleaningAudit:
    """Audit report for string cleaning across an entire DataFrame or batch."""

    entity_name: str
    field_results: Dict[str, StringFieldCleaningResult] = field(default_factory=dict)
    protected_id_columns: List[str] = field(default_factory=list)

    @property
    def total_modifications(self) -> int:
        """Total cells modified during string cleaning."""
        return sum(res.values_modified for res in self.field_results.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit to dictionary."""
        return {
            "entity_name": self.entity_name,
            "total_modifications": self.total_modifications,
            "protected_id_columns": self.protected_id_columns,
            "field_results": {col: res.to_dict() for col, res in self.field_results.items()},
        }

    def to_markdown(self) -> str:
        """Generate structured markdown summary for the string cleaning audit."""
        lines = [
            f"### String Cleaning Audit: `{self.entity_name}`",
            f"- **Protected ID Columns**: `{', '.join(self.protected_id_columns) if self.protected_id_columns else 'None'}`",
            f"- **Total Text Adjustments**: {self.total_modifications:,}",
            "",
            "| Column | Unique Before | Unique After | Modified Cells | Sample Transformations |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for col, res in self.field_results.items():
            samples = ", ".join(f"'{k}' -> '{v}'" for k, v in list(res.sample_mappings.items())[:3])
            if not samples:
                samples = "None (Already clean)"
            lines.append(
                f"| `{col}` | {res.unique_before:,} | {res.unique_after:,} | {res.values_modified:,} | {samples} |"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Text Normalization Functions
# ---------------------------------------------------------------------------


def clean_whitespace(val: Any) -> Optional[str]:
    """Strip leading/trailing whitespace and collapse internal spaces, tabs, and newlines.

    Args:
        val: Any input value.

    Returns:
        Cleaned single-line string with uniform single spaces, or None if empty/null.
    """
    if pd.isna(val):
        return None
    s = str(val)
    # Replace non-breaking spaces (\u00a0), carriage returns, and tabs with space
    s = re.sub(r"[\r\n\t\u00a0\u200b]+", " ", s)
    # Collapse multiple consecutive spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def clean_id(val: Any) -> Optional[str]:
    """Clean an academic ID without altering its casing or internal structure incorrectly.

    Preserves exact uppercase code formats (e.g. 'CS101', 'STU_202', 'EXAM-01'),
    strips whitespace, removes quotes, and safely eliminates trailing '.0'
    floating-point conversion artifacts. NEVER title-cases or corrupts IDs.

    Args:
        val: Input ID value.

    Returns:
        Standardized uppercase string representation, or None if null/empty.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Remove enclosing single or double quotes
    s = s.strip("'\"")
    # Collapse any internal whitespace (though IDs should ideally not have whitespace)
    s = re.sub(r"\s+", "", s)
    # Remove trailing .0 from floating point string representations
    if s.endswith(".0"):
        s = s[:-2]
    # Standardize to uppercase
    s = s.upper()
    return s if s else None


def academic_title_case(val: Any, preserve_acronyms: bool = True) -> Optional[str]:
    """Intelligently convert text to Title Case while respecting academic acronyms and minor words.

    Examples:
        'intro to cs' -> 'Intro to CS'
        'data structures and algorithms' -> 'Data Structures and Algorithms'
        'calculus ii' -> 'Calculus II'
        'ADVANCED AI AND ML' -> 'Advanced AI and ML'

    Args:
        val: Input text string.
        preserve_acronyms: Whether to preserve terms in ACADEMIC_ACRONYMS in all-caps.

    Returns:
        Properly title-cased academic string, or None if null/empty.
    """
    cleaned = clean_whitespace(val)
    if not cleaned:
        return None

    # Replace underscores with spaces for readability in narrative titles
    cleaned = re.sub(r"_+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split into words while preserving punctuation attached to words
    words = cleaned.split(" ")
    title_words: List[str] = []

    for i, raw_word in enumerate(words):
        # Separate leading and trailing punctuation (e.g., '(AI)' or 'Programming:')
        m = re.match(r"^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$", raw_word)
        if not m:
            title_words.append(raw_word)
            continue

        prefix, core, suffix = m.groups()
        core_upper = core.upper()
        core_lower = core.lower()

        # 1. Check if word is an academic acronym or Roman numeral
        if preserve_acronyms and core_upper in ACADEMIC_ACRONYMS:
            cased_core = core_upper
        # 2. Check if word is a minor word (and not the first word)
        elif i > 0 and core_lower in TITLE_MINOR_WORDS:
            cased_core = core_lower
        # 3. Standard title case for general words
        else:
            cased_core = core.capitalize()

        title_words.append(f"{prefix}{cased_core}{suffix}")

    return " ".join(title_words)


# ---------------------------------------------------------------------------
# Specialized Categorical Field Cleaners
# ---------------------------------------------------------------------------


def clean_program_name(val: Any, custom_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Normalize academic program and degree names.

    Handles inconsistent abbreviations (e.g. 'btech cse', 'CS', 'B.Tech CS'),
    spacing, casing, and maps to canonical standard degrees.

    Args:
        val: Program name input.
        custom_map: Optional additional alias mapping.

    Returns:
        Canonical program name string, or None if null.
    """
    base = clean_whitespace(val)
    if not base:
        return None

    lookup_key = base.lower()
    # Normalize punctuation for lookup
    lookup_key = re.sub(r"[\-_]+", " ", lookup_key)
    lookup_key = re.sub(r"\s+", " ", lookup_key).strip()

    mapping = {**CANONICAL_PROGRAMS, **(custom_map or {})}
    if lookup_key in mapping:
        return mapping[lookup_key]

    # Try lookup with periods removed (e.g., 'b.tech' -> 'btech')
    depunct_key = re.sub(r"\.", "", lookup_key)
    if depunct_key in mapping:
        return mapping[depunct_key]

    # If no explicit alias mapping matches, apply academic title casing
    return academic_title_case(base, preserve_acronyms=True)


def clean_course_name(val: Any) -> Optional[str]:
    """Normalize course titles, formatting Roman numerals, acronyms, and subtitles.

    Examples:
        'intro_to_programming' -> 'Intro to Programming'
        'cs101: database systems' -> 'CS101: Database Systems'
        'physics ii' -> 'Physics II'

    Args:
        val: Course title input.

    Returns:
        Formatted course name, or None if null.
    """
    base = clean_whitespace(val)
    if not base:
        return None

    # Handle colon separated course codes (e.g. 'CS101: data structures')
    if ":" in base:
        parts = base.split(":", 1)
        code_part = clean_id(parts[0]) or parts[0].strip()
        title_part = academic_title_case(parts[1]) or ""
        return f"{code_part}: {title_part}"

    return academic_title_case(base, preserve_acronyms=True)


def clean_faculty_name(val: Any, custom_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Normalize faculty or academic department names.

    Examples:
        'engg' -> 'Engineering'
        'faculty of science' -> 'Sciences'
        'dr. john doe' -> 'Dr. John Doe'

    Args:
        val: Faculty or department input.
        custom_map: Optional custom alias mapping.

    Returns:
        Canonical faculty name, or None if null.
    """
    base = clean_whitespace(val)
    if not base:
        return None

    lookup_key = base.lower()
    lookup_key = re.sub(r"[\-_]+", " ", lookup_key)
    lookup_key = re.sub(r"\s+", " ", lookup_key).strip()

    mapping = {**CANONICAL_FACULTY, **(custom_map or {})}
    if lookup_key in mapping:
        return mapping[lookup_key]

    # Standardize common instructor honorifics ('dr.' -> 'Dr.', 'prof.' -> 'Prof.')
    s = academic_title_case(base, preserve_acronyms=True)
    if s:
        s = re.sub(r"\bDr\.\s*", "Dr. ", s)
        s = re.sub(r"\bProf\.\s*", "Prof. ", s)
        s = s.strip()
    return s


def clean_attendance_status(val: Any) -> str:
    """Normalize attendance status into canonical categories:

    ['Present', 'Absent', 'Late', 'Excused', 'Unrecorded']

    Args:
        val: Raw attendance code or status string.

    Returns:
        Canonical attendance status string.
    """
    if pd.isna(val):
        return "Unrecorded"
    s = clean_whitespace(val)
    if not s:
        return "Unrecorded"

    key = s.lower()
    return CANONICAL_ATTENDANCE_STATUS.get(key, "Unrecorded")


def clean_assignment_status(val: Any) -> str:
    """Normalize assignment and submission status into canonical categories:

    ['Submitted', 'Late', 'Pending', 'Not Submitted', 'Excused', 'Graded']

    Args:
        val: Raw assignment status string.

    Returns:
        Canonical assignment status string.
    """
    if pd.isna(val):
        return "Not Submitted"
    s = clean_whitespace(val)
    if not s:
        return "Not Submitted"

    key = s.lower()
    key = re.sub(r"[\-_]+", " ", key).strip()
    return CANONICAL_ASSIGNMENT_STATUS.get(key, academic_title_case(s) or "Not Submitted")


def clean_exam_type(val: Any) -> str:
    """Normalize exam type into canonical categories:

    ['Midterm', 'Midterm 1', 'Midterm 2', 'Final', 'Quiz', 'Practical', 'Assignment', 'Project']

    Args:
        val: Raw exam type string.

    Returns:
        Canonical exam type string.
    """
    if pd.isna(val):
        return "Midterm"
    s = clean_whitespace(val)
    if not s:
        return "Midterm"

    raw_lower = s.lower().strip()
    if raw_lower in CANONICAL_EXAM_TYPES:
        return CANONICAL_EXAM_TYPES[raw_lower]

    key = re.sub(r"[\-_]+", " ", raw_lower).strip()
    if key in CANONICAL_EXAM_TYPES:
        return CANONICAL_EXAM_TYPES[key]

    collapsed = re.sub(r"\bmid\s+term\b", "midterm", key)
    if collapsed in CANONICAL_EXAM_TYPES:
        return CANONICAL_EXAM_TYPES[collapsed]

    return academic_title_case(s, preserve_acronyms=True) or "Midterm"


# ---------------------------------------------------------------------------
# Series-Level Cleaning Helpers
# ---------------------------------------------------------------------------


def clean_text_series(
    series: pd.Series,
    title_case: bool = True,
    preserve_acronyms: bool = True,
) -> pd.Series:
    """Clean a pandas Series of text values."""
    if title_case:
        cleaned = series.apply(lambda x: academic_title_case(x, preserve_acronyms=preserve_acronyms))
    else:
        cleaned = series.apply(clean_whitespace)
    return cleaned.astype("string")


def clean_id_series(series: pd.Series) -> pd.Series:
    """Clean a pandas Series of ID values, safeguarding uppercase and alphanumeric codes."""
    return series.apply(clean_id).astype("string")


# ---------------------------------------------------------------------------
# Entity-Level Cleaning Pipelines
# ---------------------------------------------------------------------------


def clean_students_strings(
    df: pd.DataFrame,
    protected_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StringCleaningAudit]:
    """Clean text columns in students DataFrame while protecting ID columns."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    protected = protected_ids or ["student_id"]
    audit = StringCleaningAudit(entity_name="students", protected_id_columns=protected)

    # 1. Clean ID safely without altering
    if "student_id" in out.columns:
        out["student_id"] = clean_id_series(out["student_id"])

    # 2. Clean student names
    if "name" in out.columns:
        before = out["name"].copy()
        out["name"] = out["name"].apply(lambda x: academic_title_case(x, preserve_acronyms=False)).astype("string")
        audit.field_results["name"] = _create_field_result("name", before, out["name"])

    # 3. Clean program names
    if "program" in out.columns:
        before = out["program"].copy()
        out["program"] = out["program"].apply(clean_program_name).astype("string")
        audit.field_results["program"] = _create_field_result("program", before, out["program"])

    return out, audit


def clean_courses_strings(
    df: pd.DataFrame,
    protected_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StringCleaningAudit]:
    """Clean text columns in courses DataFrame while protecting course_id."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    protected = protected_ids or ["course_id"]
    audit = StringCleaningAudit(entity_name="courses", protected_id_columns=protected)

    if "course_id" in out.columns:
        out["course_id"] = clean_id_series(out["course_id"])

    if "course_name" in out.columns:
        before = out["course_name"].copy()
        out["course_name"] = out["course_name"].apply(clean_course_name).astype("string")
        audit.field_results["course_name"] = _create_field_result("course_name", before, out["course_name"])

    if "faculty" in out.columns:
        before = out["faculty"].copy()
        out["faculty"] = out["faculty"].apply(clean_faculty_name).astype("string")
        audit.field_results["faculty"] = _create_field_result("faculty", before, out["faculty"])

    return out, audit


def clean_attendance_strings(
    df: pd.DataFrame,
    protected_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StringCleaningAudit]:
    """Clean attendance status while safeguarding student_id and course_id."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    protected = protected_ids or ["student_id", "course_id"]
    audit = StringCleaningAudit(entity_name="attendance", protected_id_columns=protected)

    if "student_id" in out.columns:
        out["student_id"] = clean_id_series(out["student_id"])
    if "course_id" in out.columns:
        out["course_id"] = clean_id_series(out["course_id"])

    if "status" in out.columns:
        before = out["status"].copy()
        out["status"] = out["status"].apply(clean_attendance_status).astype("string")
        audit.field_results["status"] = _create_field_result("status", before, out["status"])

    return out, audit


def clean_assignments_strings(
    df: pd.DataFrame,
    protected_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StringCleaningAudit]:
    """Clean assignment titles while safeguarding assignment_id and course_id."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    protected = protected_ids or ["assignment_id", "course_id"]
    audit = StringCleaningAudit(entity_name="assignments", protected_id_columns=protected)

    if "assignment_id" in out.columns:
        out["assignment_id"] = clean_id_series(out["assignment_id"])
    if "course_id" in out.columns:
        out["course_id"] = clean_id_series(out["course_id"])

    if "title" in out.columns:
        before = out["title"].copy()
        out["title"] = out["title"].apply(lambda x: academic_title_case(x, preserve_acronyms=True)).astype("string")
        audit.field_results["title"] = _create_field_result("title", before, out["title"])

    return out, audit


def clean_submissions_strings(
    df: pd.DataFrame,
    protected_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StringCleaningAudit]:
    """Clean submission status while safeguarding assignment_id and student_id."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    protected = protected_ids or ["assignment_id", "student_id"]
    audit = StringCleaningAudit(entity_name="submissions", protected_id_columns=protected)

    if "assignment_id" in out.columns:
        out["assignment_id"] = clean_id_series(out["assignment_id"])
    if "student_id" in out.columns:
        out["student_id"] = clean_id_series(out["student_id"])

    # Check for status column
    status_col = None
    for candidate in ["status", "submission_status", "state"]:
        if candidate in out.columns:
            status_col = candidate
            break

    if status_col is not None:
        before = out[status_col].copy()
        out[status_col] = out[status_col].apply(clean_assignment_status).astype("string")
        audit.field_results[status_col] = _create_field_result(status_col, before, out[status_col])

    return out, audit


def clean_exams_strings(
    df: pd.DataFrame,
    protected_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, StringCleaningAudit]:
    """Clean exam types while safeguarding exam_id, student_id, and course_id."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    out = df.copy()
    protected = protected_ids or ["exam_id", "student_id", "course_id"]
    audit = StringCleaningAudit(entity_name="exams", protected_id_columns=protected)

    if "exam_id" in out.columns:
        out["exam_id"] = clean_id_series(out["exam_id"])
    if "student_id" in out.columns:
        out["student_id"] = clean_id_series(out["student_id"])
    if "course_id" in out.columns:
        out["course_id"] = clean_id_series(out["course_id"])

    if "exam_type" in out.columns:
        before = out["exam_type"].copy()
        out["exam_type"] = out["exam_type"].apply(clean_exam_type).astype("string")
        audit.field_results["exam_type"] = _create_field_result("exam_type", before, out["exam_type"])

    return out, audit


def clean_academic_strings(
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, StringCleaningAudit]]:
    """Clean text fields across an entire batch of academic datasets.

    Args:
        datasets: Dictionary mapping entity names to DataFrames.

    Returns:
        Tuple of:
          - Cleaned DataFrames dictionary.
          - Dictionary of StringCleaningAudit reports.
    """
    clean_fns = {
        "students": clean_students_strings,
        "courses": clean_courses_strings,
        "attendance": clean_attendance_strings,
        "assignments": clean_assignments_strings,
        "submissions": clean_submissions_strings,
        "exams": clean_exams_strings,
    }

    cleaned_data: Dict[str, pd.DataFrame] = {}
    audits: Dict[str, StringCleaningAudit] = {}

    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            continue
        clean_fn = clean_fns.get(name.lower())
        if clean_fn:
            cleaned_df, audit = clean_fn(df)
            cleaned_data[name] = cleaned_df
            audits[name] = audit
        else:
            cleaned_data[name] = df.copy()

    return cleaned_data, audits


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _create_field_result(col_name: str, before: pd.Series, after: pd.Series) -> StringFieldCleaningResult:
    """Compute summary statistics for changes made to a text column."""
    total_records = len(before)
    unique_before = int(before.dropna().nunique())
    unique_after = int(after.dropna().nunique())

    # Count modified non-null values
    diff_mask = (before.astype(str) != after.astype(str)) & before.notna()
    values_modified = int(diff_mask.sum())

    sample_mappings: Dict[str, str] = {}
    if values_modified > 0:
        modified_before = before[diff_mask]
        modified_after = after[diff_mask]
        for b_val, a_val in zip(modified_before.head(5), modified_after.head(5)):
            if pd.notna(b_val) and pd.notna(a_val):
                sample_mappings[str(b_val)] = str(a_val)

    return StringFieldCleaningResult(
        column_name=col_name,
        total_records=total_records,
        unique_before=unique_before,
        unique_after=unique_after,
        values_modified=values_modified,
        sample_mappings=sample_mappings,
    )
