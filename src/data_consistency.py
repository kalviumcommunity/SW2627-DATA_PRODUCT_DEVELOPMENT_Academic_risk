"""Academic Data Consistency and Relational Integrity Engine.

Provides deep validation rules and reporting across academic datasets:
- Scores bounded between 0 and 100
- Attendance within valid range (0-100% or non-negative counts)
- Approved attendance statuses (Present, Absent, Late, Excused, Unrecorded)
- Logically valid submission and assignment due dates
- Valid student and course ID format and syntax
- Referential integrity for student and course foreign keys
- Valid student enrollment relationships across course activities
"""

from dataclasses import asdict, dataclass, field
import datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.data_consistency")

# Approved canonical attendance statuses
APPROVED_ATTENDANCE_STATUSES: Set[str] = {
    "Present",
    "Absent",
    "Late",
    "Excused",
    "Unrecorded",
}

# Standard regex for academic student and course IDs
STUDENT_ID_REGEX = r"^[A-Z0-9_\-]+$"
COURSE_ID_REGEX = r"^[A-Z0-9_\-]+$"


@dataclass
class ConsistencyIssue:
    """Detailed record of a single data consistency rule violation."""

    rule_name: str
    entity_name: str
    column_name: Optional[str]
    severity: str  # "ERROR" or "WARNING"
    violating_count: int
    total_checked: int
    violation_pct: float
    description: str
    sample_violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return asdict(self)


@dataclass
class EntityConsistencyReport:
    """Consistency audit report for a specific academic entity."""

    entity_name: str
    total_rows: int
    passed_rules: int = 0
    failed_rules: int = 0
    issues: List[ConsistencyIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Entity is valid if no ERROR severity consistency issues exist."""
        return not any(issue.severity == "ERROR" and issue.violating_count > 0 for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "entity_name": self.entity_name,
            "total_rows": self.total_rows,
            "is_valid": self.is_valid,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_markdown(self) -> str:
        """Generate markdown summary for the entity consistency report."""
        status_icon = "Passed" if self.is_valid else "Failed"
        lines = [
            f"### Consistency Report: `{self.entity_name}` ({status_icon})",
            f"- **Records Checked**: {self.total_rows:,}",
            f"- **Rules Evaluated**: {self.passed_rules + self.failed_rules} (Passed: {self.passed_rules}, Failed: {self.failed_rules})",
            "",
            "| Rule | Column | Violations | % Violations | Severity | Details |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        if not self.issues:
            lines.append("| All Rules Passed | - | 0 | 0.0% | INFO | Fully compliant |")
        else:
            for issue in self.issues:
                col = f"`{issue.column_name}`" if issue.column_name else "-"
                lines.append(
                    f"| `{issue.rule_name}` | {col} | {issue.violating_count:,} | "
                    f"{issue.violation_pct:.1f}% | `{issue.severity}` | {issue.description} |"
                )

        return "\n".join(lines)


@dataclass
class AcademicConsistencyReport:
    """Comprehensive consistency and referential integrity report across all datasets."""

    entity_reports: Dict[str, EntityConsistencyReport] = field(default_factory=dict)
    relational_issues: List[ConsistencyIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Entire dataset batch is consistent if all entities and relations pass without ERRORs."""
        entities_ok = all(rep.is_valid for rep in self.entity_reports.values())
        relations_ok = not any(iss.severity == "ERROR" and iss.violating_count > 0 for iss in self.relational_issues)
        return entities_ok and relations_ok

    @property
    def total_errors(self) -> int:
        """Total count of ERROR violations across all entities and relations."""
        entity_errs = sum(
            sum(i.violating_count for i in rep.issues if i.severity == "ERROR")
            for rep in self.entity_reports.values()
        )
        relation_errs = sum(i.violating_count for i in self.relational_issues if i.severity == "ERROR")
        return entity_errs + relation_errs

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_errors": self.total_errors,
            "entities": {name: rep.to_dict() for name, rep in self.entity_reports.items()},
            "relational_issues": [i.to_dict() for i in self.relational_issues],
        }

    def to_markdown(self) -> str:
        """Generate markdown summary for the comprehensive report."""
        status = "COMPLIANT" if self.is_valid else "NON-COMPLIANT"
        lines = [
            f"## Academic Data Consistency & Relational Integrity Audit: {status}",
            f"- **Datasets Audited**: {len(self.entity_reports)}",
            f"- **Total Inconsistency Errors**: {self.total_errors:,}",
            "",
            "| Entity / Relationship | Status | Rows Checked | Failed Checks | Total Violations |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for name, rep in self.entity_reports.items():
            rep_status = "PASS" if rep.is_valid else "FAIL"
            total_viols = sum(i.violating_count for i in rep.issues)
            lines.append(
                f"| `{name}` | `{rep_status}` | {rep.total_rows:,} | {rep.failed_rules} | {total_viols:,} |"
            )

        if self.relational_issues:
            lines.append("")
            lines.append("### Relational & Referential Integrity Violations")
            lines.append("| Relationship | Violations | % Violations | Description |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for rel in self.relational_issues:
                lines.append(f"| `{rel.rule_name}` | {rel.violating_count:,} | {rel.violation_pct:.1f}% | {rel.description} |")

        lines.append("")
        for rep in self.entity_reports.values():
            lines.append(rep.to_markdown())
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atomic Consistency Validation Rules
# ---------------------------------------------------------------------------


def validate_scores_range(
    df: pd.DataFrame,
    col: str = "score",
    entity_name: str = "assessments",
    min_val: float = 0.0,
    max_val: float = 100.0,
) -> Optional[ConsistencyIssue]:
    """Validate that numerical scores fall strictly between min_val and max_val (0 to 100)."""
    if col not in df.columns or len(df) == 0:
        return None

    numeric_col = pd.to_numeric(df[col], errors="coerce")
    valid_scores = numeric_col.dropna()
    total_checked = len(valid_scores)
    if total_checked == 0:
        return None

    out_of_bounds = valid_scores[(valid_scores < min_val) | (valid_scores > max_val)]
    violating_count = len(out_of_bounds)

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample = df.loc[out_of_bounds.head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name="scores_between_0_and_100",
        entity_name=entity_name,
        column_name=col,
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} scores out of allowed [{min_val}, {max_val}] range (min: {out_of_bounds.min()}, max: {out_of_bounds.max()}).",
        sample_violations=sample,
    )


def validate_attendance_range(
    df: pd.DataFrame,
    col: Optional[str] = None,
    entity_name: str = "attendance",
) -> Optional[ConsistencyIssue]:
    """Validate that aggregate attendance rates/percentages are within legitimate range [0.0, 100.0]."""
    target_col = col
    if target_col is None:
        for candidate in ["attendance_pct", "attendance_rate", "rate", "percentage"]:
            if candidate in df.columns:
                target_col = candidate
                break

    if target_col is None or target_col not in df.columns or len(df) == 0:
        return None

    numeric_col = pd.to_numeric(df[target_col], errors="coerce").dropna()
    total_checked = len(numeric_col)
    if total_checked == 0:
        return None

    # Handle ratio scale [0.0, 1.0] vs percentage scale [0.0, 100.0]
    max_limit = 1.0 if numeric_col.max() <= 1.0 and numeric_col.min() >= 0.0 else 100.0

    violating = numeric_col[(numeric_col < 0.0) | (numeric_col > max_limit)]
    violating_count = len(violating)

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample = df.loc[violating.head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name="attendance_within_valid_range",
        entity_name=entity_name,
        column_name=target_col,
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} attendance entries outside valid range [0, {max_limit}].",
        sample_violations=sample,
    )


def validate_approved_attendance_statuses(
    df: pd.DataFrame,
    col: str = "status",
    entity_name: str = "attendance",
    approved_statuses: Optional[Set[str]] = None,
) -> Optional[ConsistencyIssue]:
    """Validate that attendance status codes match approved canonical categories."""
    if col not in df.columns or len(df) == 0:
        return None

    allowed = approved_statuses or APPROVED_ATTENDANCE_STATUSES
    statuses = df[col].dropna().astype(str).str.strip()
    total_checked = len(statuses)
    if total_checked == 0:
        return None

    unapproved = statuses[~statuses.isin(allowed)]
    violating_count = len(unapproved)

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    unique_unapproved = list(unapproved.unique())[:5]
    sample = df.loc[unapproved.head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name="approved_attendance_statuses",
        entity_name=entity_name,
        column_name=col,
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} status entries not in approved list {sorted(list(allowed))}. Unapproved values: {unique_unapproved}.",
        sample_violations=sample,
    )


def validate_dates_logical(
    df: pd.DataFrame,
    date_col: str,
    entity_name: str,
    rule_name: str,
    min_date: Optional[str] = "2020-01-01",
    max_date: Optional[str] = "2030-12-31",
) -> Optional[ConsistencyIssue]:
    """Validate that date strings are parseable and fall within plausible academic epoch boundaries."""
    if date_col not in df.columns or len(df) == 0:
        return None

    raw_series = df[date_col].dropna()
    total_checked = len(raw_series)
    if total_checked == 0:
        return None

    parsed_dates = pd.to_datetime(raw_series, errors="coerce", format="mixed")

    # 1. Unparseable dates
    unparseable = raw_series[parsed_dates.isna()]
    unparseable_count = len(unparseable)

    # 2. Chronologically implausible dates (out of min/max epoch bounds)
    min_dt = pd.to_datetime(min_date) if min_date else None
    max_dt = pd.to_datetime(max_date) if max_date else None

    out_of_bounds = pd.Series(False, index=raw_series.index)
    if min_dt is not None:
        out_of_bounds |= parsed_dates < min_dt
    if max_dt is not None:
        out_of_bounds |= parsed_dates > max_dt

    out_of_bounds_count = int(out_of_bounds.sum())
    violating_count = unparseable_count + out_of_bounds_count

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample_indices = list(unparseable.index[:3]) + list(raw_series[out_of_bounds].index[:3])
    sample = df.loc[sample_indices[:5]].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name=rule_name,
        entity_name=entity_name,
        column_name=date_col,
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} date entries are invalid or outside academic calendar window [{min_date}, {max_date}].",
        sample_violations=sample,
    )


def validate_submission_timing_logic(
    submissions_df: pd.DataFrame,
    assignments_df: Optional[pd.DataFrame] = None,
    sub_date_col: str = "submission_date",
    due_date_col: str = "due_date",
    assignment_id_col: str = "assignment_id",
) -> Optional[ConsistencyIssue]:
    """Validate that submission dates are logically coherent relative to due dates."""
    if not isinstance(submissions_df, pd.DataFrame) or len(submissions_df) == 0:
        return None
    if sub_date_col not in submissions_df.columns:
        return None

    # Merge due_date if not present
    work_df = submissions_df.copy()
    if due_date_col not in work_df.columns:
        if assignments_df is None or assignment_id_col not in assignments_df.columns:
            return None
        due_map = assignments_df[[assignment_id_col, due_date_col]].drop_duplicates(subset=[assignment_id_col])
        work_df = work_df.merge(due_map, on=assignment_id_col, how="left")

    if due_date_col not in work_df.columns:
        return None

    sub_dt = pd.to_datetime(work_df[sub_date_col], errors="coerce", format="mixed")
    due_dt = pd.to_datetime(work_df[due_date_col], errors="coerce", format="mixed")

    valid_mask = sub_dt.notna() & due_dt.notna()
    total_checked = int(valid_mask.sum())
    if total_checked == 0:
        return None

    delta_days = (sub_dt[valid_mask] - due_dt[valid_mask]).dt.total_seconds() / 86400.0

    # Anomalies: submitted more than 180 days before due date, or more than 180 days after due date
    implausible = delta_days[(delta_days < -180.0) | (delta_days > 180.0)]
    violating_count = len(implausible)

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample = work_df.loc[implausible.head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name="submission_dates_logically_valid",
        entity_name="submissions",
        column_name=sub_date_col,
        severity="WARNING",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} submissions have implausible timing relative to assignment deadline (|delta| > 180 days).",
        sample_violations=sample,
    )


def validate_id_format(
    df: pd.DataFrame,
    id_col: str,
    entity_name: str,
    pattern: str = r"^[A-Z0-9_\-]+$",
    rule_name: Optional[str] = None,
) -> Optional[ConsistencyIssue]:
    """Validate that ID columns conform to non-empty alphanumeric formatting rules."""
    if id_col not in df.columns or len(df) == 0:
        return None

    series = df[id_col].dropna().astype(str).str.strip()
    total_checked = len(series)
    if total_checked == 0:
        return None

    regex = re.compile(pattern)
    invalid_mask = ~series.apply(lambda x: bool(regex.match(x)))
    violating_count = int(invalid_mask.sum())

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample = df.loc[series[invalid_mask].head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name=rule_name or f"valid_{id_col}_format",
        entity_name=entity_name,
        column_name=id_col,
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} IDs fail syntactic pattern format `{pattern}`.",
        sample_violations=sample,
    )


def validate_referential_integrity(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    foreign_key: str,
    child_entity: str,
    parent_entity: str,
) -> Optional[ConsistencyIssue]:
    """Validate referential integrity between child and parent tables (no orphan keys)."""
    if foreign_key not in child_df.columns or foreign_key not in parent_df.columns:
        return None

    child_keys = child_df[foreign_key].dropna().astype(str).str.strip().str.upper()
    total_checked = len(child_keys)
    if total_checked == 0:
        return None

    parent_keys = set(parent_df[foreign_key].dropna().astype(str).str.strip().str.upper().unique())

    orphans = child_keys[~child_keys.isin(parent_keys)]
    violating_count = len(orphans)

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample = child_df.loc[orphans.head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name=f"referential_integrity_{child_entity}_{foreign_key}",
        entity_name=child_entity,
        column_name=foreign_key,
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} orphan records in `{child_entity}` reference non-existent `{foreign_key}` in `{parent_entity}`.",
        sample_violations=sample,
    )


def validate_enrollment_relationships(
    activity_df: pd.DataFrame,
    enrollments_df: pd.DataFrame,
    activity_entity: str = "attendance",
    student_id_col: str = "student_id",
    course_id_col: str = "course_id",
) -> Optional[ConsistencyIssue]:
    """Validate that course activity belongs strictly to enrolled students."""
    if (
        student_id_col not in activity_df.columns
        or course_id_col not in activity_df.columns
        or student_id_col not in enrollments_df.columns
        or course_id_col not in enrollments_df.columns
    ):
        return None

    # Construct paired keys (student_id, course_id)
    act_pairs = (
        activity_df[student_id_col].astype(str).str.strip().str.upper()
        + "::"
        + activity_df[course_id_col].astype(str).str.strip().str.upper()
    )
    total_checked = len(act_pairs)
    if total_checked == 0:
        return None

    enrolled_pairs = set(
        enrollments_df[student_id_col].astype(str).str.strip().str.upper()
        + "::"
        + enrollments_df[course_id_col].astype(str).str.strip().str.upper()
    )

    unauthorized = act_pairs[~act_pairs.isin(enrolled_pairs)]
    violating_count = len(unauthorized)

    if violating_count == 0:
        return None

    pct = round((violating_count / total_checked) * 100.0, 2)
    sample = activity_df.loc[unauthorized.head(5).index].to_dict(orient="records")

    return ConsistencyIssue(
        rule_name="valid_enrollment_relationships",
        entity_name=activity_entity,
        column_name=f"({student_id_col}, {course_id_col})",
        severity="ERROR",
        violating_count=violating_count,
        total_checked=total_checked,
        violation_pct=pct,
        description=f"{violating_count:,} records in `{activity_entity}` belong to students not enrolled in the respective course.",
        sample_violations=sample,
    )


# ---------------------------------------------------------------------------
# Entity & Batch Consistency Pipelines
# ---------------------------------------------------------------------------


def validate_entity_consistency(
    df: pd.DataFrame,
    entity_name: str,
) -> EntityConsistencyReport:
    """Run all applicable consistency checks on a single academic entity DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    report = EntityConsistencyReport(entity_name=entity_name, total_rows=len(df))
    clean_name = entity_name.strip().lower()

    checks: List[Optional[ConsistencyIssue]] = []

    # 1. ID checks
    if "student_id" in df.columns:
        checks.append(validate_id_format(df, "student_id", entity_name, rule_name="valid_student_ids"))
    if "course_id" in df.columns:
        checks.append(validate_id_format(df, "course_id", entity_name, rule_name="valid_course_ids"))

    # 2. Score checks (0 to 100)
    for col in ["score", "marks", "grade"]:
        if col in df.columns:
            checks.append(validate_scores_range(df, col=col, entity_name=entity_name))

    # 3. Attendance specific checks
    if clean_name == "attendance":
        if "status" in df.columns:
            checks.append(validate_approved_attendance_statuses(df, col="status", entity_name=entity_name))
        if "date" in df.columns:
            checks.append(validate_dates_logical(df, "date", entity_name, rule_name="attendance_dates_valid"))
        checks.append(validate_attendance_range(df, entity_name=entity_name))

    # 4. Assignments specific checks
    if clean_name == "assignments":
        if "due_date" in df.columns:
            checks.append(validate_dates_logical(df, "due_date", entity_name, rule_name="due_dates_logically_valid"))

    # 5. Submissions specific checks
    if clean_name == "submissions":
        if "submission_date" in df.columns:
            checks.append(validate_dates_logical(df, "submission_date", entity_name, rule_name="submission_dates_logically_valid"))

    # 6. Exams specific checks
    if clean_name == "exams":
        for date_candidate in ["exam_date", "date"]:
            if date_candidate in df.columns:
                checks.append(validate_dates_logical(df, date_candidate, entity_name, rule_name="exam_dates_logically_valid"))
                break

    # Aggregate issues
    for issue in checks:
        if issue is not None:
            report.failed_rules += 1
            report.issues.append(issue)
        else:
            report.passed_rules += 1

    return report


def validate_academic_consistency(
    datasets: Dict[str, pd.DataFrame],
    strict: bool = False,
) -> AcademicConsistencyReport:
    """Perform full-system consistency and relational integrity validation across all academic datasets.

    Args:
        datasets: Dictionary mapping entity names to DataFrames.
        strict: If True, raises DataValidationError upon encountering any ERROR consistency issue.

    Returns:
        AcademicConsistencyReport with comprehensive audit logs and pass/fail statuses.

    Raises:
        DataValidationError: If strict=True and any consistency checks fail.
    """
    master_report = AcademicConsistencyReport()

    # 1. Entity-level validation
    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            continue
        ent_report = validate_entity_consistency(df, name)
        master_report.entity_reports[name] = ent_report

    # 2. Relational Integrity: Student foreign keys
    if "students" in datasets:
        for child in ["enrollments", "attendance", "submissions", "exams"]:
            if child in datasets:
                issue = validate_referential_integrity(
                    child_df=datasets[child],
                    parent_df=datasets["students"],
                    foreign_key="student_id",
                    child_entity=child,
                    parent_entity="students",
                )
                if issue:
                    master_report.relational_issues.append(issue)

    # 3. Relational Integrity: Course foreign keys
    if "courses" in datasets:
        for child in ["enrollments", "attendance", "assignments", "exams"]:
            if child in datasets:
                issue = validate_referential_integrity(
                    child_df=datasets[child],
                    parent_df=datasets["courses"],
                    foreign_key="course_id",
                    child_entity=child,
                    parent_entity="courses",
                )
                if issue:
                    master_report.relational_issues.append(issue)

    # 4. Enrollment Relationships (activity belongs to enrolled courses)
    if "enrollments" in datasets:
        if "attendance" in datasets:
            issue = validate_enrollment_relationships(
                activity_df=datasets["attendance"],
                enrollments_df=datasets["enrollments"],
                activity_entity="attendance",
            )
            if issue:
                master_report.relational_issues.append(issue)

        if "exams" in datasets:
            issue = validate_enrollment_relationships(
                activity_df=datasets["exams"],
                enrollments_df=datasets["enrollments"],
                activity_entity="exams",
            )
            if issue:
                master_report.relational_issues.append(issue)

    # 5. Submission & Due Date Logic
    if "submissions" in datasets and "assignments" in datasets:
        issue = validate_submission_timing_logic(
            submissions_df=datasets["submissions"],
            assignments_df=datasets["assignments"],
        )
        if issue:
            master_report.relational_issues.append(issue)

    logger.info(
        "Academic consistency check complete. Valid: %s, Total Errors: %d across %d datasets.",
        master_report.is_valid,
        master_report.total_errors,
        len(master_report.entity_reports),
    )

    if strict and not master_report.is_valid:
        raise DataValidationError(
            f"Academic consistency validation failed with {master_report.total_errors} errors. See report for details."
        )

    return master_report
