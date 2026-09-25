"""Academic duplicate detection, business-key deduplication, and before/after audit reporting.

This module ensures that duplicate records across academic entities (students, courses,
attendance, assignments, submissions, exams) are detected using meaningful business keys
rather than naive row-level drops, preserving data integrity and avoiding blind deletions.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.duplicate_detection")

# Canonical primary business keys for core academic entities
DEFAULT_BUSINESS_KEYS: Dict[str, List[str]] = {
    "students": ["student_id"],
    "courses": ["course_id"],
    "enrollments": ["student_id", "course_id"],
    "attendance": ["student_id", "course_id", "date"],
    "assignments": ["assignment_id"],
    "submissions": ["assignment_id", "student_id"],
    "exams": ["exam_id", "student_id"],
}

# Domain-specific fallback keys when primary synthetic ID is absent
FALLBACK_BUSINESS_KEYS: Dict[str, List[str]] = {
    "assignments": ["course_id", "title"],
    "exams": ["student_id", "course_id", "exam_type"],
    "students": ["name", "program"],
    "courses": ["course_name", "faculty"],
}

# Default resolution strategies by entity
DEFAULT_RESOLUTION_STRATEGIES: Dict[str, str] = {
    "students": "most_complete",
    "courses": "most_complete",
    "enrollments": "first",
    "attendance": "last",
    "assignments": "most_complete",
    "submissions": "highest_score",
    "exams": "highest_score",
}


@dataclass
class DuplicateMetrics:
    """Metrics snapshot for duplicate status at a specific point in time."""

    total_rows: int
    unique_business_keys: int
    duplicate_groups: int
    key_duplicate_rows: int
    exact_duplicate_rows: int
    conflicting_rows: int

    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class DuplicateReport:
    """Detailed before-and-after audit report for duplicate detection and resolution."""

    entity_name: str
    business_keys: List[str]
    resolution_strategy: str
    before_metrics: DuplicateMetrics
    after_metrics: DuplicateMetrics
    rows_removed: int
    rows_retained: int
    quarantined_records: Optional[pd.DataFrame] = None
    applied_rules: List[str] = field(default_factory=list)

    @property
    def total_rows_before(self) -> int:
        """Total rows before deduplication."""
        return self.before_metrics.total_rows

    @property
    def total_rows_after(self) -> int:
        """Total rows after deduplication."""
        return self.after_metrics.total_rows

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format, excluding raw DataFrame for serialization."""
        return {
            "entity_name": self.entity_name,
            "business_keys": self.business_keys,
            "resolution_strategy": self.resolution_strategy,
            "total_rows_before": self.total_rows_before,
            "total_rows_after": self.total_rows_after,
            "rows_removed": self.rows_removed,
            "rows_retained": self.rows_retained,
            "before_metrics": self.before_metrics.to_dict(),
            "after_metrics": self.after_metrics.to_dict(),
            "quarantined_count": len(self.quarantined_records) if self.quarantined_records is not None else 0,
            "applied_rules": self.applied_rules,
        }

    def to_markdown(self) -> str:
        """Generate structured markdown summary of before/after duplicate analysis."""
        lines = [
            f"### Duplicate Audit Report: `{self.entity_name}`",
            f"- **Business Keys**: `{', '.join(self.business_keys)}`",
            f"- **Resolution Strategy**: `{self.resolution_strategy}`",
            f"- **Rows Before**: {self.total_rows_before:,}",
            f"- **Rows After**: {self.total_rows_after:,}",
            f"- **Rows Quarantined / Removed**: {self.rows_removed:,}",
            "",
            "| Metric | Before Cleaning | After Cleaning | Net Change |",
            "| :--- | :--- | :--- | :--- |",
        ]

        b = self.before_metrics
        a = self.after_metrics

        def row(name: str, before_val: int, after_val: int) -> str:
            diff = after_val - before_val
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            return f"| {name} | {before_val:,} | {after_val:,} | {diff_str} |"

        lines.append(row("Total Records", b.total_rows, a.total_rows))
        lines.append(row("Duplicate Groups", b.duplicate_groups, a.duplicate_groups))
        lines.append(row("Key Duplicate Rows", b.key_duplicate_rows, a.key_duplicate_rows))
        lines.append(row("Exact Duplicate Rows", b.exact_duplicate_rows, a.exact_duplicate_rows))
        lines.append(row("Conflicting Rows", b.conflicting_rows, a.conflicting_rows))

        lines.append("")
        if self.applied_rules:
            lines.append("**Applied Resolution Rules:**")
            for rule in self.applied_rules:
                lines.append(f"- {rule}")
            lines.append("")

        if self.quarantined_records is not None and not self.quarantined_records.empty:
            lines.append(f"**Quarantine Audit Sample (First {min(5, len(self.quarantined_records))} of {len(self.quarantined_records)}):**")
            cols_to_show = [c for c in self.business_keys if c in self.quarantined_records.columns]
            if "quarantine_reason" in self.quarantined_records.columns:
                cols_to_show.append("quarantine_reason")
            sample_df = self.quarantined_records[cols_to_show].head(5)
            lines.append(_format_markdown_table(sample_df))

        return "\n".join(lines)


def _format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without external tabulate dependency."""
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_vals = [str(val).replace("|", "/") if pd.notna(val) else "" for val in row]
        data_lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join([header_line, sep_line] + data_lines)


@dataclass
class BatchDuplicateReport:
    """Aggregated duplicate detection report across an entire academic dataset batch."""

    reports: Dict[str, DuplicateReport] = field(default_factory=dict)

    @property
    def total_rows_before(self) -> int:
        """Total rows across all entities before deduplication."""
        return sum(r.total_rows_before for r in self.reports.values())

    @property
    def total_rows_after(self) -> int:
        """Total rows across all entities after deduplication."""
        return sum(r.total_rows_after for r in self.reports.values())

    @property
    def total_removed(self) -> int:
        """Total rows removed across all entities."""
        return sum(r.rows_removed for r in self.reports.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch report to dictionary."""
        return {
            "total_rows_before": self.total_rows_before,
            "total_rows_after": self.total_rows_after,
            "total_removed": self.total_removed,
            "entities": {name: r.to_dict() for name, r in self.reports.items()},
        }

    def to_markdown(self) -> str:
        """Generate markdown summary for the batch report."""
        lines = [
            "## Academic Dataset Duplicate Detection & Deduplication Summary",
            f"- **Entities Processed**: {len(self.reports)}",
            f"- **Total Rows Before**: {self.total_rows_before:,}",
            f"- **Total Rows After**: {self.total_rows_after:,}",
            f"- **Total Rows Quarantined / Removed**: {self.total_removed:,}",
            "",
            "| Entity | Business Keys | Strategy | Before | After | Removed | Conflicting |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for name, r in self.reports.items():
            keys_str = ", ".join(r.business_keys)
            lines.append(
                f"| `{name}` | `{keys_str}` | `{r.resolution_strategy}` | {r.total_rows_before:,} | "
                f"{r.total_rows_after:,} | {r.rows_removed:,} | {r.before_metrics.conflicting_rows:,} |"
            )

        lines.append("")
        for name, r in self.reports.items():
            lines.append(r.to_markdown())
            lines.append("")

        return "\n".join(lines)


def resolve_business_keys(
    df: pd.DataFrame,
    entity_name: str,
    custom_keys: Optional[List[str]] = None,
) -> List[str]:
    """Resolve and validate the business keys to use for duplicate detection.

    Args:
        df: Input DataFrame.
        entity_name: Name of the academic entity (e.g., 'students', 'attendance').
        custom_keys: Optional explicit business keys.

    Returns:
        List of verified column names forming the business key.

    Raises:
        DataValidationError: If neither custom nor default business keys exist in DataFrame.
    """
    if custom_keys is not None:
        missing = [k for k in custom_keys if k not in df.columns]
        if missing:
            raise DataValidationError(f"Specified business keys {missing} not found in DataFrame for '{entity_name}'.")
        return list(custom_keys)

    clean_entity = entity_name.strip().lower()

    # Check primary default keys
    if clean_entity in DEFAULT_BUSINESS_KEYS:
        primary = DEFAULT_BUSINESS_KEYS[clean_entity]
        if all(k in df.columns for k in primary):
            return primary

    # Check fallback keys
    if clean_entity in FALLBACK_BUSINESS_KEYS:
        fallback = FALLBACK_BUSINESS_KEYS[clean_entity]
        if all(k in df.columns for k in fallback):
            logger.info("Using fallback business keys %s for entity '%s'", fallback, entity_name)
            return fallback

    # If entity not recognized or columns missing, raise descriptive error
    available = list(df.columns)
    expected_primary = DEFAULT_BUSINESS_KEYS.get(clean_entity, [])
    raise DataValidationError(
        f"Cannot resolve business keys for entity '{entity_name}'. "
        f"Expected columns {expected_primary}, but DataFrame only has columns: {available}."
    )


def compute_duplicate_metrics(df: pd.DataFrame, business_keys: List[str]) -> DuplicateMetrics:
    """Calculate detailed duplicate statistics on a DataFrame using given business keys.

    Args:
        df: Input DataFrame.
        business_keys: List of column names forming the business key.

    Returns:
        DuplicateMetrics dataclass instance.
    """
    total_rows = len(df)
    if total_rows == 0:
        return DuplicateMetrics(
            total_rows=0,
            unique_business_keys=0,
            duplicate_groups=0,
            key_duplicate_rows=0,
            exact_duplicate_rows=0,
            conflicting_rows=0,
        )

    # Missing business keys check
    for key in business_keys:
        if key not in df.columns:
            raise DataValidationError(f"Business key '{key}' not found in DataFrame.")

    # 1. Exact row duplicates (all columns match)
    exact_duplicate_rows = int(df.duplicated(keep=False).sum())

    # 2. Key duplicates (rows sharing the business keys)
    key_dup_mask = df.duplicated(subset=business_keys, keep=False)
    key_duplicate_rows = int(key_dup_mask.sum())

    # Unique business keys
    unique_business_keys = int(df.drop_duplicates(subset=business_keys).shape[0])

    if key_duplicate_rows == 0:
        return DuplicateMetrics(
            total_rows=total_rows,
            unique_business_keys=unique_business_keys,
            duplicate_groups=0,
            key_duplicate_rows=0,
            exact_duplicate_rows=exact_duplicate_rows,
            conflicting_rows=0,
        )

    # 3. Duplicate groups
    dup_df = df[key_dup_mask]
    dup_groups = dup_df.groupby(business_keys, sort=False, dropna=False)
    duplicate_groups_count = dup_groups.ngroups

    # 4. Conflicting rows: Within a duplicate business-key group, non-key columns have >1 distinct non-null value
    non_key_cols = [c for c in df.columns if c not in business_keys]
    conflicting_indices: Set[Any] = set()

    if non_key_cols:
        for _, group in dup_groups:
            is_group_conflicting = False
            for col in non_key_cols:
                # Count non-null unique values
                series = group[col].dropna()
                if series.nunique() > 1:
                    is_group_conflicting = True
                    break
            if is_group_conflicting:
                conflicting_indices.update(group.index.tolist())

    conflicting_rows_count = len(conflicting_indices)

    return DuplicateMetrics(
        total_rows=total_rows,
        unique_business_keys=unique_business_keys,
        duplicate_groups=duplicate_groups_count,
        key_duplicate_rows=key_duplicate_rows,
        exact_duplicate_rows=exact_duplicate_rows,
        conflicting_rows=conflicting_rows_count,
    )


def detect_duplicates(
    df: pd.DataFrame,
    entity_name: str,
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateMetrics]:
    """Detect duplicate records in a DataFrame and annotate duplicate status.

    Args:
        df: Input DataFrame.
        entity_name: Academic entity name (e.g. 'attendance', 'students').
        business_keys: Optional explicit business keys.

    Returns:
        Tuple of:
          - Annotated DataFrame copy with columns:
            `is_duplicate_key` (bool), `is_exact_duplicate` (bool), `is_conflicting` (bool), `duplicate_group_id` (str)
          - DuplicateMetrics snapshot.

    Raises:
        DataValidationError: If input is not a DataFrame or business keys are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    keys = resolve_business_keys(df, entity_name, business_keys)
    metrics = compute_duplicate_metrics(df, keys)

    annotated = df.copy()
    if len(annotated) == 0:
        annotated["is_duplicate_key"] = pd.Series(dtype=bool)
        annotated["is_exact_duplicate"] = pd.Series(dtype=bool)
        annotated["is_conflicting"] = pd.Series(dtype=bool)
        annotated["duplicate_group_id"] = pd.Series(dtype=str)
        return annotated, metrics

    # Mark exact duplicates
    annotated["is_exact_duplicate"] = annotated.duplicated(keep=False)

    # Mark key duplicates
    annotated["is_duplicate_key"] = annotated.duplicated(subset=keys, keep=False)

    # Assign group IDs and detect conflicts
    annotated["is_conflicting"] = False
    annotated["duplicate_group_id"] = ""

    dup_mask = annotated["is_duplicate_key"]
    if dup_mask.any():
        non_key_cols = [c for c in df.columns if c not in keys]

        for group_idx, (_, group) in enumerate(annotated[dup_mask].groupby(keys, sort=False, dropna=False), 1):
            group_id = f"{entity_name}_grp_{group_idx:04d}"
            annotated.loc[group.index, "duplicate_group_id"] = group_id

            if non_key_cols:
                has_conflict = any(group[col].dropna().nunique() > 1 for col in non_key_cols)
                if has_conflict:
                    annotated.loc[group.index, "is_conflicting"] = True

    logger.info(
        "Duplicate detection for '%s' (%s): %d key duplicates, %d exact, %d conflicting across %d rows.",
        entity_name,
        keys,
        metrics.key_duplicate_rows,
        metrics.exact_duplicate_rows,
        metrics.conflicting_rows,
        metrics.total_rows,
    )

    return annotated, metrics


def _resolve_group_winner(
    group: pd.DataFrame,
    strategy: str,
    business_keys: List[str],
) -> Tuple[Any, List[Tuple[Any, str]]]:
    """Select the single row index to keep in a duplicate group based on strategy.

    Returns:
        Tuple of (winner_index, list of (loser_index, quarantine_reason)).
    """
    indices = group.index.tolist()
    if len(indices) <= 1:
        return indices[0], []

    reasons: List[Tuple[Any, str]] = []

    # 1. Exact duplicates: if all rows are identical, keep first
    if group.drop_duplicates().shape[0] == 1:
        winner = indices[0]
        for idx in indices[1:]:
            reasons.append((idx, f"Exact identical duplicate of record at index {winner}."))
        return winner, reasons

    # Strategy: "highest_score" (for submissions, exams)
    if strategy in {"highest_score", "best_score"}:
        score_col = None
        for candidate in ["score", "marks", "grade", "points"]:
            if candidate in group.columns and pd.api.types.is_numeric_dtype(group[candidate]):
                score_col = candidate
                break

        if score_col is not None:
            # Sort by score descending (NaNs placed at bottom), then completeness descending
            non_null_counts = group.notna().sum(axis=1)
            temp = group.copy()
            temp["_score"] = pd.to_numeric(temp[score_col], errors="coerce")
            temp["_completeness"] = non_null_counts
            temp = temp.sort_values(by=["_score", "_completeness"], ascending=[False, False])
            winner = temp.index[0]
            winner_score = group.loc[winner, score_col]

            for idx in indices:
                if idx != winner:
                    idx_score = group.loc[idx, score_col]
                    reasons.append(
                        (idx, f"Superseded by higher/valid score at index {winner} ({winner_score} vs {idx_score}).")
                    )
            return winner, reasons

        # If no score column found, fall back to "most_complete"
        strategy = "most_complete"

    # Strategy: "last" (e.g. for attendance corrections or latest submissions)
    if strategy == "last":
        # Check if date/timestamp column exists to sort chronologically
        date_col = None
        for candidate in ["date", "submission_date", "updated_at", "timestamp"]:
            if candidate in group.columns:
                date_col = candidate
                break

        if date_col is not None:
            temp = group.copy()
            temp["_parsed_dt"] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.sort_values(by="_parsed_dt", ascending=True)
            winner = temp.index[-1]
            winner_date = group.loc[winner, date_col]
            for idx in indices:
                if idx != winner:
                    reasons.append(
                        (idx, f"Superseded by chronologically newer record at index {winner} ({winner_date}).")
                    )
            return winner, reasons
        else:
            winner = indices[-1]
            for idx in indices[:-1]:
                reasons.append((idx, f"Superseded by last recorded entry at index {winner}."))
            return winner, reasons

    # Strategy: "first"
    if strategy == "first":
        winner = indices[0]
        for idx in indices[1:]:
            reasons.append((idx, f"Superseded by first recorded entry at index {winner}."))
        return winner, reasons

    # Strategy: "most_complete" (default for students, courses, assignments)
    # Ranks by maximum non-null values across non-key columns
    non_null_counts = group.notna().sum(axis=1)
    max_non_null = non_null_counts.max()
    candidates = non_null_counts[non_null_counts == max_non_null].index.tolist()

    # Tie-break candidate by last recorded entry
    winner = candidates[-1]
    winner_completeness = non_null_counts[winner]

    for idx in indices:
        if idx != winner:
            idx_comp = non_null_counts[idx]
            reasons.append(
                (idx, f"Superseded by more complete record at index {winner} ({winner_completeness} non-nulls vs {idx_comp}).")
            )

    return winner, reasons


def clean_duplicates(
    df: pd.DataFrame,
    entity_name: str,
    business_keys: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    action: str = "resolve",
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Clean duplicate academic records using business keys and an explicit resolution strategy.

    Does not blindly remove records. Evaluates conflicts, retains the optimal record
    (based on score, completeness, or chronological order), and quarantines discarded records
    with full reasoning for auditability.

    Args:
        df: Input DataFrame.
        entity_name: Academic entity name (e.g. 'students', 'courses', 'attendance', 'assignments', 'submissions', 'exams').
        business_keys: Optional explicit business keys.
        strategy: Deduplication strategy ('most_complete', 'highest_score', 'last', 'first', 'merge_non_null', 'flag_only').
                  If None, selects the domain default for the entity.
        action: 'resolve' (drops duplicates according to strategy) or 'flag_only' (annotates without dropping).

    Returns:
        Tuple of:
          - Cleaned (or flagged) DataFrame.
          - DuplicateReport with complete before/after audit metrics and quarantined records.

    Raises:
        DataValidationError: If input is not a DataFrame or business keys are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got: {type(df).__name__}")

    keys = resolve_business_keys(df, entity_name, business_keys)
    applied_rules: List[str] = []

    # Determine strategy
    chosen_strategy = strategy
    if chosen_strategy is None:
        chosen_strategy = DEFAULT_RESOLUTION_STRATEGIES.get(entity_name.lower(), "most_complete")

    if action == "flag_only":
        chosen_strategy = "flag_only"

    # Compute baseline "before" metrics
    before_metrics = compute_duplicate_metrics(df, keys)

    # Edge case: Empty DataFrame
    if len(df) == 0:
        empty_report = DuplicateReport(
            entity_name=entity_name,
            business_keys=keys,
            resolution_strategy=chosen_strategy,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            rows_removed=0,
            rows_retained=0,
            quarantined_records=pd.DataFrame(),
            applied_rules=["Empty DataFrame provided; no cleaning needed."],
        )
        return df.copy(), empty_report

    # Edge case: No duplicates present
    if before_metrics.key_duplicate_rows == 0:
        no_dup_report = DuplicateReport(
            entity_name=entity_name,
            business_keys=keys,
            resolution_strategy=chosen_strategy,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            rows_removed=0,
            rows_retained=len(df),
            quarantined_records=pd.DataFrame(),
            applied_rules=["No business key duplicates detected. Preserved 100% of records."],
        )
        return df.copy(), no_dup_report

    # Strategy: "flag_only"
    if chosen_strategy == "flag_only":
        flagged_df, _ = detect_duplicates(df, entity_name, business_keys=keys)
        applied_rules.append("Flag-only strategy: Annotated duplicates with flags without dropping any records.")
        report = DuplicateReport(
            entity_name=entity_name,
            business_keys=keys,
            resolution_strategy="flag_only",
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            rows_removed=0,
            rows_retained=len(df),
            quarantined_records=pd.DataFrame(),
            applied_rules=applied_rules,
        )
        return flagged_df, report

    # Strategy: "merge_non_null"
    if chosen_strategy == "merge_non_null":
        # Group by business keys and combine non-null values
        applied_rules.append(
            f"Merged non-null attributes within {before_metrics.duplicate_groups} duplicate groups using business keys {keys}."
        )
        # Identify rows to keep and quarantine
        non_key_cols = [c for c in df.columns if c not in keys]

        kept_rows: List[pd.Series] = []
        quarantined_list: List[Dict[str, Any]] = []

        for _, group in df.groupby(keys, sort=False, dropna=False):
            if len(group) == 1:
                kept_rows.append(group.iloc[0])
            else:
                # Merge non-nulls: take first non-null for each column
                merged_series = group.iloc[0].copy()
                for col in non_key_cols:
                    valid_vals = group[col].dropna()
                    if len(valid_vals) > 0:
                        merged_series[col] = valid_vals.iloc[-1]  # Prefer newest non-null

                kept_rows.append(merged_series)

                # Log quarantined rows
                for idx in group.index[1:]:
                    q_row = group.loc[idx].to_dict()
                    q_row["quarantine_reason"] = f"Merged into consolidated business key record for {keys}."
                    quarantined_list.append(q_row)

        cleaned_df = pd.DataFrame(kept_rows).reset_index(drop=True)
        quarantined_df = pd.DataFrame(quarantined_list)
        after_metrics = compute_duplicate_metrics(cleaned_df, keys)

        report = DuplicateReport(
            entity_name=entity_name,
            business_keys=keys,
            resolution_strategy=chosen_strategy,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            rows_removed=len(df) - len(cleaned_df),
            rows_retained=len(cleaned_df),
            quarantined_records=quarantined_df,
            applied_rules=applied_rules,
        )
        return cleaned_df, report

    # Standard resolution strategies ("most_complete", "highest_score", "last", "first")
    applied_rules.append(
        f"Resolved duplicate business keys {keys} using strategy '{chosen_strategy}' across "
        f"{before_metrics.duplicate_groups} collision groups."
    )

    winner_indices: Set[Any] = set()
    quarantine_audit: List[Tuple[Any, str]] = []

    # Non-duplicate rows are automatically kept
    non_dup_mask = ~df.duplicated(subset=keys, keep=False)
    winner_indices.update(df[non_dup_mask].index.tolist())

    # Resolve each duplicate group
    dup_df = df[~non_dup_mask]
    for _, group in dup_df.groupby(keys, sort=False, dropna=False):
        winner, losers_with_reasons = _resolve_group_winner(group, chosen_strategy, keys)
        winner_indices.add(winner)
        quarantine_audit.extend(losers_with_reasons)

    # Construct cleaned DataFrame
    cleaned_df = df.loc[sorted(winner_indices)].copy().reset_index(drop=True)

    # Construct quarantined DataFrame with audit reasons
    if quarantine_audit:
        loser_indices = [idx for idx, _ in quarantine_audit]
        quarantine_df = df.loc[loser_indices].copy()
        reason_map = {idx: r for idx, r in quarantine_audit}
        quarantine_df["quarantine_reason"] = quarantine_df.index.map(reason_map)
        quarantine_df = quarantine_df.reset_index(drop=True)
    else:
        quarantine_df = pd.DataFrame()

    after_metrics = compute_duplicate_metrics(cleaned_df, keys)

    rows_removed = len(df) - len(cleaned_df)
    report = DuplicateReport(
        entity_name=entity_name,
        business_keys=keys,
        resolution_strategy=chosen_strategy,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        rows_removed=rows_removed,
        rows_retained=len(cleaned_df),
        quarantined_records=quarantine_df,
        applied_rules=applied_rules,
    )

    logger.info(
        "Completed deduplication for '%s': %d rows before, %d rows after, %d quarantined using '%s'.",
        entity_name,
        before_metrics.total_rows,
        after_metrics.total_rows,
        rows_removed,
        chosen_strategy,
    )

    return cleaned_df, report


# --- Dedicated entity-level deduplication functions ---


def deduplicate_students(
    df: pd.DataFrame,
    strategy: str = "most_complete",
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Deduplicate student records using student_id (or name/program fallback)."""
    return clean_duplicates(df, entity_name="students", business_keys=business_keys, strategy=strategy)


def deduplicate_courses(
    df: pd.DataFrame,
    strategy: str = "most_complete",
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Deduplicate course records using course_id."""
    return clean_duplicates(df, entity_name="courses", business_keys=business_keys, strategy=strategy)


def deduplicate_attendance(
    df: pd.DataFrame,
    strategy: str = "last",
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Deduplicate attendance records using (student_id, course_id, date).

    Defaults to 'last' to capture the latest attendance audit correction.
    """
    return clean_duplicates(df, entity_name="attendance", business_keys=business_keys, strategy=strategy)


def deduplicate_assignments(
    df: pd.DataFrame,
    strategy: str = "most_complete",
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Deduplicate assignment records using assignment_id (or course_id, title)."""
    return clean_duplicates(df, entity_name="assignments", business_keys=business_keys, strategy=strategy)


def deduplicate_submissions(
    df: pd.DataFrame,
    strategy: str = "highest_score",
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Deduplicate student assignment submissions using (assignment_id, student_id).

    Defaults to 'highest_score' to preserve the student's best valid academic submission.
    """
    return clean_duplicates(df, entity_name="submissions", business_keys=business_keys, strategy=strategy)


def deduplicate_exams(
    df: pd.DataFrame,
    strategy: str = "highest_score",
    business_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DuplicateReport]:
    """Deduplicate student exam records using (exam_id, student_id) or (student_id, course_id, exam_type).

    Defaults to 'highest_score' to preserve the student's best valid exam performance.
    """
    return clean_duplicates(df, entity_name="exams", business_keys=business_keys, strategy=strategy)


def deduplicate_academic_dataset(
    datasets: Dict[str, pd.DataFrame],
    strategies: Optional[Dict[str, str]] = None,
    business_keys: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, pd.DataFrame], BatchDuplicateReport]:
    """Perform comprehensive duplicate detection and cleaning across a batch of academic datasets.

    Args:
        datasets: Dictionary mapping entity names to DataFrames.
        strategies: Optional mapping of entity name to deduplication strategy.
        business_keys: Optional mapping of entity name to custom business keys.

    Returns:
        Tuple of:
          - Dictionary of deduplicated DataFrames.
          - BatchDuplicateReport with before/after comparisons across all entities.
    """
    strategies = strategies or {}
    business_keys = business_keys or {}

    cleaned_datasets: Dict[str, pd.DataFrame] = {}
    batch_report = BatchDuplicateReport()

    for name, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            logger.warning("Skipping non-DataFrame entity '%s'", name)
            continue

        strat = strategies.get(name)
        keys = business_keys.get(name)

        cleaned_df, report = clean_duplicates(
            df=df,
            entity_name=name,
            business_keys=keys,
            strategy=strat,
        )

        cleaned_datasets[name] = cleaned_df
        batch_report.reports[name] = report

    return cleaned_datasets, batch_report
