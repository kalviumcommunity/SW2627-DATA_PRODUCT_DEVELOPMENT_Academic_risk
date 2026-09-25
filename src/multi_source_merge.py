"""Academic Multi-Source Merge and Data Integration Layer.

Integrates the 7 core academic data sources:
- students
- courses
- enrollments
- attendance
- assignments
- submissions
- exams

Performs comprehensive validation:
- Unmatched students and courses
- Cartesian duplicate expansion checks
- Missing academic relationships
- Row counts before and after joins
- Full preservation and tracking of orphan records (never hidden)
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.multi_source_merge")


@dataclass
class MergeAuditReport:
    """Audit report for multi-source integration, tracking joins, orphans, and integrity."""

    input_counts: Dict[str, int] = field(default_factory=dict)
    output_counts: Dict[str, int] = field(default_factory=dict)
    unmatched_students: List[str] = field(default_factory=list)
    unmatched_courses: List[str] = field(default_factory=list)
    orphan_attendance_students: List[str] = field(default_factory=list)
    orphan_submission_students: List[str] = field(default_factory=list)
    orphan_exam_students: List[str] = field(default_factory=list)
    orphan_activity_courses: List[str] = field(default_factory=list)
    students_with_no_attendance: List[str] = field(default_factory=list)
    students_with_no_submissions: List[str] = field(default_factory=list)
    students_with_no_exams: List[str] = field(default_factory=list)
    duplicate_expansion_detected: bool = False
    expansion_details: Dict[str, Any] = field(default_factory=dict)
    orphan_records: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def total_input_rows(self) -> int:
        """Total input rows across all ingested entities."""
        return sum(self.input_counts.values())

    @property
    def has_orphans(self) -> bool:
        """True if any orphan or unmatched foreign records were detected."""
        return bool(
            self.orphan_attendance_students
            or self.orphan_submission_students
            or self.orphan_exam_students
            or self.orphan_activity_courses
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary, excluding raw DataFrames for clean serialization."""
        return {
            "input_counts": self.input_counts,
            "output_counts": self.output_counts,
            "total_input_rows": self.total_input_rows,
            "has_orphans": self.has_orphans,
            "duplicate_expansion_detected": self.duplicate_expansion_detected,
            "unmatched_students_count": len(self.unmatched_students),
            "unmatched_courses_count": len(self.unmatched_courses),
            "orphan_attendance_count": len(self.orphan_attendance_students),
            "orphan_submissions_count": len(self.orphan_submission_students),
            "orphan_exams_count": len(self.orphan_exam_students),
            "students_with_no_attendance_count": len(self.students_with_no_attendance),
            "students_with_no_submissions_count": len(self.students_with_no_submissions),
            "students_with_no_exams_count": len(self.students_with_no_exams),
            "orphan_records_summary": {k: len(v) for k, v in self.orphan_records.items()},
        }

    def to_markdown(self) -> str:
        """Generate structured markdown summary for the multi-source merge audit."""
        expansion_str = "None (Safe)" if not self.duplicate_expansion_detected else "DETECTED (Warning)"
        orphan_str = "Found (Quarantined/Logged)" if self.has_orphans else "None (Clean)"

        lines = [
            "## Academic Multi-Source Merge Audit Report",
            f"- **Duplicate Expansion**: `{expansion_str}`",
            f"- **Orphan Records**: `{orphan_str}`",
            f"- **Unenrolled Students**: {len(self.unmatched_students):,}",
            f"- **Empty Courses**: {len(self.unmatched_courses):,}",
            "",
            "### Dataset Row Counts: Before & After Integration",
            "| Entity / View | Input Rows | Integrated Output Rows | Net Difference |",
            "| :--- | :--- | :--- | :--- |",
        ]

        all_keys = sorted(set(list(self.input_counts.keys()) + list(self.output_counts.keys())))
        for k in all_keys:
            in_cnt = self.input_counts.get(k, 0)
            out_cnt = self.output_counts.get(k, 0)
            diff = out_cnt - in_cnt
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            lines.append(f"| `{k}` | {in_cnt:,} | {out_cnt:,} | {diff_str} |")

        lines.append("")
        lines.append("### Relationship Completeness & Missing Activity")
        lines.append(f"- Enrolled students with **0 attendance records**: {len(self.students_with_no_attendance):,}")
        lines.append(f"- Enrolled students with **0 assignment submissions**: {len(self.students_with_no_submissions):,}")
        lines.append(f"- Enrolled students with **0 exam attempts**: {len(self.students_with_no_exams):,}")

        if self.has_orphans:
            lines.append("")
            lines.append("### Orphan Records Audit (Do Not Hide Orphans)")
            lines.append(f"- Orphan attendance student IDs: `{len(self.orphan_attendance_students):,}`")
            lines.append(f"- Orphan submission student IDs: `{len(self.orphan_submission_students):,}`")
            lines.append(f"- Orphan exam student IDs: `{len(self.orphan_exam_students):,}`")
            lines.append(f"- Orphan activity course IDs: `{len(self.orphan_activity_courses):,}`")

            for k, orphan_df in self.orphan_records.items():
                if not orphan_df.empty:
                    lines.append(f"\n#### Sample Orphan Records: `{k}` ({len(orphan_df):,} total)")
                    cols = list(orphan_df.columns[:6])
                    sample = orphan_df[cols].head(3)
                    headers = [str(c) for c in sample.columns]
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for _, r in sample.iterrows():
                        row_vals = [str(v) if pd.notna(v) else "" for v in r]
                        lines.append("| " + " | ".join(row_vals) + " |")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation Engines (Prevent Cartesian Product Duplicate Expansion)
# ---------------------------------------------------------------------------


def aggregate_attendance_by_course(attendance_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate granular attendance logs to (student_id, course_id) level without row explosion.

    Derives:
    - attendance_sessions: Total recorded attendance sessions
    - attendance_present: Count of 'Present' statuses
    - attendance_absent: Count of 'Absent' statuses
    - attendance_late: Count of 'Late' statuses
    - attendance_excused: Count of 'Excused' statuses
    - attendance_rate: Effective attendance percentage [0.0, 100.0]
    """
    if attendance_df.empty or "student_id" not in attendance_df.columns or "course_id" not in attendance_df.columns:
        return pd.DataFrame(columns=[
            "student_id", "course_id", "attendance_sessions", "attendance_present",
            "attendance_absent", "attendance_late", "attendance_excused", "attendance_rate",
        ])

    df = attendance_df.copy()
    status_col = "status" if "status" in df.columns else None

    # Group by student and course
    grouped = df.groupby(["student_id", "course_id"], as_index=False)

    agg_df = grouped.size().rename(columns={"size": "attendance_sessions"})

    if status_col:
        # Vectorized status counting
        df["_is_present"] = (df[status_col].astype(str).str.lower() == "present").astype(int)
        df["_is_absent"] = (df[status_col].astype(str).str.lower() == "absent").astype(int)
        df["_is_late"] = (df[status_col].astype(str).str.lower() == "late").astype(int)
        df["_is_excused"] = (df[status_col].astype(str).str.lower() == "excused").astype(int)

        status_aggs = df.groupby(["student_id", "course_id"], as_index=False)[
            ["_is_present", "_is_absent", "_is_late", "_is_excused"]
        ].sum()

        status_aggs = status_aggs.rename(columns={
            "_is_present": "attendance_present",
            "_is_absent": "attendance_absent",
            "_is_late": "attendance_late",
            "_is_excused": "attendance_excused",
        })

        agg_df = agg_df.merge(status_aggs, on=["student_id", "course_id"], how="left")

        # Compute attendance rate percentage (late counts as present or half; default standard is present / (sessions - excused))
        effective_sessions = agg_df["attendance_sessions"] - agg_df["attendance_excused"]
        # Where effective sessions > 0
        rate = np.where(
            effective_sessions > 0,
            ((agg_df["attendance_present"] + 0.5 * agg_df["attendance_late"]) / effective_sessions) * 100.0,
            np.nan,
        )
        agg_df["attendance_rate"] = np.round(rate, 2)
    else:
        agg_df["attendance_present"] = 0
        agg_df["attendance_absent"] = 0
        agg_df["attendance_late"] = 0
        agg_df["attendance_excused"] = 0
        agg_df["attendance_rate"] = np.nan

    return agg_df


def aggregate_submissions_by_course(
    submissions_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join submissions with assignments and aggregate to (student_id, course_id) level.

    Derives:
    - assignments_submitted_count: Count of submitted assignments
    - assignments_avg_score: Mean score achieved on assignments
    - assignments_min_score: Minimum score
    - assignments_max_score: Maximum score
    - assignments_late_count: Count of late submissions
    """
    if submissions_df.empty or assignments_df.empty:
        return pd.DataFrame(columns=[
            "student_id", "course_id", "assignments_submitted_count",
            "assignments_avg_score", "assignments_min_score", "assignments_max_score",
            "assignments_late_count",
        ])

    # Join submissions with assignments on assignment_id to get course_id
    sub = submissions_df.copy()
    assign = assignments_df[["assignment_id", "course_id"]].drop_duplicates(subset=["assignment_id"])

    merged = sub.merge(assign, on="assignment_id", how="inner")
    if merged.empty or "student_id" not in merged.columns or "course_id" not in merged.columns:
        return pd.DataFrame(columns=[
            "student_id", "course_id", "assignments_submitted_count",
            "assignments_avg_score", "assignments_min_score", "assignments_max_score",
            "assignments_late_count",
        ])

    score_col = "score" if "score" in merged.columns else None
    merged["_score_num"] = pd.to_numeric(merged[score_col], errors="coerce") if score_col else np.nan

    is_late_col = "is_late" if "is_late" in merged.columns else None
    merged["_is_late_num"] = merged[is_late_col].astype(bool).astype(int) if is_late_col else 0

    grouped = merged.groupby(["student_id", "course_id"], as_index=False)

    agg_df = grouped.size().rename(columns={"size": "assignments_submitted_count"})

    score_stats = grouped["_score_num"].agg(
        assignments_avg_score="mean",
        assignments_min_score="min",
        assignments_max_score="max",
    )
    score_stats["assignments_avg_score"] = score_stats["assignments_avg_score"].round(2)

    late_stats = grouped["_is_late_num"].sum().rename(columns={"_is_late_num": "assignments_late_count"})

    agg_df = agg_df.merge(score_stats, on=["student_id", "course_id"], how="left")
    agg_df = agg_df.merge(late_stats, on=["student_id", "course_id"], how="left")

    return agg_df


def aggregate_exams_by_course(exams_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exam scores to (student_id, course_id) level.

    Derives:
    - exams_taken_count: Count of exam sittings
    - exams_avg_score: Mean exam score
    - exam_midterm_score: Midterm score (if available)
    - exam_final_score: Final exam score (if available)
    """
    if exams_df.empty or "student_id" not in exams_df.columns or "course_id" not in exams_df.columns:
        return pd.DataFrame(columns=[
            "student_id", "course_id", "exams_taken_count",
            "exams_avg_score", "exam_midterm_score", "exam_final_score",
        ])

    df = exams_df.copy()
    score_col = "score" if "score" in df.columns else None
    df["_score_num"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else np.nan

    grouped = df.groupby(["student_id", "course_id"], as_index=False)
    agg_df = grouped.size().rename(columns={"size": "exams_taken_count"})

    score_stats = grouped["_score_num"].mean().rename(columns={"_score_num": "exams_avg_score"})
    score_stats["exams_avg_score"] = score_stats["exams_avg_score"].round(2)
    agg_df = agg_df.merge(score_stats, on=["student_id", "course_id"], how="left")

    # Extract Midterm and Final if exam_type present
    type_col = "exam_type" if "exam_type" in df.columns else None
    if type_col:
        midterms = df[df[type_col].astype(str).str.lower().str.contains("midterm")].groupby(
            ["student_id", "course_id"]
        )["_score_num"].max().rename("exam_midterm_score").reset_index()

        finals = df[df[type_col].astype(str).str.lower().str.contains("final")].groupby(
            ["student_id", "course_id"]
        )["_score_num"].max().rename("exam_final_score").reset_index()

        agg_df = agg_df.merge(midterms, on=["student_id", "course_id"], how="left")
        agg_df = agg_df.merge(finals, on=["student_id", "course_id"], how="left")
    else:
        agg_df["exam_midterm_score"] = np.nan
        agg_df["exam_final_score"] = np.nan

    return agg_df


# ---------------------------------------------------------------------------
# Multi-Source Integration Pipeline
# ---------------------------------------------------------------------------


def integrate_academic_data(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    enrollments: pd.DataFrame,
    attendance: pd.DataFrame,
    assignments: pd.DataFrame,
    submissions: pd.DataFrame,
    exams: pd.DataFrame,
) -> Tuple[pd.DataFrame, MergeAuditReport]:
    """Integrate all 7 academic datasets into a unified analytical view without Cartesian explosions.

    Preserves unmatched and orphan records, tracking them explicitly in the audit report.

    Args:
        students: Student demographic master table.
        courses: Course catalog table.
        enrollments: Student-course enrollment relationships.
        attendance: Attendance event log.
        assignments: Coursework metadata.
        submissions: Student submissions and marks.
        exams: Exam assessment results.

    Returns:
        Tuple of:
          - Enriched student-course master DataFrame.
          - MergeAuditReport with full audit metrics, orphan records, and relationship diagnostics.
    """
    for name, df in [
        ("students", students),
        ("courses", courses),
        ("enrollments", enrollments),
        ("attendance", attendance),
        ("assignments", assignments),
        ("submissions", submissions),
        ("exams", exams),
    ]:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"Expected DataFrame for '{name}', got: {type(df).__name__}")

    audit = MergeAuditReport(
        input_counts={
            "students": len(students),
            "courses": len(courses),
            "enrollments": len(enrollments),
            "attendance": len(attendance),
            "assignments": len(assignments),
            "submissions": len(submissions),
            "exams": len(exams),
        }
    )

    # 1. Audit Unmatched Students (in students master with 0 enrollments)
    registered_student_ids = set(students["student_id"].dropna().unique()) if "student_id" in students.columns else set()
    enrolled_student_ids = set(enrollments["student_id"].dropna().unique()) if "student_id" in enrollments.columns else set()
    audit.unmatched_students = sorted(list(registered_student_ids - enrolled_student_ids))

    # Audit Unmatched Courses (in courses catalog with 0 enrollments)
    catalog_course_ids = set(courses["course_id"].dropna().unique()) if "course_id" in courses.columns else set()
    enrolled_course_ids = set(enrollments["course_id"].dropna().unique()) if "course_id" in enrollments.columns else set()
    audit.unmatched_courses = sorted(list(catalog_course_ids - enrolled_course_ids))

    # 2. Audit Orphan Activity Records (Activity referencing unregistered students or courses)
    if "student_id" in attendance.columns:
        att_students = set(attendance["student_id"].dropna().unique())
        orphan_att = att_students - registered_student_ids
        audit.orphan_attendance_students = sorted(list(orphan_att))
        if orphan_att:
            audit.orphan_records["orphan_attendance"] = attendance[attendance["student_id"].isin(orphan_att)].copy()

    if "student_id" in submissions.columns:
        sub_students = set(submissions["student_id"].dropna().unique())
        orphan_sub = sub_students - registered_student_ids
        audit.orphan_submission_students = sorted(list(orphan_sub))
        if orphan_sub:
            audit.orphan_records["orphan_submissions"] = submissions[submissions["student_id"].isin(orphan_sub)].copy()

    if "student_id" in exams.columns:
        exam_students = set(exams["student_id"].dropna().unique())
        orphan_ex = exam_students - registered_student_ids
        audit.orphan_exam_students = sorted(list(orphan_ex))
        if orphan_ex:
            audit.orphan_records["orphan_exams"] = exams[exams["student_id"].isin(orphan_ex)].copy()

    # 3. Base Enrollment Backbone: Outer join students with enrollments and courses
    # If enrollments is empty, base on students directly so no students are hidden
    if enrollments.empty:
        base_df = students.copy()
        base_df["course_id"] = pd.NA
        base_df["enrollment_status"] = "No Enrollments"
    else:
        # Join enrollments with students
        base_df = enrollments.merge(students, on="student_id", how="left")
        # Join with courses
        base_df = base_df.merge(courses, on="course_id", how="left")
        base_df["enrollment_status"] = "Enrolled"

        # Preserve students with 0 enrollments (do not hide them!)
        if audit.unmatched_students:
            unmatched_students_df = students[students["student_id"].isin(audit.unmatched_students)].copy()
            unmatched_students_df["course_id"] = pd.NA
            unmatched_students_df["enrollment_status"] = "Unenrolled"
            base_df = pd.concat([base_df, unmatched_students_df], ignore_index=True)

    expected_base_count = len(enrollments) + len(audit.unmatched_students)
    if len(base_df) > expected_base_count:
        audit.duplicate_expansion_detected = True
        audit.expansion_details["base_enrollments"] = {
            "expected_max": expected_base_count,
            "actual": len(base_df),
        }

    # 4. Aggregations: Course-level summary metrics
    att_agg = aggregate_attendance_by_course(attendance)
    sub_agg = aggregate_submissions_by_course(submissions, assignments)
    exam_agg = aggregate_exams_by_course(exams)

    # 5. Enrich Base Backbone with Aggregations (1-to-1 merge on student_id, course_id)
    # Check row counts before and after join to confirm NO Cartesian product duplicate expansion
    pre_merge_len = len(base_df)

    integrated = base_df.merge(att_agg, on=["student_id", "course_id"], how="left")
    if len(integrated) != pre_merge_len:
        audit.duplicate_expansion_detected = True
        audit.expansion_details["attendance_join"] = {"before": pre_merge_len, "after": len(integrated)}

    integrated = integrated.merge(sub_agg, on=["student_id", "course_id"], how="left")
    if len(integrated) != pre_merge_len:
        audit.duplicate_expansion_detected = True
        audit.expansion_details["submissions_join"] = {"before": pre_merge_len, "after": len(integrated)}

    integrated = integrated.merge(exam_agg, on=["student_id", "course_id"], how="left")
    if len(integrated) != pre_merge_len:
        audit.duplicate_expansion_detected = True
        audit.expansion_details["exams_join"] = {"before": pre_merge_len, "after": len(integrated)}

    # 6. Audit Missing Activity for Enrolled Students
    enrolled_mask = integrated["enrollment_status"] == "Enrolled"
    enrolled_df = integrated[enrolled_mask]

    if "attendance_sessions" in enrolled_df.columns:
        no_att = enrolled_df[enrolled_df["attendance_sessions"].isna() | (enrolled_df["attendance_sessions"] == 0)]
        audit.students_with_no_attendance = sorted(list(no_att["student_id"].unique()))

    if "assignments_submitted_count" in enrolled_df.columns:
        no_sub = enrolled_df[enrolled_df["assignments_submitted_count"].isna() | (enrolled_df["assignments_submitted_count"] == 0)]
        audit.students_with_no_submissions = sorted(list(no_sub["student_id"].unique()))

    if "exams_taken_count" in enrolled_df.columns:
        no_ex = enrolled_df[enrolled_df["exams_taken_count"].isna() | (enrolled_df["exams_taken_count"] == 0)]
        audit.students_with_no_exams = sorted(list(no_ex["student_id"].unique()))

    # Store final output counts
    audit.output_counts["integrated_student_course"] = len(integrated)
    audit.output_counts["enrolled_records"] = int(enrolled_mask.sum())
    audit.output_counts["unenrolled_students"] = int((~enrolled_mask).sum())
    audit.output_counts["attendance_summary"] = len(att_agg)
    audit.output_counts["submissions_summary"] = len(sub_agg)
    audit.output_counts["exams_summary"] = len(exam_agg)

    logger.info(
        "Multi-source merge complete: %d integrated student-course rows (Enrolled: %d, Unenrolled: %d). "
        "Duplicate expansion: %s, Orphans detected: %s",
        len(integrated),
        audit.output_counts["enrolled_records"],
        audit.output_counts["unenrolled_students"],
        audit.duplicate_expansion_detected,
        audit.has_orphans,
    )

    return integrated, audit
