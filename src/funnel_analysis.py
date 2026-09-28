"""Academic Funnel Analysis Module.

Creates an academic progression funnel:
Enrolled → Active → Assignment Participation → Assessment Participation → Consistent Engagement

Calculates student counts and drop-off rates at each stage to identify engagement gaps.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.funnel_analysis")


# Funnel Stage Definitions
FUNNEL_STAGES = [
    "enrolled",
    "active",
    "assignment_participation",
    "assessment_participation",
    "consistent_engagement",
]


@dataclass
class FunnelStage:
    """Data for a single funnel stage."""

    stage_name: str
    student_count: int
    percentage_of_enrolled: float
    percentage_of_previous: Optional[float]  # None for first stage
    drop_off_from_previous: Optional[float]  # None for first stage
    criteria_description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class FunnelAnalysisResult:
    """Complete funnel analysis result."""

    total_enrolled: int
    stages: List[FunnelStage]
    overall_drop_off_rate: float
    largest_drop_off_stage: Optional[str]
    largest_drop_off_rate: float
    engagement_gaps: List[Dict[str, Any]]
    analysis_date: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "total_enrolled": self.total_enrolled,
            "stages": [stage.to_dict() for stage in self.stages],
            "overall_drop_off_rate": self.overall_drop_off_rate,
            "largest_drop_off_stage": self.largest_drop_off_stage,
            "largest_drop_off_rate": self.largest_drop_off_rate,
            "engagement_gaps": self.engagement_gaps,
            "analysis_date": self.analysis_date,
        }


def count_enrolled_students(
    enrollments_df: pd.DataFrame,
    student_id_column: str = "student_id",
) -> int:
    """Count total enrolled students.

    Args:
        enrollments_df: DataFrame with enrollment records.
        student_id_column: Name of the student ID column.

    Returns:
        Count of enrolled students.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if student_id_column not in enrollments_df.columns:
        raise DataValidationError(f"enrollments_df must contain '{student_id_column}' column")

    return enrollments_df[student_id_column].nunique()


def count_active_students(
    attendance_df: pd.DataFrame,
    student_id_column: str = "student_id",
    min_attendance_records: int = 1,
) -> int:
    """Count active students (students with at least one attendance record).

    Args:
        attendance_df: DataFrame with attendance records.
        student_id_column: Name of the student ID column.
        min_attendance_records: Minimum attendance records to be considered active.

    Returns:
        Count of active students.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if student_id_column not in attendance_df.columns:
        raise DataValidationError(f"attendance_df must contain '{student_id_column}' column")

    attendance_counts = attendance_df[student_id_column].value_counts()
    active_students = attendance_counts[attendance_counts >= min_attendance_records]
    return len(active_students)


def count_assignment_participants(
    submissions_df: pd.DataFrame,
    student_id_column: str = "student_id",
    min_submissions: int = 1,
) -> int:
    """Count students who participated in assignments.

    Args:
        submissions_df: DataFrame with submission records.
        student_id_column: Name of the student ID column.
        min_submissions: Minimum submissions to be considered participant.

    Returns:
        Count of assignment participants.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if student_id_column not in submissions_df.columns:
        raise DataValidationError(f"submissions_df must contain '{student_id_column}' column")

    submission_counts = submissions_df[student_id_column].value_counts()
    participants = submission_counts[submission_counts >= min_submissions]
    return len(participants)


def count_assessment_participants(
    exams_df: pd.DataFrame,
    student_id_column: str = "student_id",
    min_exams: int = 1,
) -> int:
    """Count students who participated in assessments.

    Args:
        exams_df: DataFrame with exam records.
        student_id_column: Name of the student ID column.
        min_exams: Minimum exams to be considered participant.

    Returns:
        Count of assessment participants.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if student_id_column not in exams_df.columns:
        raise DataValidationError(f"exams_df must contain '{student_id_column}' column")

    exam_counts = exams_df[student_id_column].value_counts()
    participants = exam_counts[exam_counts >= min_exams]
    return len(participants)


def count_consistently_engaged(
    student_features_df: pd.DataFrame,
    student_id_column: str = "student_id",
    attendance_threshold: float = 70.0,
    completion_threshold: float = 70.0,
) -> int:
    """Count consistently engaged students.

    Consistently engaged: students with attendance ≥70% and completion ≥70%.

    Args:
        student_features_df: DataFrame with student features.
        student_id_column: Name of the student ID column.
        attendance_threshold: Minimum attendance percentage.
        completion_threshold: Minimum completion rate.

    Returns:
        Count of consistently engaged students.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if student_id_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{student_id_column}' column")

    if "attendance_percentage" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'attendance_percentage' column")

    if "assignment_completion_rate" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'assignment_completion_rate' column")

    # Filter students meeting both thresholds
    consistent = student_features_df[
        (student_features_df["attendance_percentage"] >= attendance_threshold) &
        (student_features_df["assignment_completion_rate"] >= completion_threshold)
    ]

    return consistent[student_id_column].nunique()


def calculate_drop_off_rate(
    current_count: int,
    previous_count: int,
) -> float:
    """Calculate drop-off rate between stages.

    Args:
        current_count: Count at current stage.
        previous_count: Count at previous stage.

    Returns:
        Drop-off rate as percentage (0.0 to 100.0).
    """
    if previous_count == 0:
        return 0.0

    drop_off = ((previous_count - current_count) / previous_count) * 100.0
    return round(drop_off, 2)


def identify_engagement_gaps(
    stages: List[FunnelStage],
    drop_off_threshold: float = 20.0,
) -> List[Dict[str, Any]]:
    """Identify stages with significant engagement gaps.

    Args:
        stages: List of funnel stages.
        drop_off_threshold: Drop-off rate threshold for gap identification.

    Returns:
        List of engagement gaps with details.
    """
    gaps: List[Dict[str, Any]] = []

    for i, stage in enumerate(stages[1:], start=1):  # Skip first stage (no previous)
        if stage.drop_off_from_previous is not None and stage.drop_off_from_previous >= drop_off_threshold:
            gaps.append({
                "stage": stage.stage_name,
                "previous_stage": stages[i - 1].stage_name,
                "drop_off_rate": stage.drop_off_from_previous,
                "student_count": stage.student_count,
                "previous_count": stages[i - 1].student_count,
                "severity": "high" if stage.drop_off_from_previous >= 40.0 else "moderate",
            })

    return gaps


def analyze_academic_funnel(
    enrollments_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    exams_df: pd.DataFrame,
    student_features_df: pd.DataFrame,
    student_id_column: str = "student_id",
    min_attendance_records: int = 1,
    min_submissions: int = 1,
    min_exams: int = 1,
    attendance_threshold: float = 70.0,
    completion_threshold: float = 70.0,
) -> FunnelAnalysisResult:
    """Perform complete academic funnel analysis.

    Args:
        enrollments_df: DataFrame with enrollment records.
        attendance_df: DataFrame with attendance records.
        submissions_df: DataFrame with submission records.
        exams_df: DataFrame with exam records.
        student_features_df: DataFrame with student features.
        student_id_column: Name of the student ID column.
        min_attendance_records: Minimum attendance records for active status.
        min_submissions: Minimum submissions for participation.
        min_exams: Minimum exams for participation.
        attendance_threshold: Attendance threshold for consistent engagement.
        completion_threshold: Completion threshold for consistent engagement.

    Returns:
        FunnelAnalysisResult with complete funnel analysis.

    Raises:
        DataValidationError: If required columns are missing.
    """
    # Count students at each stage
    enrolled_count = count_enrolled_students(enrollments_df, student_id_column)
    active_count = count_active_students(attendance_df, student_id_column, min_attendance_records)
    assignment_count = count_assignment_participants(submissions_df, student_id_column, min_submissions)
    assessment_count = count_assessment_participants(exams_df, student_id_column, min_exams)
    consistent_count = count_consistently_engaged(
        student_features_df,
        student_id_column,
        attendance_threshold,
        completion_threshold,
    )

    # Create stage data
    stages: List[FunnelStage] = []

    # Stage 1: Enrolled
    stages.append(
        FunnelStage(
            stage_name="enrolled",
            student_count=enrolled_count,
            percentage_of_enrolled=100.0,
            percentage_of_previous=None,
            drop_off_from_previous=None,
            criteria_description="All enrolled students",
        )
    )

    # Stage 2: Active
    if enrolled_count > 0:
        active_percentage = (active_count / enrolled_count) * 100.0
        active_drop_off = calculate_drop_off_rate(active_count, enrolled_count)
    else:
        active_percentage = 0.0
        active_drop_off = 0.0

    stages.append(
        FunnelStage(
            stage_name="active",
            student_count=active_count,
            percentage_of_enrolled=round(active_percentage, 2),
            percentage_of_previous=round(active_percentage, 2),
            drop_off_from_previous=active_drop_off,
            criteria_description=f"Students with ≥{min_attendance_records} attendance record(s)",
        )
    )

    # Stage 3: Assignment Participation
    if enrolled_count > 0:
        assignment_percentage = (assignment_count / enrolled_count) * 100.0
        assignment_drop_off = calculate_drop_off_rate(assignment_count, active_count)
    else:
        assignment_percentage = 0.0
        assignment_drop_off = 0.0

    stages.append(
        FunnelStage(
            stage_name="assignment_participation",
            student_count=assignment_count,
            percentage_of_enrolled=round(assignment_percentage, 2),
            percentage_of_previous=round((assignment_count / active_count) * 100.0, 2) if active_count > 0 else 0.0,
            drop_off_from_previous=assignment_drop_off,
            criteria_description=f"Students with ≥{min_submissions} assignment submission(s)",
        )
    )

    # Stage 4: Assessment Participation
    if enrolled_count > 0:
        assessment_percentage = (assessment_count / enrolled_count) * 100.0
        assessment_drop_off = calculate_drop_off_rate(assessment_count, assignment_count)
    else:
        assessment_percentage = 0.0
        assessment_drop_off = 0.0

    stages.append(
        FunnelStage(
            stage_name="assessment_participation",
            student_count=assessment_count,
            percentage_of_enrolled=round(assessment_percentage, 2),
            percentage_of_previous=round((assessment_count / assignment_count) * 100.0, 2) if assignment_count > 0 else 0.0,
            drop_off_from_previous=assessment_drop_off,
            criteria_description=f"Students with ≥{min_exams} exam attempt(s)",
        )
    )

    # Stage 5: Consistent Engagement
    if enrolled_count > 0:
        consistent_percentage = (consistent_count / enrolled_count) * 100.0
        consistent_drop_off = calculate_drop_off_rate(consistent_count, assessment_count)
    else:
        consistent_percentage = 0.0
        consistent_drop_off = 0.0

    stages.append(
        FunnelStage(
            stage_name="consistent_engagement",
            student_count=consistent_count,
            percentage_of_enrolled=round(consistent_percentage, 2),
            percentage_of_previous=round((consistent_count / assessment_count) * 100.0, 2) if assessment_count > 0 else 0.0,
            drop_off_from_previous=consistent_drop_off,
            criteria_description=f"Students with attendance ≥{attendance_threshold}% and completion ≥{completion_threshold}%",
        )
    )

    # Calculate overall metrics
    overall_drop_off = calculate_drop_off_rate(consistent_count, enrolled_count)

    # Find largest drop-off
    largest_drop_off_stage = None
    largest_drop_off_rate = 0.0

    for stage in stages[1:]:  # Skip first stage
        if stage.drop_off_from_previous is not None and stage.drop_off_from_previous > largest_drop_off_rate:
            largest_drop_off_rate = stage.drop_off_from_previous
            largest_drop_off_stage = stage.stage_name

    # Identify engagement gaps
    engagement_gaps = identify_engagement_gaps(stages)

    logger.info(
        "Funnel analysis: %d enrolled → %d consistently engaged (%.1f%% overall drop-off)",
        enrolled_count,
        consistent_count,
        overall_drop_off,
    )

    return FunnelAnalysisResult(
        total_enrolled=enrolled_count,
        stages=stages,
        overall_drop_off_rate=overall_drop_off,
        largest_drop_off_stage=largest_drop_off_stage,
        largest_drop_off_rate=largest_drop_off_rate,
        engagement_gaps=engagement_gaps,
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def generate_funnel_report_markdown(
    result: FunnelAnalysisResult,
) -> str:
    """Generate a markdown summary of funnel analysis.

    Args:
        result: FunnelAnalysisResult from analysis.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic Engagement Funnel Analysis",
        "",
        f"**Analysis Date**: {result.analysis_date}",
        f"**Total Enrolled**: {result.total_enrolled:,}",
        f"**Overall Drop-off Rate**: {result.overall_drop_off_rate:.2f}%",
        "",
        "## Funnel Stages",
        "",
    ]

    for stage in result.stages:
        lines.append(f"### {stage.stage_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("**Stage Metrics**")
        lines.append(f"- **Student Count**: {stage.student_count:,}")
        lines.append(f"- **% of Enrolled**: {stage.percentage_of_enrolled}%")
        if stage.percentage_of_previous is not None:
            lines.append(f"- **% of Previous Stage**: {stage.percentage_of_previous}%")
        if stage.drop_off_from_previous is not None:
            lines.append(f"- **Drop-off from Previous**: {stage.drop_off_from_previous}%")
        lines.append("")
        lines.append(f"**Criteria**: {stage.criteria_description}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Key Insights")
    lines.append("")
    lines.append(f"**Largest Drop-off**: {result.largest_drop_off_stage.replace('_', ' ').title() if result.largest_drop_off_stage else 'N/A'}")
    lines.append(f"**Largest Drop-off Rate**: {result.largest_drop_off_rate:.2f}%")
    lines.append("")

    if result.engagement_gaps:
        lines.append("## Engagement Gaps")
        lines.append("")
        lines.append("Stages with significant drop-off rates:")
        lines.append("")

        for gap in result.engagement_gaps:
            lines.append(f"### {gap['stage'].replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **From**: {gap['previous_stage'].replace('_', ' ').title()}")
            lines.append(f"- **Drop-off Rate**: {gap['drop_off_rate']:.2f}%")
            lines.append(f"- **Students Lost**: {gap['previous_count']:,} → {gap['student_count']:,}")
            lines.append(f"- **Severity**: {gap['severity'].title()}")
            lines.append("")
    else:
        lines.append("## Engagement Gaps")
        lines.append("")
        lines.append("No significant engagement gaps detected.")
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")

    if result.engagement_gaps:
        lines.append("Based on identified engagement gaps:")
        lines.append("")
        for gap in result.engagement_gaps:
            stage = gap['stage'].replace('_', ' ').title()
            if gap['severity'] == 'high':
                lines.append(f"- **{stage}**: High priority intervention needed - {gap['drop_off_rate']:.1f}% drop-off")
            else:
                lines.append(f"- **{stage}**: Monitor and investigate - {gap['drop_off_rate']:.1f}% drop-off")
        lines.append("")
    else:
        lines.append("Funnel progression is healthy. Continue monitoring for early signs of engagement decline.")
        lines.append("")

    return "\n".join(lines)


def get_funnel_stage_definitions() -> Dict[str, str]:
    """Get definitions of all funnel stages.

    Returns:
        Dictionary mapping stage names to descriptions.
    """
    return {
        "enrolled": "All students enrolled in the course/program",
        "active": "Students with at least one attendance record",
        "assignment_participation": "Students who have submitted at least one assignment",
        "assessment_participation": "Students who have attempted at least one exam",
        "consistent_engagement": "Students with consistent attendance and assignment completion",
    }
