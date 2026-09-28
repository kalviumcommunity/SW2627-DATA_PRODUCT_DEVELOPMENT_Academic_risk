"""Academic GroupBy Segmentation Module.

Provides reusable functions for segmenting students using Pandas GroupBy:
- program
- year
- course
- risk level
- engagement segment

Calculates aggregated metrics for each segment:
- average attendance
- assignment completion
- average exam score
- students requiring review
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.groupby_segmentation")


@dataclass
class SegmentMetrics:
    """Aggregated metrics for a student segment."""

    segment_name: str
    segment_value: Any
    student_count: int
    avg_attendance: Optional[float]
    avg_completion: Optional[float]
    avg_exam_score: Optional[float]
    students_requiring_review: int
    review_percentage: float
    missing_attendance_count: int
    missing_completion_count: int
    missing_exam_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class SegmentationResult:
    """Complete segmentation analysis result."""

    segment_by: str
    total_students: int
    segments: List[SegmentMetrics]
    segment_count: int
    analysis_date: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "segment_by": self.segment_by,
            "total_students": self.total_students,
            "segments": [seg.to_dict() for seg in self.segments],
            "segment_count": self.segment_count,
            "analysis_date": self.analysis_date,
        }


def calculate_segment_metrics(
    group_df: pd.DataFrame,
    segment_name: str,
    segment_value: Any,
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> SegmentMetrics:
    """Calculate aggregated metrics for a student segment.

    Args:
        group_df: DataFrame for the segment group.
        segment_name: Name of the segmentation dimension.
        segment_value: Value of this segment.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        SegmentMetrics with calculated metrics.
    """
    student_count = len(group_df)

    # Average attendance
    avg_attendance = None
    missing_attendance_count = 0
    if "attendance_percentage" in group_df.columns:
        valid_attendance = group_df["attendance_percentage"].dropna()
        missing_attendance_count = int(group_df["attendance_percentage"].isna().sum())
        if not valid_attendance.empty:
            avg_attendance = round(float(valid_attendance.mean()), 2)

    # Average completion
    avg_completion = None
    missing_completion_count = 0
    if "assignment_completion_rate" in group_df.columns:
        valid_completion = group_df["assignment_completion_rate"].dropna()
        missing_completion_count = int(group_df["assignment_completion_rate"].isna().sum())
        if not valid_completion.empty:
            avg_completion = round(float(valid_completion.mean()), 2)

    # Average exam score
    avg_exam_score = None
    missing_exam_count = 0
    if "average_exam_score" in group_df.columns:
        valid_exam = group_df["average_exam_score"].dropna()
        missing_exam_count = int(group_df["average_exam_score"].isna().sum())
        if not valid_exam.empty:
            avg_exam_score = round(float(valid_exam.mean()), 2)

    # Students requiring review (flagged if any metric below threshold)
    students_requiring_review = 0
    for _, row in group_df.iterrows():
        needs_review = False
        if avg_attendance is not None and row.get("attendance_percentage") is not None:
            if row["attendance_percentage"] < review_threshold_attendance:
                needs_review = True
        if avg_completion is not None and row.get("assignment_completion_rate") is not None:
            if row["assignment_completion_rate"] < review_threshold_completion:
                needs_review = True
        if avg_exam_score is not None and row.get("average_exam_score") is not None:
            if row["average_exam_score"] < review_threshold_exam:
                needs_review = True
        if needs_review:
            students_requiring_review += 1

    review_percentage = round((students_requiring_review / student_count) * 100.0, 2) if student_count > 0 else 0.0

    return SegmentMetrics(
        segment_name=segment_name,
        segment_value=segment_value,
        student_count=student_count,
        avg_attendance=avg_attendance,
        avg_completion=avg_completion,
        avg_exam_score=avg_exam_score,
        students_requiring_review=students_requiring_review,
        review_percentage=review_percentage,
        missing_attendance_count=missing_attendance_count,
        missing_completion_count=missing_completion_count,
        missing_exam_count=missing_exam_count,
    )


def segment_by_program(
    student_features_df: pd.DataFrame,
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> SegmentationResult:
    """Segment students by academic program.

    Args:
        student_features_df: DataFrame with student features including program column.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        SegmentationResult with program-based segments.

    Raises:
        DataValidationError: If program column is missing.
    """
    if "program" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'program' column")

    total_students = len(student_features_df)
    segments: List[SegmentMetrics] = []

    for program_value, group_df in student_features_df.groupby("program"):
        metrics = calculate_segment_metrics(
            group_df,
            segment_name="program",
            segment_value=program_value,
            review_threshold_attendance=review_threshold_attendance,
            review_threshold_completion=review_threshold_completion,
            review_threshold_exam=review_threshold_exam,
        )
        segments.append(metrics)

    logger.info("Segmented %d students by %d programs", total_students, len(segments))

    return SegmentationResult(
        segment_by="program",
        total_students=total_students,
        segments=segments,
        segment_count=len(segments),
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def segment_by_year(
    student_features_df: pd.DataFrame,
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> SegmentationResult:
    """Segment students by academic year.

    Args:
        student_features_df: DataFrame with student features including year column.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        SegmentationResult with year-based segments.

    Raises:
        DataValidationError: If year column is missing.
    """
    if "year" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'year' column")

    total_students = len(student_features_df)
    segments: List[SegmentMetrics] = []

    for year_value, group_df in student_features_df.groupby("year"):
        metrics = calculate_segment_metrics(
            group_df,
            segment_name="year",
            segment_value=year_value,
            review_threshold_attendance=review_threshold_attendance,
            review_threshold_completion=review_threshold_completion,
            review_threshold_exam=review_threshold_exam,
        )
        segments.append(metrics)

    logger.info("Segmented %d students by %d years", total_students, len(segments))

    return SegmentationResult(
        segment_by="year",
        total_students=total_students,
        segments=segments,
        segment_count=len(segments),
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def segment_by_course(
    student_features_df: pd.DataFrame,
    enrollments_df: pd.DataFrame,
    courses_df: Optional[pd.DataFrame] = None,
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> SegmentationResult:
    """Segment students by course enrollment.

    Args:
        student_features_df: DataFrame with student features.
        enrollments_df: DataFrame with student-course enrollments.
        courses_df: Optional DataFrame with course metadata.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        SegmentationResult with course-based segments.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if "student_id" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'student_id' column")
    if "student_id" not in enrollments_df.columns or "course_id" not in enrollments_df.columns:
        raise DataValidationError("enrollments_df must contain 'student_id' and 'course_id' columns")

    # Merge students with enrollments
    merged = student_features_df.merge(enrollments_df, on="student_id", how="inner")

    # Add course names if available
    if courses_df is not None and "course_id" in courses_df.columns:
        if "course_name" in courses_df.columns:
            merged = merged.merge(courses_df[["course_id", "course_name"]], on="course_id", how="left")
            segment_col = "course_name"
        else:
            segment_col = "course_id"
    else:
        segment_col = "course_id"

    total_students = len(student_features_df)
    segments: List[SegmentMetrics] = []

    for course_value, group_df in merged.groupby(segment_col):
        metrics = calculate_segment_metrics(
            group_df,
            segment_name="course",
            segment_value=course_value,
            review_threshold_attendance=review_threshold_attendance,
            review_threshold_completion=review_threshold_completion,
            review_threshold_exam=review_threshold_exam,
        )
        segments.append(metrics)

    logger.info("Segmented %d students by %d courses", total_students, len(segments))

    return SegmentationResult(
        segment_by="course",
        total_students=total_students,
        segments=segments,
        segment_count=len(segments),
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def segment_by_risk_level(
    student_features_df: pd.DataFrame,
    risk_column: str = "risk_level",
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> SegmentationResult:
    """Segment students by risk level.

    Args:
        student_features_df: DataFrame with student features including risk column.
        risk_column: Name of the risk level column.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        SegmentationResult with risk-based segments.

    Raises:
        DataValidationError: If risk column is missing.
    """
    if risk_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{risk_column}' column")

    total_students = len(student_features_df)
    segments: List[SegmentMetrics] = []

    for risk_value, group_df in student_features_df.groupby(risk_column):
        metrics = calculate_segment_metrics(
            group_df,
            segment_name="risk_level",
            segment_value=risk_value,
            review_threshold_attendance=review_threshold_attendance,
            review_threshold_completion=review_threshold_completion,
            review_threshold_exam=review_threshold_exam,
        )
        segments.append(metrics)

    logger.info("Segmented %d students by %d risk levels", total_students, len(segments))

    return SegmentationResult(
        segment_by="risk_level",
        total_students=total_students,
        segments=segments,
        segment_count=len(segments),
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def segment_by_engagement(
    student_features_df: pd.DataFrame,
    engagement_column: str = "engagement_segment",
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> SegmentationResult:
    """Segment students by engagement segment.

    Args:
        student_features_df: DataFrame with student features including engagement column.
        engagement_column: Name of the engagement segment column.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        SegmentationResult with engagement-based segments.

    Raises:
        DataValidationError: If engagement column is missing.
    """
    if engagement_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{engagement_column}' column")

    total_students = len(student_features_df)
    segments: List[SegmentMetrics] = []

    for engagement_value, group_df in student_features_df.groupby(engagement_column):
        metrics = calculate_segment_metrics(
            group_df,
            segment_name="engagement_segment",
            segment_value=engagement_value,
            review_threshold_attendance=review_threshold_attendance,
            review_threshold_completion=review_threshold_completion,
            review_threshold_exam=review_threshold_exam,
        )
        segments.append(metrics)

    logger.info("Segmented %d students by %d engagement segments", total_students, len(segments))

    return SegmentationResult(
        segment_by="engagement_segment",
        total_students=total_students,
        segments=segments,
        segment_count=len(segments),
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def create_engagement_segments(
    student_features_df: pd.DataFrame,
    attendance_threshold: float = 75.0,
    completion_threshold: float = 75.0,
) -> pd.DataFrame:
    """Create engagement segments based on attendance and completion.

    Args:
        student_features_df: DataFrame with student features.
        attendance_threshold: Threshold for high attendance.
        completion_threshold: Threshold for high completion.

    Returns:
        DataFrame with added engagement_segment column.
    """
    df = student_features_df.copy()

    def classify_engagement(row: pd.Series) -> str:
        att = row.get("attendance_percentage")
        comp = row.get("assignment_completion_rate")

        if pd.isna(att) or pd.isna(comp):
            return "Unknown"

        if att >= attendance_threshold and comp >= completion_threshold:
            return "Highly Engaged"
        elif att >= attendance_threshold and comp < completion_threshold:
            return "Attender (Low Completion)"
        elif att < attendance_threshold and comp >= completion_threshold:
            return "Completer (Low Attendance)"
        else:
            return "Disengaged"

    df["engagement_segment"] = df.apply(classify_engagement, axis=1)

    logger.info("Created engagement segments for %d students", len(df))

    return df


def multi_level_segmentation(
    student_features_df: pd.DataFrame,
    primary_segment: str = "program",
    secondary_segment: str = "year",
    review_threshold_attendance: float = 70.0,
    review_threshold_completion: float = 70.0,
    review_threshold_exam: float = 50.0,
) -> Dict[str, SegmentationResult]:
    """Perform multi-level segmentation (e.g., program × year combination).

    Args:
        student_features_df: DataFrame with student features.
        primary_segment: Primary segmentation column.
        secondary_segment: Secondary segmentation column.
        review_threshold_attendance: Attendance threshold for review flag.
        review_threshold_completion: Completion threshold for review flag.
        review_threshold_exam: Exam score threshold for review flag.

    Returns:
        Dictionary mapping segment combinations to SegmentationResult.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if primary_segment not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{primary_segment}' column")
    if secondary_segment not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{secondary_segment}' column")

    results: Dict[str, SegmentationResult] = {}

    for primary_value, primary_group in student_features_df.groupby(primary_segment):
        segments: List[SegmentMetrics] = []

        for secondary_value, group_df in primary_group.groupby(secondary_segment):
            metrics = calculate_segment_metrics(
                group_df,
                segment_name=f"{primary_segment} × {secondary_segment}",
                segment_value=f"{primary_value} - {secondary_value}",
                review_threshold_attendance=review_threshold_attendance,
                review_threshold_completion=review_threshold_completion,
                review_threshold_exam=review_threshold_exam,
            )
            segments.append(metrics)

        result = SegmentationResult(
            segment_by=f"{primary_segment} × {secondary_segment}",
            total_students=len(primary_group),
            segments=segments,
            segment_count=len(segments),
            analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        results[str(primary_value)] = result

    logger.info(
        "Multi-level segmentation: %s × %s with %d primary groups",
        primary_segment,
        secondary_segment,
        len(results),
    )

    return results


def generate_segmentation_markdown(
    result: SegmentationResult,
) -> str:
    """Generate a markdown summary of segmentation analysis.

    Args:
        result: SegmentationResult from analysis.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic Segmentation Analysis",
        "",
        f"**Segment By**: {result.segment_by}",
        f"**Total Students**: {result.total_students:,}",
        f"**Number of Segments**: {result.segment_count}",
        f"**Analysis Date**: {result.analysis_date}",
        "",
        "## Segment Details",
        "",
    ]

    # Sort segments by student count (descending)
    sorted_segments = sorted(result.segments, key=lambda x: x.student_count, reverse=True)

    for seg in sorted_segments:
        lines.append(f"### {seg.segment_value}")
        lines.append("")
        lines.append("**Student Count**")
        lines.append(f"- {seg.student_count:,} students")
        lines.append("")

        lines.append("**Average Metrics**")
        if seg.avg_attendance is not None:
            lines.append(f"- **Attendance**: {seg.avg_attendance}%")
        else:
            lines.append(f"- **Attendance**: N/A ({seg.missing_attendance_count} missing)")
        
        if seg.avg_completion is not None:
            lines.append(f"- **Completion Rate**: {seg.avg_completion}%")
        else:
            lines.append(f"- **Completion Rate**: N/A ({seg.missing_completion_count} missing)")
        
        if seg.avg_exam_score is not None:
            lines.append(f"- **Exam Score**: {seg.avg_exam_score}")
        else:
            lines.append(f"- **Exam Score**: N/A ({seg.missing_exam_count} missing)")
        
        lines.append("")

        lines.append("**Review Status**")
        lines.append(f"- **Students Requiring Review**: {seg.students_requiring_review:,} ({seg.review_percentage}%)")
        lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def compare_segments(
    result: SegmentationResult,
    metric: str = "avg_attendance",
) -> pd.DataFrame:
    """Compare segments on a specific metric.

    Args:
        result: SegmentationResult from analysis.
        metric: Metric to compare (avg_attendance, avg_completion, avg_exam_score, review_percentage).

    Returns:
        DataFrame with segment comparison.

    Raises:
        DataValidationError: If metric is invalid.
    """
    valid_metrics = ["avg_attendance", "avg_completion", "avg_exam_score", "review_percentage"]
    if metric not in valid_metrics:
        raise DataValidationError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

    comparison_data = []
    for seg in result.segments:
        comparison_data.append({
            "segment_value": seg.segment_value,
            "student_count": seg.student_count,
            metric: getattr(seg, metric),
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values(by=metric, ascending=False, na_position="last")

    return df
