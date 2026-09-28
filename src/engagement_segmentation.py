"""Student Engagement Segmentation Module.

Creates measurable student segments based on:
- attendance
- assignment completion
- exam performance
- recent trends

Segments are assigned based on objective criteria with evidence tracking.
No arbitrary labels - each segment has clear, measurable thresholds.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.engagement_segmentation")


# Segment Criteria Thresholds
# These are measurable, evidence-based thresholds
SEGMENT_CRITERIA = {
    "consistent": {
        "attendance_threshold": 75.0,
        "completion_threshold": 75.0,
        "exam_threshold": 60.0,
        "trend_stability_range": 10.0,  # Trend must be within ±10%
        "description": "Students with consistently good attendance, completion, and performance",
    },
    "strong_performance": {
        "attendance_threshold": 85.0,
        "completion_threshold": 85.0,
        "exam_threshold": 75.0,
        "description": "Students with excellent performance across all metrics",
    },
    "inconsistent": {
        "attendance_threshold": 60.0,
        "completion_threshold": 60.0,
        "exam_threshold": 50.0,
        "variance_threshold": 20.0,  # High variance across metrics
        "description": "Students with fluctuating performance across different areas",
    },
    "declining_engagement": {
        "attendance_trend_threshold": -10.0,
        "completion_trend_threshold": -10.0,
        "exam_trend_threshold": -10.0,
        "description": "Students showing negative trends in engagement metrics",
    },
}


@dataclass
class SegmentEvidence:
    """Evidence supporting a segment assignment."""

    student_id: Any
    assigned_segment: str
    evidence: List[str]
    metric_values: Dict[str, float]
    threshold_comparisons: Dict[str, Tuple[float, float]]  # (value, threshold)
    confidence: float  # 0.0 to 1.0 based on how many criteria are met

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class SegmentationResult:
    """Complete segmentation analysis result."""

    total_students: int
    segment_counts: Dict[str, int]
    segment_percentages: Dict[str, float]
    student_evidence: List[SegmentEvidence]
    unassigned_students: List[Any]
    analysis_date: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "total_students": self.total_students,
            "segment_counts": self.segment_counts,
            "segment_percentages": self.segment_percentages,
            "student_evidence": [e.to_dict() for e in self.student_evidence],
            "unassigned_students": self.unassigned_students,
            "analysis_date": self.analysis_date,
        }


def check_consistent_segment(
    row: pd.Series,
) -> Tuple[bool, List[str], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Check if student meets criteria for Consistent segment.

    Criteria:
    - Attendance >= 75%
    - Completion >= 75%
    - Exam score >= 60%
    - Trends stable (within ±10%)

    Args:
        row: Student feature row.

    Returns:
        Tuple of (is_segment, evidence, metric_values, threshold_comparisons).
    """
    criteria = SEGMENT_CRITERIA["consistent"]
    evidence: List[str] = []
    metric_values: Dict[str, float] = {}
    threshold_comparisons: Dict[str, Tuple[float, float]] = {}

    met_count = 0
    total_checks = 0

    # Check attendance
    if "attendance_percentage" in row and pd.notna(row["attendance_percentage"]):
        att = row["attendance_percentage"]
        att_thresh = criteria["attendance_threshold"]
        metric_values["attendance"] = att
        threshold_comparisons["attendance"] = (att, att_thresh)
        total_checks += 1

        if att >= att_thresh:
            evidence.append(f"Attendance {att:.1f}% meets threshold ({att_thresh}%)")
            met_count += 1
        else:
            evidence.append(f"Attendance {att:.1f}% below threshold ({att_thresh}%)")

    # Check completion
    if "assignment_completion_rate" in row and pd.notna(row["assignment_completion_rate"]):
        comp = row["assignment_completion_rate"]
        comp_thresh = criteria["completion_threshold"]
        metric_values["completion"] = comp
        threshold_comparisons["completion"] = (comp, comp_thresh)
        total_checks += 1

        if comp >= comp_thresh:
            evidence.append(f"Completion {comp:.1f}% meets threshold ({comp_thresh}%)")
            met_count += 1
        else:
            evidence.append(f"Completion {comp:.1f}% below threshold ({comp_thresh}%)")

    # Check exam performance
    if "average_exam_score" in row and pd.notna(row["average_exam_score"]):
        exam = row["average_exam_score"]
        exam_thresh = criteria["exam_threshold"]
        metric_values["exam_score"] = exam
        threshold_comparisons["exam_score"] = (exam, exam_thresh)
        total_checks += 1

        if exam >= exam_thresh:
            evidence.append(f"Exam score {exam:.1f} meets threshold ({exam_thresh})")
            met_count += 1
        else:
            evidence.append(f"Exam score {exam:.1f} below threshold ({exam_thresh})")

    # Check trend stability
    trend_stable = True
    for trend_col in ["attendance_trend", "assignment_trend", "exam_trend"]:
        if trend_col in row and pd.notna(row[trend_col]):
            trend = row[trend_col]
            stability_range = criteria["trend_stability_range"]
            metric_values[trend_col] = trend
            threshold_comparisons[trend_col] = (trend, stability_range)
            total_checks += 1

            if abs(trend) <= stability_range:
                evidence.append(f"{trend_col} {trend:.1f} within stable range (±{stability_range})")
                met_count += 1
            else:
                evidence.append(f"{trend_col} {trend:.1f} outside stable range (±{stability_range})")
                trend_stable = False

    # Must meet at least 75% of checked criteria
    is_segment = total_checks > 0 and (met_count / total_checks) >= 0.75

    return is_segment, evidence, metric_values, threshold_comparisons


def check_strong_performance_segment(
    row: pd.Series,
) -> Tuple[bool, List[str], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Check if student meets criteria for Strong Performance segment.

    Criteria:
    - Attendance >= 85%
    - Completion >= 85%
    - Exam score >= 75%

    Args:
        row: Student feature row.

    Returns:
        Tuple of (is_segment, evidence, metric_values, threshold_comparisons).
    """
    criteria = SEGMENT_CRITERIA["strong_performance"]
    evidence: List[str] = []
    metric_values: Dict[str, float] = {}
    threshold_comparisons: Dict[str, Tuple[float, float]] = {}

    met_count = 0
    total_checks = 0

    # Check attendance
    if "attendance_percentage" in row and pd.notna(row["attendance_percentage"]):
        att = row["attendance_percentage"]
        att_thresh = criteria["attendance_threshold"]
        metric_values["attendance"] = att
        threshold_comparisons["attendance"] = (att, att_thresh)
        total_checks += 1

        if att >= att_thresh:
            evidence.append(f"Attendance {att:.1f}% meets high threshold ({att_thresh}%)")
            met_count += 1
        else:
            evidence.append(f"Attendance {att:.1f}% below high threshold ({att_thresh}%)")

    # Check completion
    if "assignment_completion_rate" in row and pd.notna(row["assignment_completion_rate"]):
        comp = row["assignment_completion_rate"]
        comp_thresh = criteria["completion_threshold"]
        metric_values["completion"] = comp
        threshold_comparisons["completion"] = (comp, comp_thresh)
        total_checks += 1

        if comp >= comp_thresh:
            evidence.append(f"Completion {comp:.1f}% meets high threshold ({comp_thresh}%)")
            met_count += 1
        else:
            evidence.append(f"Completion {comp:.1f}% below high threshold ({comp_thresh}%)")

    # Check exam performance
    if "average_exam_score" in row and pd.notna(row["average_exam_score"]):
        exam = row["average_exam_score"]
        exam_thresh = criteria["exam_threshold"]
        metric_values["exam_score"] = exam
        threshold_comparisons["exam_score"] = (exam, exam_thresh)
        total_checks += 1

        if exam >= exam_thresh:
            evidence.append(f"Exam score {exam:.1f} meets high threshold ({exam_thresh})")
            met_count += 1
        else:
            evidence.append(f"Exam score {exam:.1f}% below high threshold ({exam_thresh})")

    # Must meet all checked criteria for strong performance
    is_segment = total_checks > 0 and met_count == total_checks

    return is_segment, evidence, metric_values, threshold_comparisons


def check_inconsistent_segment(
    row: pd.Series,
) -> Tuple[bool, List[str], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Check if student meets criteria for Inconsistent segment.

    Criteria:
    - High variance across metrics (some high, some low)
    - At least one metric below 60% threshold
    - At least one metric above 75% threshold

    Args:
        row: Student feature row.

    Returns:
        Tuple of (is_segment, evidence, metric_values, threshold_comparisons).
    """
    criteria = SEGMENT_CRITERIA["inconsistent"]
    evidence: List[str] = []
    metric_values: Dict[str, float] = {}
    threshold_comparisons: Dict[str, Tuple[float, float]] = {}

    metrics = []
    low_count = 0
    high_count = 0

    # Collect metrics
    if "attendance_percentage" in row and pd.notna(row["attendance_percentage"]):
        att = row["attendance_percentage"]
        metrics.append(att)
        metric_values["attendance"] = att

        if att < criteria["attendance_threshold"]:
            low_count += 1
            evidence.append(f"Attendance {att:.1f}% below threshold ({criteria['attendance_threshold']}%)")
        elif att > 75.0:
            high_count += 1
            evidence.append(f"Attendance {att:.1f}% above good threshold (75%)")

    if "assignment_completion_rate" in row and pd.notna(row["assignment_completion_rate"]):
        comp = row["assignment_completion_rate"]
        metrics.append(comp)
        metric_values["completion"] = comp

        if comp < criteria["completion_threshold"]:
            low_count += 1
            evidence.append(f"Completion {comp:.1f}% below threshold ({criteria['completion_threshold']}%)")
        elif comp > 75.0:
            high_count += 1
            evidence.append(f"Completion {comp:.1f}% above good threshold (75%)")

    if "average_exam_score" in row and pd.notna(row["average_exam_score"]):
        exam = row["average_exam_score"]
        metrics.append(exam)
        metric_values["exam_score"] = exam

        if exam < criteria["exam_threshold"]:
            low_count += 1
            evidence.append(f"Exam score {exam:.1f} below threshold ({criteria['exam_threshold']})")
        elif exam > 75.0:
            high_count += 1
            evidence.append(f"Exam score {exam:.1f} above good threshold (75%)")

    # Check variance
    if len(metrics) >= 2:
        variance = np.var(metrics)
        metric_values["variance"] = variance
        threshold_comparisons["variance"] = (variance, criteria["variance_threshold"])

        if variance >= criteria["variance_threshold"]:
            evidence.append(f"High variance ({variance:.1f}) across metrics indicates inconsistency")
        else:
            evidence.append(f"Variance ({variance:.1f}) below threshold ({criteria['variance_threshold']})")

    # Inconsistent if both low and high metrics exist
    is_segment = low_count > 0 and high_count > 0

    return is_segment, evidence, metric_values, threshold_comparisons


def check_declining_engagement_segment(
    row: pd.Series,
) -> Tuple[bool, List[str], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Check if student meets criteria for Declining Engagement segment.

    Criteria:
    - Negative attendance trend <= -10%
    - Negative completion trend <= -10%
    - Negative exam trend <= -10%
    - At least one declining trend

    Args:
        row: Student feature row.

    Returns:
        Tuple of (is_segment, evidence, metric_values, threshold_comparisons).
    """
    criteria = SEGMENT_CRITERIA["declining_engagement"]
    evidence: List[str] = []
    metric_values: Dict[str, float] = {}
    threshold_comparisons: Dict[str, Tuple[float, float]] = {}

    declining_count = 0

    # Check attendance trend
    if "attendance_trend" in row and pd.notna(row["attendance_trend"]):
        att_trend = row["attendance_trend"]
        att_thresh = criteria["attendance_trend_threshold"]
        metric_values["attendance_trend"] = att_trend
        threshold_comparisons["attendance_trend"] = (att_trend, att_thresh)

        if att_trend <= att_thresh:
            evidence.append(f"Attendance trend {att_trend:.1f}% declining (threshold: {att_thresh}%)")
            declining_count += 1
        else:
            evidence.append(f"Attendance trend {att_trend:.1f}% stable or improving")

    # Check completion trend
    if "assignment_trend" in row and pd.notna(row["assignment_trend"]):
        comp_trend = row["assignment_trend"]
        comp_thresh = criteria["completion_trend_threshold"]
        metric_values["assignment_trend"] = comp_trend
        threshold_comparisons["assignment_trend"] = (comp_trend, comp_thresh)

        if comp_trend <= comp_thresh:
            evidence.append(f"Assignment trend {comp_trend:.1f}% declining (threshold: {comp_thresh}%)")
            declining_count += 1
        else:
            evidence.append(f"Assignment trend {comp_trend:.1f}% stable or improving")

    # Check exam trend
    if "exam_trend" in row and pd.notna(row["exam_trend"]):
        exam_trend = row["exam_trend"]
        exam_thresh = criteria["exam_trend_threshold"]
        metric_values["exam_trend"] = exam_trend
        threshold_comparisons["exam_trend"] = (exam_trend, exam_thresh)

        if exam_trend <= exam_thresh:
            evidence.append(f"Exam trend {exam_trend:.1f}% declining (threshold: {exam_thresh}%)")
            declining_count += 1
        else:
            evidence.append(f"Exam trend {exam_trend:.1f}% stable or improving")

    # Declining if at least one trend is declining
    is_segment = declining_count > 0

    return is_segment, evidence, metric_values, threshold_comparisons


def assign_student_segment(
    row: pd.Series,
) -> Tuple[Optional[str], List[str], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Assign a student to a segment based on measurable criteria.

    Priority order:
    1. Strong Performance (most selective)
    2. Declining Engagement (risk-focused)
    3. Inconsistent (pattern-focused)
    4. Consistent (baseline good performance)

    Args:
        row: Student feature row.

    Returns:
        Tuple of (segment_name, evidence, metric_values, threshold_comparisons).
    """
    # Check Strong Performance first (most selective)
    is_strong, strong_evidence, strong_metrics, strong_thresholds = check_strong_performance_segment(row)
    if is_strong:
        return "strong_performance", strong_evidence, strong_metrics, strong_thresholds

    # Check Declining Engagement (risk-focused)
    is_declining, declining_evidence, declining_metrics, declining_thresholds = check_declining_engagement_segment(row)
    if is_declining:
        return "declining_engagement", declining_evidence, declining_metrics, declining_thresholds

    # Check Inconsistent (pattern-focused)
    is_inconsistent, inconsistent_evidence, inconsistent_metrics, inconsistent_thresholds = check_inconsistent_segment(row)
    if is_inconsistent:
        return "inconsistent", inconsistent_evidence, inconsistent_metrics, inconsistent_thresholds

    # Check Consistent (baseline)
    is_consistent, consistent_evidence, consistent_metrics, consistent_thresholds = check_consistent_segment(row)
    if is_consistent:
        return "consistent", consistent_evidence, consistent_metrics, consistent_thresholds

    # No segment assigned
    return None, ["Insufficient data to assign segment"], {}, {}


def calculate_confidence(
    evidence: List[str],
    total_checks: int,
) -> float:
    """Calculate confidence score for segment assignment.

    Args:
        evidence: List of evidence strings.
        total_checks: Total number of criteria checked.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if total_checks == 0:
        return 0.0

    # Count positive evidence (evidence indicating criteria met)
    positive_count = sum(1 for e in evidence if "meets" in e.lower() or "within" in e.lower())

    return round(positive_count / total_checks, 2)


def segment_students(
    student_features_df: pd.DataFrame,
) -> SegmentationResult:
    """Segment students based on measurable engagement criteria.

    Args:
        student_features_df: DataFrame with engineered student features.

    Returns:
        SegmentationResult with segment assignments and evidence.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if "student_id" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'student_id' column")

    total_students = len(student_features_df)
    segment_counts: Dict[str, int] = {}
    student_evidences: List[SegmentEvidence] = []
    unassigned_students: List[Any] = []

    for _, row in student_features_df.iterrows():
        student_id = row["student_id"]
        segment, evidence, metric_values, threshold_comparisons = assign_student_segment(row)

        # Calculate confidence
        total_checks = len(metric_values)
        confidence = calculate_confidence(evidence, total_checks)

        if segment:
            segment_counts[segment] = segment_counts.get(segment, 0) + 1

            student_evidences.append(
                SegmentEvidence(
                    student_id=student_id,
                    assigned_segment=segment,
                    evidence=evidence,
                    metric_values=metric_values,
                    threshold_comparisons=threshold_comparisons,
                    confidence=confidence,
                )
            )
        else:
            unassigned_students.append(student_id)

    # Calculate percentages
    segment_percentages: Dict[str, float] = {}
    for segment, count in segment_counts.items():
        segment_percentages[segment] = round((count / total_students) * 100.0, 2) if total_students > 0 else 0.0

    logger.info(
        "Segmented %d students: %s",
        total_students,
        {k: f"{v} ({segment_percentages[k]}%)" for k, v in segment_counts.items()},
    )

    return SegmentationResult(
        total_students=total_students,
        segment_counts=segment_counts,
        segment_percentages=segment_percentages,
        student_evidence=student_evidences,
        unassigned_students=unassigned_students,
        analysis_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def generate_segmentation_report_markdown(
    result: SegmentationResult,
) -> str:
    """Generate a markdown summary of segmentation analysis.

    Args:
        result: SegmentationResult from analysis.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Student Engagement Segmentation Report",
        "",
        f"**Analysis Date**: {result.analysis_date}",
        f"**Total Students**: {result.total_students:,}",
        f"**Unassigned Students**: {len(result.unassigned_students):,}",
        "",
        "## Segment Overview",
        "",
    ]

    for segment, count in result.segment_counts.items():
        percentage = result.segment_percentages.get(segment, 0.0)
        criteria = SEGMENT_CRITERIA.get(segment, {})
        description = criteria.get("description", "No description available")

        lines.append(f"### {segment.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"- **Count**: {count:,} ({percentage}%)")
        lines.append(f"- **Description**: {description}")
        lines.append("")

    lines.append("## Segment Criteria")
    lines.append("")
    lines.append("All segment assignments are based on measurable thresholds:")
    lines.append("")

    for segment, criteria in SEGMENT_CRITERIA.items():
        lines.append(f"### {segment.replace('_', ' ').title()}")
        lines.append("")
        for key, value in criteria.items():
            if key != "description":
                lines.append(f"- **{key}**: {value}")
        lines.append(f"- **Description**: {criteria.get('description', 'N/A')}")
        lines.append("")

    lines.append("## Student Evidence Sample")
    lines.append("")
    lines.append("Sample of evidence for segment assignments:")
    lines.append("")

    # Show first 5 students from each segment
    for segment in result.segment_counts.keys():
        segment_students = [e for e in result.student_evidence if e.assigned_segment == segment]
        if segment_students:
            lines.append(f"### {segment.replace('_', ' ').title()} (Sample)")
            lines.append("")
            for student in segment_students[:3]:
                lines.append(f"**Student ID**: {student.student_id}")
                lines.append(f"**Confidence**: {student.confidence:.0%}")
                lines.append("**Evidence**:")
                for evidence in student.evidence:
                    lines.append(f"- {evidence}")
                lines.append("")

    if result.unassigned_students:
        lines.append("## Unassigned Students")
        lines.append("")
        lines.append(f"Students with insufficient data for segmentation: {len(result.unassigned_students)}")
        lines.append("")

    return "\n".join(lines)


def get_segment_criteria_summary() -> Dict[str, Dict[str, Any]]:
    """Get summary of all segment criteria.

    Returns:
        Dictionary mapping segment names to their criteria.
    """
    return SEGMENT_CRITERIA.copy()
