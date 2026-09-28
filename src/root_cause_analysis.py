"""Academic Root Cause Analysis Module.

Provides an evidence-based root cause analysis workflow for investigating why
a student or course displays elevated review indicators across:
- attendance
- assignments
- submissions
- exams
- recent trends

CRITICAL ETHICAL & NON-DIAGNOSTIC PRINCIPLE:
This module surfaces observable academic patterns and quantitative evidence to
support institutional review and advising dialogues. It strictly refrains from
diagnosing students with psychological, medical, cognitive, or personal conditions,
and never claims unsupported causes or deterministic future outcomes.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.root_cause_analysis")

# Ethical non-diagnostic disclaimer required on all root cause outputs
NON_DIAGNOSTIC_DISCLAIMER = (
    "ETHICAL NOTICE & DISCLAIMER: This root cause analysis surfaces observable academic "
    "patterns and quantitative evidence to support institutional review. It does NOT "
    "provide personal, psychological, or medical diagnoses, nor does it assert definitive "
    "causation. External life circumstances, health, and qualitative factors are not "
    "captured in institutional data. Findings should guide supportive inquiry and advising "
    "dialogue rather than definitive labeling."
)

# Configurable analytical benchmarks and thresholds
ROOT_CAUSE_THRESHOLDS = {
    # Attendance thresholds
    "attendance_critical": 50.0,
    "attendance_low": 70.0,
    "attendance_consecutive_absence_threshold": 3,
    # Assignment thresholds
    "assignment_score_critical": 40.0,
    "assignment_score_low": 60.0,
    "assignment_fail_threshold": 50.0,
    # Submission thresholds
    "missing_submission_critical": 4,
    "missing_submission_warning": 2,
    "missing_submission_rate_threshold": 0.25,
    "late_submission_warning": 3,
    "late_submission_rate_threshold": 0.30,
    "submission_delay_days_threshold": 3.0,
    # Exam thresholds
    "exam_score_critical": 40.0,
    "exam_score_low": 60.0,
    "exam_score_gap_threshold": 20.0,  # Gap between assignment and exam performance
    # Trend thresholds (negative values indicate decline)
    "trend_critical_decline": -20.0,
    "trend_moderate_decline": -10.0,
    "trend_mild_decline": -5.0,
    # Course-level thresholds
    "course_elevated_review_rate": 25.0,  # % of enrolled students with elevated indicator
    "course_moderate_review_rate": 15.0,
    "course_assignment_bottleneck_fail_rate": 35.0,  # % students failing specific assignment
    "course_attendance_drop_threshold": -10.0,
}


@dataclass
class EvidenceItem:
    """An empirical evidence item supporting an academic review indicator."""

    category: str  # 'attendance', 'assignments', 'submissions', 'exams', 'recent_trends'
    metric_name: str
    observed_value: Any
    benchmark_value: Any
    difference: Optional[float]
    severity: str  # 'CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'NEGLIGIBLE'
    finding: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence item to dictionary."""
        return asdict(self)


@dataclass
class FactorContribution:
    """Aggregated contribution of an academic dimension to review indicator."""

    category: str
    factor_title: str
    severity: str  # 'CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'NEGLIGIBLE'
    score_weight: float  # Numerical weight for ranking
    evidence_items: List[EvidenceItem]
    observed_pattern: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert factor contribution to dictionary."""
        return {
            "category": self.category,
            "factor_title": self.factor_title,
            "severity": self.severity,
            "score_weight": self.score_weight,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "observed_pattern": self.observed_pattern,
        }


@dataclass
class StudentRootCauseReport:
    """Comprehensive evidence-based root cause analysis report for a student."""

    student_id: Any
    overall_review_indicator: str  # 'ELEVATED', 'MODERATE', 'LOW'
    primary_factor: Optional[str]
    secondary_factors: List[str]
    category_evaluations: Dict[str, FactorContribution]
    all_evidence: List[EvidenceItem]
    factual_summary: str
    supportive_recommendations: List[str]
    analysis_timestamp: str
    non_diagnostic_disclaimer: str = NON_DIAGNOSTIC_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "student_id": self.student_id,
            "overall_review_indicator": self.overall_review_indicator,
            "primary_factor": self.primary_factor,
            "secondary_factors": self.secondary_factors,
            "category_evaluations": {
                k: v.to_dict() for k, v in self.category_evaluations.items()
            },
            "all_evidence": [e.to_dict() for e in self.all_evidence],
            "factual_summary": self.factual_summary,
            "supportive_recommendations": self.supportive_recommendations,
            "analysis_timestamp": self.analysis_timestamp,
            "non_diagnostic_disclaimer": self.non_diagnostic_disclaimer,
        }


@dataclass
class CourseRootCauseReport:
    """Evidence-based root cause analysis report for a course/cohort."""

    course_id: Any
    total_students: int
    elevated_students_count: int
    elevated_students_percentage: float
    course_review_indicator: str  # 'ELEVATED', 'MODERATE', 'LOW'
    primary_bottleneck: Optional[str]
    secondary_bottlenecks: List[str]
    category_evaluations: Dict[str, FactorContribution]
    all_evidence: List[EvidenceItem]
    factual_summary: str
    curriculum_recommendations: List[str]
    analysis_timestamp: str
    non_diagnostic_disclaimer: str = NON_DIAGNOSTIC_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "course_id": self.course_id,
            "total_students": self.total_students,
            "elevated_students_count": self.elevated_students_count,
            "elevated_students_percentage": self.elevated_students_percentage,
            "course_review_indicator": self.course_review_indicator,
            "primary_bottleneck": self.primary_bottleneck,
            "secondary_bottlenecks": self.secondary_bottlenecks,
            "category_evaluations": {
                k: v.to_dict() for k, v in self.category_evaluations.items()
            },
            "all_evidence": [e.to_dict() for e in self.all_evidence],
            "factual_summary": self.factual_summary,
            "curriculum_recommendations": self.curriculum_recommendations,
            "analysis_timestamp": self.analysis_timestamp,
            "non_diagnostic_disclaimer": self.non_diagnostic_disclaimer,
        }


# ---------------------------------------------------------------------------
# Individual Dimension Evidence Analyzers
# ---------------------------------------------------------------------------


def analyze_attendance_evidence(
    attendance_percentage: Optional[float],
    total_sessions: Optional[int] = None,
    absent_sessions: Optional[int] = None,
    consecutive_absences: Optional[int] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> FactorContribution:
    """Analyze empirical attendance evidence for a student.

    Args:
        attendance_percentage: Overall attendance percentage.
        total_sessions: Total number of scheduled class sessions.
        absent_sessions: Total sessions recorded absent.
        consecutive_absences: Maximum streak of consecutive absences.
        thresholds: Custom analytical thresholds.

    Returns:
        FactorContribution containing evidence items and severity rating.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS
    evidence: List[EvidenceItem] = []
    score_weight = 0.0

    if attendance_percentage is not None and not pd.isna(attendance_percentage):
        diff = attendance_percentage - th["attendance_low"]
        if attendance_percentage < th["attendance_critical"]:
            severity = "CRITICAL"
            score_weight += 40.0
            finding = (
                f"Critically low attendance of {attendance_percentage:.1f}%, which is "
                f"{abs(diff):.1f} percentage points below institutional benchmark ({th['attendance_low']}%)."
            )
        elif attendance_percentage < th["attendance_low"]:
            severity = "HIGH"
            score_weight += 25.0
            finding = (
                f"Sub-threshold attendance of {attendance_percentage:.1f}%, trailing "
                f"benchmark ({th['attendance_low']}%) by {abs(diff):.1f} percentage points."
            )
        else:
            severity = "NEGLIGIBLE"
            finding = f"Attendance is within normal parameters at {attendance_percentage:.1f}%."

        evidence.append(
            EvidenceItem(
                category="attendance",
                metric_name="attendance_percentage",
                observed_value=round(float(attendance_percentage), 2),
                benchmark_value=th["attendance_low"],
                difference=round(float(diff), 2),
                severity=severity,
                finding=finding,
                supporting_data={"total_sessions": total_sessions, "absent_sessions": absent_sessions},
            )
        )

    # Check consecutive absences
    if consecutive_absences is not None and not pd.isna(consecutive_absences):
        cons_val = int(consecutive_absences)
        cons_th = int(th.get("attendance_consecutive_absence_threshold", 3))
        if cons_val >= cons_th:
            cons_sev = "HIGH" if cons_val >= cons_th * 2 else "MODERATE"
            score_weight += 20.0 if cons_sev == "HIGH" else 10.0
            evidence.append(
                EvidenceItem(
                    category="attendance",
                    metric_name="consecutive_absences",
                    observed_value=cons_val,
                    benchmark_value=cons_th,
                    difference=float(cons_val - cons_th),
                    severity=cons_sev,
                    finding=f"Observed pattern of {cons_val} consecutive unattended class sessions.",
                    supporting_data={"consecutive_absences": cons_val},
                )
            )

    # Determine aggregated dimension severity
    if score_weight >= 35.0:
        overall_sev = "CRITICAL"
        pattern = "Severe attendance disengagement with significant missed classroom contact."
    elif score_weight >= 20.0:
        overall_sev = "HIGH"
        pattern = "Notable attendance shortfalls warranting outreach."
    elif score_weight >= 10.0:
        overall_sev = "MODERATE"
        pattern = "Occasional or borderline attendance irregularities."
    elif evidence:
        overall_sev = "LOW"
        pattern = "Regular attendance records meeting baseline expectations."
    else:
        overall_sev = "NEGLIGIBLE"
        pattern = "Insufficient or unrecorded attendance data."

    return FactorContribution(
        category="attendance",
        factor_title="Attendance Engagement",
        severity=overall_sev,
        score_weight=score_weight,
        evidence_items=evidence,
        observed_pattern=pattern,
    )


def analyze_assignments_evidence(
    average_score: Optional[float],
    completion_rate: Optional[float] = None,
    failing_assignments_count: Optional[int] = None,
    total_assignments: Optional[int] = None,
    lowest_score: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> FactorContribution:
    """Analyze assignment and coursework performance evidence.

    Args:
        average_score: Mean score achieved on submitted coursework.
        completion_rate: Coursework completion percentage.
        failing_assignments_count: Count of assignments with scores below pass mark.
        total_assignments: Total number of evaluated assignments.
        lowest_score: Lowest individual coursework score.
        thresholds: Custom analytical thresholds.

    Returns:
        FactorContribution containing evidence items and severity rating.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS
    evidence: List[EvidenceItem] = []
    score_weight = 0.0

    if average_score is not None and not pd.isna(average_score):
        diff = average_score - th["assignment_score_low"]
        if average_score < th["assignment_score_critical"]:
            severity = "CRITICAL"
            score_weight += 35.0
            finding = (
                f"Critically low coursework score average of {average_score:.1f}, "
                f"{abs(diff):.1f} points below baseline benchmark ({th['assignment_score_low']})."
            )
        elif average_score < th["assignment_score_low"]:
            severity = "HIGH"
            score_weight += 20.0
            finding = (
                f"Coursework average of {average_score:.1f} trails expected "
                f"benchmark ({th['assignment_score_low']}) by {abs(diff):.1f} points."
            )
        else:
            severity = "NEGLIGIBLE"
            finding = f"Coursework score average is satisfactory at {average_score:.1f}."

        evidence.append(
            EvidenceItem(
                category="assignments",
                metric_name="average_assignment_score",
                observed_value=round(float(average_score), 2),
                benchmark_value=th["assignment_score_low"],
                difference=round(float(diff), 2),
                severity=severity,
                finding=finding,
                supporting_data={"lowest_score": lowest_score},
            )
        )

    if failing_assignments_count is not None and not pd.isna(failing_assignments_count):
        fail_cnt = int(failing_assignments_count)
        if fail_cnt >= 2:
            severity = "HIGH" if fail_cnt >= 4 else "MODERATE"
            score_weight += 15.0 if severity == "HIGH" else 8.0
            evidence.append(
                EvidenceItem(
                    category="assignments",
                    metric_name="failing_assignments_count",
                    observed_value=fail_cnt,
                    benchmark_value=0,
                    difference=float(fail_cnt),
                    severity=severity,
                    finding=(
                        f"Student received failing evaluations (<{th['assignment_fail_threshold']}) "
                        f"on {fail_cnt} coursework submissions."
                    ),
                    supporting_data={"total_assignments": total_assignments},
                )
            )

    if score_weight >= 35.0:
        overall_sev = "CRITICAL"
        pattern = "Severe academic difficulty on formative coursework."
    elif score_weight >= 20.0:
        overall_sev = "HIGH"
        pattern = "Consistent academic struggle across multiple assignments."
    elif score_weight >= 8.0:
        overall_sev = "MODERATE"
        pattern = "Isolated coursework difficulties or mixed performance."
    elif evidence:
        overall_sev = "LOW"
        pattern = "Consistently passing coursework performance."
    else:
        overall_sev = "NEGLIGIBLE"
        pattern = "No evaluated coursework records available."

    return FactorContribution(
        category="assignments",
        factor_title="Coursework Performance",
        severity=overall_sev,
        score_weight=score_weight,
        evidence_items=evidence,
        observed_pattern=pattern,
    )


def analyze_submissions_evidence(
    missing_count: Optional[int] = None,
    total_assignments: Optional[int] = None,
    late_count: Optional[int] = None,
    late_rate: Optional[float] = None,
    average_delay_days: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> FactorContribution:
    """Analyze assignment submission habits and timeliness evidence.

    Args:
        missing_count: Count of assignments with no submission record.
        total_assignments: Total expected course assignments.
        late_count: Count of assignments submitted after deadline.
        late_rate: Proportion of submitted assignments that were late.
        average_delay_days: Mean days past deadline for late submissions.
        thresholds: Custom analytical thresholds.

    Returns:
        FactorContribution containing evidence items and severity rating.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS
    evidence: List[EvidenceItem] = []
    score_weight = 0.0

    # Missing submissions
    if missing_count is not None and not pd.isna(missing_count):
        m_cnt = int(missing_count)
        missing_rate = (
            float(m_cnt) / float(total_assignments)
            if (total_assignments and total_assignments > 0)
            else None
        )

        if m_cnt >= th["missing_submission_critical"]:
            severity = "CRITICAL"
            score_weight += 40.0
            finding = (
                f"{m_cnt} expected assignments have no submission record"
                + (f" ({missing_rate:.1%} omission rate)" if missing_rate is not None else "")
                + f", exceeding critical threshold ({th['missing_submission_critical']})."
            )
        elif m_cnt >= th["missing_submission_warning"]:
            severity = "HIGH"
            score_weight += 20.0
            finding = (
                f"{m_cnt} unsubmitted assignments detected"
                + (f" ({missing_rate:.1%} omission rate)" if missing_rate is not None else "")
                + f", reaching warning threshold ({th['missing_submission_warning']})."
            )
        elif m_cnt > 0:
            severity = "MODERATE"
            score_weight += 5.0
            finding = f"{m_cnt} assignment currently has no submission record."
        else:
            severity = "NEGLIGIBLE"
            finding = "All expected assignments were submitted."

        evidence.append(
            EvidenceItem(
                category="submissions",
                metric_name="missing_submission_count",
                observed_value=m_cnt,
                benchmark_value=0,
                difference=float(m_cnt),
                severity=severity,
                finding=finding,
                supporting_data={"missing_rate": missing_rate, "total_assignments": total_assignments},
            )
        )

    # Late submissions
    if late_count is not None and not pd.isna(late_count):
        l_cnt = int(late_count)
        if l_cnt >= th["late_submission_warning"]:
            severity = "HIGH" if l_cnt >= th["late_submission_warning"] * 2 else "MODERATE"
            score_weight += 15.0 if severity == "HIGH" else 8.0
            finding = (
                f"{l_cnt} assignments were submitted past the designated deadline"
                + (f" (late rate: {late_rate:.1%})" if late_rate is not None else "")
                + "."
            )
            evidence.append(
                EvidenceItem(
                    category="submissions",
                    metric_name="late_submission_count",
                    observed_value=l_cnt,
                    benchmark_value=0,
                    difference=float(l_cnt),
                    severity=severity,
                    finding=finding,
                    supporting_data={"late_rate": late_rate, "average_delay_days": average_delay_days},
                )
            )

    # Late submission rate
    if late_rate is not None and not pd.isna(late_rate):
        lr_val = float(late_rate)
        if lr_val >= th["late_submission_rate_threshold"]:
            score_weight += 10.0
            evidence.append(
                EvidenceItem(
                    category="submissions",
                    metric_name="late_submission_rate",
                    observed_value=round(lr_val, 2),
                    benchmark_value=th["late_submission_rate_threshold"],
                    difference=round(lr_val - th["late_submission_rate_threshold"], 2),
                    severity="HIGH" if lr_val >= th["late_submission_rate_threshold"] * 1.5 else "MODERATE",
                    finding=f"Elevated late submission rate of {lr_val:.1%} (benchmark: <{th['late_submission_rate_threshold']:.0%}).",
                    supporting_data={"late_rate": lr_val},
                )
            )

    # Latency delays
    if average_delay_days is not None and not pd.isna(average_delay_days):
        delay_val = float(average_delay_days)
        if delay_val >= th["submission_delay_days_threshold"]:
            score_weight += 10.0
            evidence.append(
                EvidenceItem(
                    category="submissions",
                    metric_name="average_submission_delay_days",
                    observed_value=round(delay_val, 1),
                    benchmark_value=0.0,
                    difference=round(delay_val, 1),
                    severity="MODERATE",
                    finding=f"Average submission delay among late coursework is {delay_val:.1f} days past deadline.",
                    supporting_data={"average_delay_days": delay_val},
                )
            )

    if score_weight >= 35.0:
        overall_sev = "CRITICAL"
        pattern = "Severe coursework omission and persistent deadline failure."
    elif score_weight >= 20.0:
        overall_sev = "HIGH"
        pattern = "Multiple missing or late assignments indicating coursework workflow disruption."
    elif score_weight >= 8.0:
        overall_sev = "MODERATE"
        pattern = "Occasional missed deadlines or late submissions."
    elif evidence:
        overall_sev = "LOW"
        pattern = "Timely and complete coursework submissions."
    else:
        overall_sev = "NEGLIGIBLE"
        pattern = "No submission tracking records available."

    return FactorContribution(
        category="submissions",
        factor_title="Submission Timeliness & Completion",
        severity=overall_sev,
        score_weight=score_weight,
        evidence_items=evidence,
        observed_pattern=pattern,
    )


def analyze_exams_evidence(
    average_exam_score: Optional[float],
    recent_exam_score: Optional[float] = None,
    failing_exams_count: Optional[int] = None,
    total_exams: Optional[int] = None,
    average_assignment_score: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> FactorContribution:
    """Analyze summative examination performance evidence.

    Args:
        average_exam_score: Mean score across completed examinations.
        recent_exam_score: Score on the most recent examination sitting.
        failing_exams_count: Number of exams with score below passing mark.
        total_exams: Total number of examinations taken.
        average_assignment_score: Formative assignment average to detect divergence.
        thresholds: Custom analytical thresholds.

    Returns:
        FactorContribution containing evidence items and severity rating.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS
    evidence: List[EvidenceItem] = []
    score_weight = 0.0

    if average_exam_score is not None and not pd.isna(average_exam_score):
        diff = average_exam_score - th["exam_score_low"]
        if average_exam_score < th["exam_score_critical"]:
            severity = "CRITICAL"
            score_weight += 40.0
            finding = (
                f"Critically low summative examination average of {average_exam_score:.1f}, "
                f"{abs(diff):.1f} points below benchmark ({th['exam_score_low']})."
            )
        elif average_exam_score < th["exam_score_low"]:
            severity = "HIGH"
            score_weight += 25.0
            finding = (
                f"Examination average of {average_exam_score:.1f} trails benchmark "
                f"({th['exam_score_low']}) by {abs(diff):.1f} points."
            )
        else:
            severity = "NEGLIGIBLE"
            finding = f"Summative examination average is sound at {average_exam_score:.1f}."

        evidence.append(
            EvidenceItem(
                category="exams",
                metric_name="average_exam_score",
                observed_value=round(float(average_exam_score), 2),
                benchmark_value=th["exam_score_low"],
                difference=round(float(diff), 2),
                severity=severity,
                finding=finding,
                supporting_data={"total_exams": total_exams},
            )
        )

    # Exam vs Assignment Divergence check
    if (
        average_exam_score is not None
        and not pd.isna(average_exam_score)
        and average_assignment_score is not None
        and not pd.isna(average_assignment_score)
    ):
        gap = average_assignment_score - average_exam_score
        if gap >= th["exam_score_gap_threshold"]:
            severity = "HIGH" if gap >= 30.0 else "MODERATE"
            score_weight += 15.0 if severity == "HIGH" else 8.0
            evidence.append(
                EvidenceItem(
                    category="exams",
                    metric_name="exam_vs_assignment_gap",
                    observed_value=round(float(gap), 2),
                    benchmark_value=th["exam_score_gap_threshold"],
                    difference=round(float(gap - th["exam_score_gap_threshold"]), 2),
                    severity=severity,
                    finding=(
                        f"Performance divergence observed: coursework average ({average_assignment_score:.1f}) "
                        f"exceeds exam average ({average_exam_score:.1f}) by {gap:.1f} points, "
                        "indicating difficulty translating coursework preparation into summative test performance."
                    ),
                    supporting_data={
                        "assignment_avg": average_assignment_score,
                        "exam_avg": average_exam_score,
                    },
                )
            )

    if score_weight >= 35.0:
        overall_sev = "CRITICAL"
        pattern = "Severe summative assessment failure risking course completion."
    elif score_weight >= 20.0:
        overall_sev = "HIGH"
        pattern = "Struggling on summative assessments despite class progression."
    elif score_weight >= 8.0:
        overall_sev = "MODERATE"
        pattern = "Moderate gap or inconsistency on exam evaluations."
    elif evidence:
        overall_sev = "LOW"
        pattern = "Consistently passing summative examination marks."
    else:
        overall_sev = "NEGLIGIBLE"
        pattern = "No examination records recorded."

    return FactorContribution(
        category="exams",
        factor_title="Summative Assessment Performance",
        severity=overall_sev,
        score_weight=score_weight,
        evidence_items=evidence,
        observed_pattern=pattern,
    )


def analyze_recent_trends_evidence(
    attendance_trend: Optional[float] = None,
    assignment_trend: Optional[float] = None,
    exam_trend: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> FactorContribution:
    """Analyze temporal trajectories and recent performance momentum.

    Args:
        attendance_trend: Delta in attendance % between late and early terms.
        assignment_trend: Delta in assignment scores between late and early coursework.
        exam_trend: Delta between recent exam and initial exam score.
        thresholds: Custom analytical thresholds.

    Returns:
        FactorContribution containing evidence items and severity rating.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS
    evidence: List[EvidenceItem] = []
    score_weight = 0.0

    trend_specs = [
        ("attendance_trend", attendance_trend, "attendance engagement"),
        ("assignment_trend", assignment_trend, "coursework scores"),
        ("exam_trend", exam_trend, "examination marks"),
    ]

    for metric_name, val, label in trend_specs:
        if val is None or pd.isna(val):
            continue

        float_val = float(val)
        if float_val <= th["trend_critical_decline"]:
            severity = "CRITICAL"
            score_weight += 25.0
            finding = (
                f"Critical downward trajectory in {label} ({float_val:+.1f} points), "
                f"exceeding severe decline threshold ({th['trend_critical_decline']}%)."
            )
        elif float_val <= th["trend_moderate_decline"]:
            severity = "HIGH"
            score_weight += 15.0
            finding = (
                f"Moderate downward trajectory in {label} ({float_val:+.1f} points), "
                f"reaching decline threshold ({th['trend_moderate_decline']}%)."
            )
        elif float_val <= th["trend_mild_decline"]:
            severity = "MODERATE"
            score_weight += 5.0
            finding = f"Mild recent cooling in {label} ({float_val:+.1f} points)."
        elif float_val >= 5.0:
            severity = "NEGLIGIBLE"
            finding = f"Positive upward trajectory observed in {label} ({float_val:+.1f} points)."
        else:
            severity = "NEGLIGIBLE"
            finding = f"{label.capitalize()} has remained stable ({float_val:+.1f} points)."

        evidence.append(
            EvidenceItem(
                category="recent_trends",
                metric_name=metric_name,
                observed_value=round(float_val, 2),
                benchmark_value=0.0,
                difference=round(float_val, 2),
                severity=severity,
                finding=finding,
                supporting_data={"metric": metric_name, "delta": float_val},
            )
        )

    if score_weight >= 35.0:
        overall_sev = "CRITICAL"
        pattern = "Compounding multi-dimensional decline across recent academic periods."
    elif score_weight >= 15.0:
        overall_sev = "HIGH"
        pattern = "Clear recent deterioration in student engagement or scores."
    elif score_weight >= 5.0:
        overall_sev = "MODERATE"
        pattern = "Isolated downward trend in one academic signal."
    elif evidence:
        overall_sev = "LOW"
        pattern = "Stable or positive academic trajectories."
    else:
        overall_sev = "NEGLIGIBLE"
        pattern = "Insufficient longitudinal data to establish trend direction."

    return FactorContribution(
        category="recent_trends",
        factor_title="Recent Academic Trends & Momentum",
        severity=overall_sev,
        score_weight=score_weight,
        evidence_items=evidence,
        observed_pattern=pattern,
    )


# ---------------------------------------------------------------------------
# Student Root Cause Investigation Workflow
# ---------------------------------------------------------------------------


def analyze_student_root_cause(
    student_id: Any,
    student_features: Optional[Union[pd.Series, Dict[str, Any]]] = None,
    attendance_records: Optional[pd.DataFrame] = None,
    assignment_records: Optional[pd.DataFrame] = None,
    submission_records: Optional[pd.DataFrame] = None,
    exam_records: Optional[pd.DataFrame] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> StudentRootCauseReport:
    """Execute complete evidence-based root cause analysis for an individual student.

    Investigates why a student shows elevated review indicators across:
    - attendance
    - assignments
    - submissions
    - exams
    - recent trends

    Args:
        student_id: Unique student identifier.
        student_features: Pre-calculated feature row or dictionary.
        attendance_records: Raw attendance DataFrame for granular inspection.
        assignment_records: Course assignments DataFrame.
        submission_records: Student submission history DataFrame.
        exam_records: Student exam results DataFrame.
        thresholds: Custom analytical thresholds.

    Returns:
        StudentRootCauseReport detailing evidence, factor ranking, and supportive actions.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS
    features: Dict[str, Any] = {}

    if student_features is not None:
        if isinstance(student_features, pd.Series):
            features = student_features.to_dict()
        elif isinstance(student_features, dict):
            features = student_features.copy()

    # Extract or derive attendance metrics
    att_pct = features.get("attendance_percentage")
    tot_sessions = features.get("total_sessions")
    absent_sessions = features.get("absent_sessions")
    consecutive_absences = features.get("consecutive_absences")

    if attendance_records is not None and not attendance_records.empty:
        stu_att = attendance_records[attendance_records["student_id"] == student_id]
        if not stu_att.empty:
            tot_sessions = len(stu_att)
            absent_cnt = (stu_att["status"].astype(str).str.lower() == "absent").sum()
            present_cnt = (stu_att["status"].astype(str).str.lower() == "present").sum()
            late_cnt = (stu_att["status"].astype(str).str.lower() == "late").sum()
            absent_sessions = int(absent_cnt)
            if att_pct is None and tot_sessions > 0:
                att_pct = float(present_cnt + 0.5 * late_cnt) / float(tot_sessions) * 100.0

    # Extract or derive assignment metrics
    avg_assign = features.get("average_assignment_score")
    assign_comp = features.get("assignment_completion_rate")
    failing_assign = features.get("failing_assignments_count")
    total_assign = features.get("total_assignments")
    lowest_assign = features.get("lowest_assignment_score")

    # Extract or derive submission metrics
    missing_sub = features.get("missing_submission_count")
    late_sub = features.get("late_submission_count")
    late_rate = features.get("late_submission_rate")
    avg_delay = features.get("average_submission_delay_days")

    if submission_records is not None and not submission_records.empty:
        stu_sub = submission_records[submission_records["student_id"] == student_id]
        if not stu_sub.empty:
            if avg_assign is None and "score" in stu_sub.columns:
                valid_scores = stu_sub["score"].dropna()
                if not valid_scores.empty:
                    avg_assign = float(valid_scores.mean())
                    lowest_assign = float(valid_scores.min())
                    failing_assign = int((valid_scores < th["assignment_fail_threshold"]).sum())

    # Extract or derive exam metrics
    avg_exam = features.get("average_exam_score")
    recent_exam = features.get("recent_exam_score")
    failing_exams = features.get("failing_exams_count")
    total_exams = features.get("total_exams")

    if exam_records is not None and not exam_records.empty:
        stu_exam = exam_records[exam_records["student_id"] == student_id]
        if not stu_exam.empty:
            if avg_exam is None and "score" in stu_exam.columns:
                valid_exam_scores = stu_exam["score"].dropna()
                if not valid_exam_scores.empty:
                    avg_exam = float(valid_exam_scores.mean())
                    failing_exams = int((valid_exam_scores < th["exam_score_low"]).sum())
                    total_exams = len(valid_exam_scores)

    # Extract trend metrics
    att_trend = features.get("attendance_trend")
    assign_trend = features.get("assignment_trend")
    exam_trend = features.get("exam_trend")

    # Perform dimensional evaluations
    cat_evals: Dict[str, FactorContribution] = {
        "attendance": analyze_attendance_evidence(
            att_pct, tot_sessions, absent_sessions, consecutive_absences, th
        ),
        "assignments": analyze_assignments_evidence(
            avg_assign, assign_comp, failing_assign, total_assign, lowest_assign, th
        ),
        "submissions": analyze_submissions_evidence(
            missing_sub, total_assign, late_sub, late_rate, avg_delay, th
        ),
        "exams": analyze_exams_evidence(
            avg_exam, recent_exam, failing_exams, total_exams, avg_assign, th
        ),
        "recent_trends": analyze_recent_trends_evidence(
            att_trend, assign_trend, exam_trend, th
        ),
    }

    # Aggregate all individual evidence items
    all_evidence: List[EvidenceItem] = []
    for cat_contrib in cat_evals.values():
        all_evidence.extend(cat_contrib.evidence_items)

    # Rank factors by score weight
    sorted_factors = sorted(
        [fc for fc in cat_evals.values() if fc.score_weight > 0],
        key=lambda x: x.score_weight,
        reverse=True,
    )

    primary_factor: Optional[str] = None
    secondary_factors: List[str] = []

    if sorted_factors:
        primary_fc = sorted_factors[0]
        primary_factor = f"{primary_fc.factor_title} ({primary_fc.severity})"
        for sec in sorted_factors[1:]:
            if sec.score_weight >= 10.0 or sec.severity in ["CRITICAL", "HIGH", "MODERATE"]:
                secondary_factors.append(f"{sec.factor_title} ({sec.severity})")

    # Overall review indicator determination
    max_severity = max(
        (fc.severity for fc in cat_evals.values()),
        key=lambda s: ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"].index(s),
        default="LOW",
    )
    total_weight = sum(fc.score_weight for fc in cat_evals.values())

    if max_severity == "CRITICAL" or total_weight >= 50.0:
        overall_indicator = "ELEVATED"
    elif max_severity in ["HIGH", "MODERATE"] or total_weight >= 20.0:
        overall_indicator = "MODERATE"
    else:
        overall_indicator = "LOW"

    # Construct factual narrative summary
    summary_parts: List[str] = []
    if primary_factor:
        summary_parts.append(
            f"The primary academic indicator contributing to review is {sorted_factors[0].factor_title.lower()}: "
            f"{sorted_factors[0].observed_pattern}"
        )
    if secondary_factors:
        secondary_titles = [s.split(" (")[0].lower() for s in secondary_factors]
        summary_parts.append(
            f"Secondary contributing factors include {', '.join(secondary_titles)}."
        )
    if not summary_parts:
        summary_parts.append("All observed academic signals remain within standard institutional benchmarks.")

    factual_summary = " ".join(summary_parts)

    # Construct supportive, non-diagnostic recommendations
    recommendations: List[str] = []
    if cat_evals["attendance"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Schedule an advising check-in to discuss attendance patterns and explore academic support resources."
        )
    if cat_evals["submissions"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Review outstanding coursework submissions and discuss time management or workflow strategies."
        )
    if cat_evals["assignments"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Provide formative assignment feedback review and refer to departmental tutoring or TA office hours."
        )
    if cat_evals["exams"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Encourage participation in exam preparation workshops and review past assessment rubrics."
        )
    if cat_evals["recent_trends"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Conduct proactive mid-term check-in to address recent trajectory cooling before upcoming assessments."
        )

    if not recommendations:
        recommendations.append("Continue standard academic monitoring; no special intervention required.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        "Completed root cause analysis for student %s: Indicator=%s, Primary=%s",
        student_id,
        overall_indicator,
        primary_factor,
    )

    return StudentRootCauseReport(
        student_id=student_id,
        overall_review_indicator=overall_indicator,
        primary_factor=primary_factor,
        secondary_factors=secondary_factors,
        category_evaluations=cat_evals,
        all_evidence=all_evidence,
        factual_summary=factual_summary,
        supportive_recommendations=recommendations,
        analysis_timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Course / Cohort Root Cause Investigation Workflow
# ---------------------------------------------------------------------------


def analyze_course_root_cause(
    course_id: Any,
    course_student_features: pd.DataFrame,
    course_assignments: Optional[pd.DataFrame] = None,
    course_submissions: Optional[pd.DataFrame] = None,
    course_exams: Optional[pd.DataFrame] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> CourseRootCauseReport:
    """Execute evidence-based root cause analysis for a course or cohort.

    Investigates why a course exhibits elevated review indicators across:
    - attendance patterns across cohort
    - assignment difficulty and bottlenecks
    - submission completion and missing rates
    - examination distribution
    - cohort recent trends

    Args:
        course_id: Unique course identifier.
        course_student_features: DataFrame of student features enrolled in course.
        course_assignments: Metadata for course assignments.
        course_submissions: Course submissions dataset.
        course_exams: Course exams dataset.
        thresholds: Custom analytical thresholds.

    Returns:
        CourseRootCauseReport detailing cohort evidence, bottleneck factors, and curriculum actions.

    Raises:
        DataValidationError: If course_student_features is empty or missing student_id.
    """
    th = thresholds or ROOT_CAUSE_THRESHOLDS

    if course_student_features is None or course_student_features.empty:
        raise DataValidationError("course_student_features must not be empty")

    if "student_id" not in course_student_features.columns:
        raise DataValidationError("course_student_features must contain 'student_id' column")

    total_students = len(course_student_features)
    evidence: List[EvidenceItem] = []
    cat_evals: Dict[str, FactorContribution] = {}

    # 1. Course Attendance Analysis
    att_weight = 0.0
    att_evidence: List[EvidenceItem] = []
    if "attendance_percentage" in course_student_features.columns:
        att_series = course_student_features["attendance_percentage"].dropna()
        if not att_series.empty:
            course_avg_att = float(att_series.mean())
            low_att_pct = float((att_series < th["attendance_low"]).sum()) / total_students * 100.0

            if course_avg_att < th["attendance_low"]:
                sev = "CRITICAL" if course_avg_att < th["attendance_critical"] else "HIGH"
                att_weight += 30.0
                finding = (
                    f"Course average attendance of {course_avg_att:.1f}% is below benchmark ({th['attendance_low']}%), "
                    f"with {low_att_pct:.1f}% of enrolled students below threshold."
                )
            elif low_att_pct >= 30.0:
                sev = "MODERATE"
                att_weight += 15.0
                finding = f"{low_att_pct:.1f}% of enrolled students display sub-threshold attendance."
            else:
                sev = "NEGLIGIBLE"
                finding = f"Course attendance is healthy at {course_avg_att:.1f}% average."

            att_evidence.append(
                EvidenceItem(
                    category="attendance",
                    metric_name="course_average_attendance",
                    observed_value=round(course_avg_att, 2),
                    benchmark_value=th["attendance_low"],
                    difference=round(course_avg_att - th["attendance_low"], 2),
                    severity=sev,
                    finding=finding,
                    supporting_data={"low_attendance_percentage": low_att_pct},
                )
            )

    cat_evals["attendance"] = FactorContribution(
        category="attendance",
        factor_title="Cohort Attendance Distribution",
        severity="CRITICAL" if att_weight >= 25 else "HIGH" if att_weight >= 15 else "LOW",
        score_weight=att_weight,
        evidence_items=att_evidence,
        observed_pattern="Widespread attendance deficits across class sessions" if att_weight >= 15 else "Standard class attendance",
    )

    # 2. Course Assignment & Bottleneck Analysis
    assign_weight = 0.0
    assign_evidence: List[EvidenceItem] = []
    bottleneck_assignments: List[str] = []

    if course_submissions is not None and not course_submissions.empty:
        # Check by assignment_id
        if "assignment_id" in course_submissions.columns and "score" in course_submissions.columns:
            grouped = course_submissions.groupby("assignment_id")["score"].agg(["mean", "count"])
            for assign_id, row in grouped.iterrows():
                sub_df = course_submissions[course_submissions["assignment_id"] == assign_id]
                fail_rate = float((sub_df["score"] < th["assignment_fail_threshold"]).sum()) / len(sub_df) * 100.0
                if fail_rate >= th["course_assignment_bottleneck_fail_rate"]:
                    bottleneck_assignments.append(str(assign_id))
                    assign_evidence.append(
                        EvidenceItem(
                            category="assignments",
                            metric_name=f"bottleneck_assignment_{assign_id}",
                            observed_value=round(fail_rate, 1),
                            benchmark_value=th["course_assignment_bottleneck_fail_rate"],
                            difference=round(fail_rate - th["course_assignment_bottleneck_fail_rate"], 1),
                            severity="HIGH",
                            finding=(
                                f"Coursework bottleneck on Assignment {assign_id}: {fail_rate:.1f}% of submissions "
                                f"scored below passing mark (mean score: {row['mean']:.1f})."
                            ),
                            supporting_data={"assignment_id": str(assign_id), "mean_score": float(row["mean"])},
                        )
                    )
                    assign_weight += 20.0

    if "average_assignment_score" in course_student_features.columns:
        assign_scores = course_student_features["average_assignment_score"].dropna()
        if not assign_scores.empty:
            course_avg_assign = float(assign_scores.mean())
            if course_avg_assign < th["assignment_score_low"]:
                assign_weight += 20.0
                assign_evidence.append(
                    EvidenceItem(
                        category="assignments",
                        metric_name="course_average_assignment_score",
                        observed_value=round(course_avg_assign, 2),
                        benchmark_value=th["assignment_score_low"],
                        difference=round(course_avg_assign - th["assignment_score_low"], 2),
                        severity="HIGH",
                        finding=f"Course mean coursework score ({course_avg_assign:.1f}) trails benchmark ({th['assignment_score_low']}).",
                        supporting_data={"course_avg_assignment": course_avg_assign},
                    )
                )

    cat_evals["assignments"] = FactorContribution(
        category="assignments",
        factor_title="Assignment & Coursework Bottlenecks",
        severity="CRITICAL" if assign_weight >= 35 else "HIGH" if assign_weight >= 20 else "LOW",
        score_weight=assign_weight,
        evidence_items=assign_evidence,
        observed_pattern=(
            f"Specific assignment bottlenecks identified ({', '.join(bottleneck_assignments)})"
            if bottleneck_assignments
            else "Standard coursework distribution"
        ),
    )

    # 3. Course Submissions Analysis
    sub_weight = 0.0
    sub_evidence: List[EvidenceItem] = []
    if "missing_submission_count" in course_student_features.columns:
        missing_series = course_student_features["missing_submission_count"].dropna()
        if not missing_series.empty:
            total_missing = int(missing_series.sum())
            students_with_missing = int((missing_series >= th["missing_submission_warning"]).sum())
            pct_students_missing = float(students_with_missing) / total_students * 100.0

            if pct_students_missing >= 25.0:
                sev = "HIGH"
                sub_weight += 25.0
                finding = (
                    f"{pct_students_missing:.1f}% of enrolled students have {th['missing_submission_warning']}+ "
                    f"unsubmitted assignments ({total_missing} total unsubmitted)."
                )
            elif total_missing > 0:
                sev = "MODERATE"
                sub_weight += 10.0
                finding = f"{total_missing} total unsubmitted assignments across cohort."
            else:
                sev = "NEGLIGIBLE"
                finding = "Full submission compliance across cohort."

            sub_evidence.append(
                EvidenceItem(
                    category="submissions",
                    metric_name="cohort_missing_submissions",
                    observed_value=total_missing,
                    benchmark_value=0,
                    difference=float(total_missing),
                    severity=sev,
                    finding=finding,
                    supporting_data={"students_with_missing": students_with_missing},
                )
            )

    cat_evals["submissions"] = FactorContribution(
        category="submissions",
        factor_title="Cohort Submission Timeliness",
        severity="HIGH" if sub_weight >= 20 else "MODERATE" if sub_weight >= 10 else "LOW",
        score_weight=sub_weight,
        evidence_items=sub_evidence,
        observed_pattern="Substantial coursework omission rate across cohort" if sub_weight >= 20 else "Acceptable submission completion",
    )

    # 4. Course Exam Analysis
    exam_weight = 0.0
    exam_evidence: List[EvidenceItem] = []
    if "average_exam_score" in course_student_features.columns:
        exam_series = course_student_features["average_exam_score"].dropna()
        if not exam_series.empty:
            course_exam_avg = float(exam_series.mean())
            fail_exam_pct = float((exam_series < th["exam_score_low"]).sum()) / total_students * 100.0
            if course_exam_avg < th["exam_score_low"]:
                sev = "CRITICAL" if course_exam_avg < th["exam_score_critical"] else "HIGH"
                exam_weight += 30.0
                finding = (
                    f"Course summative exam average of {course_exam_avg:.1f} is below benchmark ({th['exam_score_low']}), "
                    f"with {fail_exam_pct:.1f}% of students scoring in the sub-threshold range."
                )
            elif fail_exam_pct >= 25.0:
                sev = "MODERATE"
                exam_weight += 15.0
                finding = f"{fail_exam_pct:.1f}% of students achieved sub-threshold marks on examinations."
            else:
                sev = "NEGLIGIBLE"
                finding = f"Exam performance is sound at {course_exam_avg:.1f} average."

            exam_evidence.append(
                EvidenceItem(
                    category="exams",
                    metric_name="course_average_exam_score",
                    observed_value=round(course_exam_avg, 2),
                    benchmark_value=th["exam_score_low"],
                    difference=round(course_exam_avg - th["exam_score_low"], 2),
                    severity=sev,
                    finding=finding,
                    supporting_data={"fail_exam_pct": fail_exam_pct},
                )
            )

    cat_evals["exams"] = FactorContribution(
        category="exams",
        factor_title="Summative Assessment Outcomes",
        severity="CRITICAL" if exam_weight >= 25 else "HIGH" if exam_weight >= 15 else "LOW",
        score_weight=exam_weight,
        evidence_items=exam_evidence,
        observed_pattern="High cohort exam struggle" if exam_weight >= 15 else "Consistent exam results",
    )

    # 5. Course Trends Analysis
    trend_weight = 0.0
    trend_evidence: List[EvidenceItem] = []
    for trend_col, label in [
        ("attendance_trend", "attendance trajectory"),
        ("assignment_trend", "coursework score trajectory"),
        ("exam_trend", "exam performance trajectory"),
    ]:
        if trend_col in course_student_features.columns:
            tr_series = course_student_features[trend_col].dropna()
            if not tr_series.empty:
                avg_tr = float(tr_series.mean())
                if avg_tr <= th["trend_moderate_decline"]:
                    sev = "HIGH"
                    trend_weight += 15.0
                    trend_evidence.append(
                        EvidenceItem(
                            category="recent_trends",
                            metric_name=f"course_avg_{trend_col}",
                            observed_value=round(avg_tr, 2),
                            benchmark_value=0.0,
                            difference=round(avg_tr, 2),
                            severity=sev,
                            finding=f"Cohort displays an average decline of {avg_tr:+.1f} points in {label}.",
                            supporting_data={"average_trend": avg_tr},
                        )
                    )

    cat_evals["recent_trends"] = FactorContribution(
        category="recent_trends",
        factor_title="Cohort Momentum & Trajectory",
        severity="HIGH" if trend_weight >= 20 else "MODERATE" if trend_weight >= 10 else "LOW",
        score_weight=trend_weight,
        evidence_items=trend_evidence,
        observed_pattern="General downward trajectory across later term" if trend_weight >= 10 else "Stable cohort momentum",
    )

    # Aggregate evidence items
    all_evidence = []
    for fc in cat_evals.values():
        all_evidence.extend(fc.evidence_items)

    # Calculate review rate
    elevated_count = 0
    if "risk_level" in course_student_features.columns:
        elevated_count = int((course_student_features["risk_level"].astype(str).str.upper() == "ELEVATED").sum())
    else:
        # Fallback estimation based on features
        for _, s_row in course_student_features.iterrows():
            rep = analyze_student_root_cause(student_id=s_row["student_id"], student_features=s_row, thresholds=th)
            if rep.overall_review_indicator == "ELEVATED":
                elevated_count += 1

    elevated_pct = float(elevated_count) / float(total_students) * 100.0 if total_students > 0 else 0.0

    if elevated_pct >= th["course_elevated_review_rate"]:
        course_indicator = "ELEVATED"
    elif elevated_pct >= th["course_moderate_review_rate"]:
        course_indicator = "MODERATE"
    else:
        course_indicator = "LOW"

    # Identify course bottlenecks
    sorted_fc = sorted(
        [fc for fc in cat_evals.values() if fc.score_weight > 0],
        key=lambda x: x.score_weight,
        reverse=True,
    )

    primary_bottleneck: Optional[str] = None
    secondary_bottlenecks: List[str] = []

    if sorted_fc:
        primary_bottleneck = f"{sorted_fc[0].factor_title} ({sorted_fc[0].severity})"
        for sec in sorted_fc[1:]:
            if sec.score_weight >= 10.0:
                secondary_bottlenecks.append(f"{sec.factor_title} ({sec.severity})")

    # Curriculum recommendations
    recommendations: List[str] = []
    if bottleneck_assignments:
        recommendations.append(
            f"Review pedagogical pacing and clarify problem statements for bottleneck coursework ({', '.join(bottleneck_assignments)})."
        )
    if cat_evals["attendance"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Investigate scheduling conflicts or class delivery engagement factors contributing to low attendance."
        )
    if cat_evals["submissions"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Consider pacing coursework deadlines and introducing early formative submission check-ins."
        )
    if cat_evals["exams"].severity in ["CRITICAL", "HIGH"]:
        recommendations.append(
            "Organize targeted review sessions covering foundational topics prior to major summative exams."
        )

    if not recommendations:
        recommendations.append("Course engagement indicators are healthy; continue current instructional syllabus.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    factual_summary = (
        f"Course {course_id} includes {total_students} enrolled students, with {elevated_count} "
        f"({elevated_pct:.1f}%) exhibiting elevated review indicators. "
    )
    if primary_bottleneck:
        factual_summary += f"The leading cohort driver is {primary_bottleneck}."
    else:
        factual_summary += "Cohort engagement is in alignment with institutional targets."

    return CourseRootCauseReport(
        course_id=course_id,
        total_students=total_students,
        elevated_students_count=elevated_count,
        elevated_students_percentage=round(elevated_pct, 2),
        course_review_indicator=course_indicator,
        primary_bottleneck=primary_bottleneck,
        secondary_bottlenecks=secondary_bottlenecks,
        category_evaluations=cat_evals,
        all_evidence=all_evidence,
        factual_summary=factual_summary,
        curriculum_recommendations=recommendations,
        analysis_timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Batch Operations & Investigation Workflow Runner
# ---------------------------------------------------------------------------


def batch_analyze_students(
    student_features_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    filter_elevated_only: bool = False,
) -> List[StudentRootCauseReport]:
    """Perform root cause analysis across multiple students.

    Args:
        student_features_df: DataFrame of student features.
        thresholds: Custom analytical thresholds.
        filter_elevated_only: If True, only return reports for students with ELEVATED indicators.

    Returns:
        List of StudentRootCauseReport objects.

    Raises:
        DataValidationError: If student_features_df is missing student_id.
    """
    if "student_id" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'student_id' column")

    reports: List[StudentRootCauseReport] = []
    for _, row in student_features_df.iterrows():
        rep = analyze_student_root_cause(
            student_id=row["student_id"],
            student_features=row,
            thresholds=thresholds,
        )
        if filter_elevated_only and rep.overall_review_indicator != "ELEVATED":
            continue
        reports.append(rep)

    logger.info("Batch student root cause analysis completed: %d reports generated", len(reports))
    return reports


def identify_students_for_investigation(
    student_features_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Identify students exhibiting elevated review indicators that warrant investigation.

    Args:
        student_features_df: DataFrame of student features.
        thresholds: Custom analytical thresholds.

    Returns:
        Filtered DataFrame of priority students for root cause investigation.

    Raises:
        DataValidationError: If student_features_df is missing student_id.
    """
    if "student_id" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'student_id' column")

    reports = batch_analyze_students(student_features_df, thresholds=thresholds)
    elevated_ids = [r.student_id for r in reports if r.overall_review_indicator in ["ELEVATED", "MODERATE"]]

    return student_features_df[student_features_df["student_id"].isin(elevated_ids)].copy()


def run_root_cause_investigation(
    student_features_df: pd.DataFrame,
    courses_df: Optional[pd.DataFrame] = None,
    attendance_df: Optional[pd.DataFrame] = None,
    assignments_df: Optional[pd.DataFrame] = None,
    submissions_df: Optional[pd.DataFrame] = None,
    exams_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end root cause investigation workflow across students and courses.

    Args:
        student_features_df: DataFrame of student features.
        courses_df: Optional courses metadata DataFrame.
        attendance_df: Granular attendance DataFrame.
        assignments_df: Granular assignments DataFrame.
        submissions_df: Granular submissions DataFrame.
        exams_df: Granular exams DataFrame.
        thresholds: Custom analytical thresholds.

    Returns:
        Dictionary containing workflow results, student reports, and course reports.
    """
    student_reports = batch_analyze_students(student_features_df, thresholds=thresholds)
    elevated_reports = [r for r in student_reports if r.overall_review_indicator == "ELEVATED"]
    moderate_reports = [r for r in student_reports if r.overall_review_indicator == "MODERATE"]

    course_reports: List[CourseRootCauseReport] = []
    if "course_id" in student_features_df.columns:
        for course_id, group in student_features_df.groupby("course_id"):
            c_subs = (
                submissions_df[submissions_df["assignment_id"].isin(
                    assignments_df[assignments_df["course_id"] == course_id]["assignment_id"]
                )]
                if (submissions_df is not None and assignments_df is not None and "course_id" in assignments_df.columns)
                else None
            )
            c_rep = analyze_course_root_cause(
                course_id=course_id,
                course_student_features=group,
                course_assignments=assignments_df,
                course_submissions=c_subs,
                course_exams=exams_df,
                thresholds=thresholds,
            )
            course_reports.append(c_rep)

    return {
        "total_students_evaluated": len(student_reports),
        "elevated_students_count": len(elevated_reports),
        "moderate_students_count": len(moderate_reports),
        "student_reports": student_reports,
        "elevated_student_reports": elevated_reports,
        "course_reports": course_reports,
        "non_diagnostic_disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Markdown Evidence Reporting
# ---------------------------------------------------------------------------


def generate_student_root_cause_markdown(report: StudentRootCauseReport) -> str:
    """Generate a clean markdown report of root cause investigation for a student.

    Args:
        report: StudentRootCauseReport object.

    Returns:
        Formatted markdown document string.
    """
    lines = [
        f"# Academic Root Cause Analysis: Student {report.student_id}",
        "",
        f"**Review Indicator Level**: `{report.overall_review_indicator}`  ",
        f"**Primary Contributing Factor**: {report.primary_factor or 'None (Metrics within expected range)'}  ",
        f"**Secondary Factors**: {', '.join(report.secondary_factors) if report.secondary_factors else 'None'}  ",
        f"**Analysis Generated**: {report.analysis_timestamp}",
        "",
        "---",
        "",
        f"> **Ethical Safeguard & Non-Diagnostic Notice**  \n> {report.non_diagnostic_disclaimer}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        report.factual_summary,
        "",
        "## Empirical Evidence Table",
        "",
        "| Category | Metric | Observed | Benchmark | Delta | Severity | Factual Finding |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    if report.all_evidence:
        for ev in report.all_evidence:
            delta_str = f"{ev.difference:+.1f}" if ev.difference is not None else "N/A"
            lines.append(
                f"| {ev.category.capitalize()} | `{ev.metric_name}` | {ev.observed_value} | "
                f"{ev.benchmark_value} | {delta_str} | **{ev.severity}** | {ev.finding} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | No concerning evidence items detected |")

    lines.append("")
    lines.append("## Category Breakdowns")
    lines.append("")

    for cat_name, contrib in report.category_evaluations.items():
        lines.append(f"### {contrib.factor_title} (`{contrib.severity}`)")
        lines.append(f"- **Observed Pattern**: {contrib.observed_pattern}")
        lines.append(f"- **Evidence Weight**: {contrib.score_weight:.1f}")
        for ev in contrib.evidence_items:
            lines.append(f"  - *{ev.metric_name}*: {ev.finding}")
        lines.append("")

    lines.append("## Supportive Advising Recommendations")
    lines.append("")
    for idx, rec in enumerate(report.supportive_recommendations, 1):
        lines.append(f"{idx}. {rec}")
    lines.append("")

    return "\n".join(lines)


def generate_course_root_cause_markdown(report: CourseRootCauseReport) -> str:
    """Generate a clean markdown report of root cause investigation for a course.

    Args:
        report: CourseRootCauseReport object.

    Returns:
        Formatted markdown document string.
    """
    lines = [
        f"# Academic Root Cause Analysis: Course {report.course_id}",
        "",
        f"**Course Review Indicator**: `{report.course_review_indicator}`  ",
        f"**Enrolled Students**: {report.total_students}  ",
        f"**Students Requiring Review**: {report.elevated_students_count} ({report.elevated_students_percentage:.1f}%)  ",
        f"**Primary Bottleneck**: {report.primary_bottleneck or 'None identified'}  ",
        f"**Secondary Bottlenecks**: {', '.join(report.secondary_bottlenecks) if report.secondary_bottlenecks else 'None'}  ",
        f"**Analysis Generated**: {report.analysis_timestamp}",
        "",
        "---",
        "",
        f"> **Ethical Safeguard & Non-Diagnostic Notice**  \n> {report.non_diagnostic_disclaimer}",
        "",
        "---",
        "",
        "## Course Factual Summary",
        "",
        report.factual_summary,
        "",
        "## Cohort Empirical Evidence Table",
        "",
        "| Dimension | Metric | Observed | Benchmark | Delta | Severity | Factual Finding |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    if report.all_evidence:
        for ev in report.all_evidence:
            delta_str = f"{ev.difference:+.1f}" if ev.difference is not None else "N/A"
            lines.append(
                f"| {ev.category.capitalize()} | `{ev.metric_name}` | {ev.observed_value} | "
                f"{ev.benchmark_value} | {delta_str} | **{ev.severity}** | {ev.finding} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | No cohort bottlenecks detected |")

    lines.append("")
    lines.append("## Curriculum & Pedagogical Recommendations")
    lines.append("")
    for idx, rec in enumerate(report.curriculum_recommendations, 1):
        lines.append(f"{idx}. {rec}")
    lines.append("")

    return "\n".join(lines)


def get_root_cause_thresholds() -> Dict[str, float]:
    """Get a copy of the default root cause analytical thresholds.

    Returns:
        Dictionary mapping threshold names to numeric limits.
    """
    return ROOT_CAUSE_THRESHOLDS.copy()
