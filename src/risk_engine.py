"""Explainable Academic Risk Engine Module.

Creates a deterministic rule-based risk engine for academic support.

Signals:
- attendance below configured threshold
- low assignment completion
- low exam average
- declining recent performance
- repeated missing/late submissions

Risk Levels: LOW, MODERATE, ELEVATED

IMPORTANT: This is an academic-support indicator, NOT a definitive prediction.
Thresholds are configurable and documented as project assumptions.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.risk_engine")

# PROJECT ASSUMPTIONS - Configurable Thresholds
# These thresholds can be adjusted based on institutional requirements
RISK_THRESHOLDS = {
    # Attendance thresholds
    "attendance_low_threshold": 70.0,  # Below this is considered low attendance
    "attendance_critical_threshold": 50.0,  # Below this is critical

    # Assignment completion thresholds
    "completion_low_threshold": 70.0,  # Below this is considered low completion
    "completion_critical_threshold": 50.0,  # Below this is critical

    # Exam performance thresholds
    "exam_low_threshold": 60.0,  # Below this is considered low performance
    "exam_critical_threshold": 40.0,  # Below this is critical

    # Trend thresholds (negative values indicate decline)
    "trend_decline_threshold": -10.0,  # Below this is considered declining
    "trend_critical_decline": -20.0,  # Below this is critical decline

    # Submission thresholds
    "missing_submission_threshold": 3,  # Number of missing submissions to flag
    "late_submission_threshold": 3,  # Number of late submissions to flag
    "late_submission_rate_threshold": 0.3,  # 30% late submissions is concerning
}

# Risk level definitions
RISK_LEVELS = {
    "LOW": {
        "description": "Student is performing adequately with minimal concerns",
        "signal_count_range": (0, 1),  # 0-1 concerning signals
        "color": "green",
    },
    "MODERATE": {
        "description": "Student shows some areas of concern that may benefit from support",
        "signal_count_range": (2, 3),  # 2-3 concerning signals
        "color": "yellow",
    },
    "ELEVATED": {
        "description": "Student shows multiple significant concerns requiring immediate attention",
        "signal_count_range": (4, 99),  # 4+ concerning signals
        "color": "red",
    },
}


@dataclass
class RiskSignal:
    """A single risk signal detected for a student."""

    signal_type: str
    severity: str  # "low", "moderate", "high"
    value: float
    threshold: float
    description: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a student."""

    student_id: Any
    risk_level: str  # "LOW", "MODERATE", "ELEVATED"
    risk_score: int  # Number of concerning signals
    signals: List[RiskSignal]
    reasons: List[str]
    recommendations: List[str]
    assessment_date: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "student_id": self.student_id,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "signals": [signal.to_dict() for signal in self.signals],
            "reasons": self.reasons,
            "recommendations": self.recommendations,
            "assessment_date": self.assessment_date,
            "disclaimer": self.disclaimer,
        }


def check_attendance_risk(
    attendance_percentage: Optional[float],
    thresholds: Dict[str, float] = None,
) -> Optional[RiskSignal]:
    """Check if attendance is below threshold.

    Args:
        attendance_percentage: Student's attendance percentage.
        thresholds: Optional custom thresholds.

    Returns:
        RiskSignal if risk detected, None otherwise.
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    if attendance_percentage is None or pd.isna(attendance_percentage):
        return None

    if attendance_percentage < thresholds["attendance_critical_threshold"]:
        return RiskSignal(
            signal_type="low_attendance",
            severity="high",
            value=attendance_percentage,
            threshold=thresholds["attendance_critical_threshold"],
            description=f"Attendance critically low ({attendance_percentage:.1f}% < {thresholds['attendance_critical_threshold']}%)",
            recommendation="Immediate intervention required - discuss attendance barriers with student",
        )
    elif attendance_percentage < thresholds["attendance_low_threshold"]:
        return RiskSignal(
            signal_type="low_attendance",
            severity="moderate",
            value=attendance_percentage,
            threshold=thresholds["attendance_low_threshold"],
            description=f"Attendance below threshold ({attendance_percentage:.1f}% < {thresholds['attendance_low_threshold']}%)",
            recommendation="Monitor attendance and provide support resources",
        )

    return None


def check_completion_risk(
    completion_rate: Optional[float],
    thresholds: Dict[str, float] = None,
) -> Optional[RiskSignal]:
    """Check if assignment completion is below threshold.

    Args:
        completion_rate: Student's assignment completion rate.
        thresholds: Optional custom thresholds.

    Returns:
        RiskSignal if risk detected, None otherwise.
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    if completion_rate is None or pd.isna(completion_rate):
        return None

    if completion_rate < thresholds["completion_critical_threshold"]:
        return RiskSignal(
            signal_type="low_completion",
            severity="high",
            value=completion_rate,
            threshold=thresholds["completion_critical_threshold"],
            description=f"Assignment completion critically low ({completion_rate:.1f}% < {thresholds['completion_critical_threshold']}%)",
            recommendation="Urgent - identify barriers to assignment completion",
        )
    elif completion_rate < thresholds["completion_low_threshold"]:
        return RiskSignal(
            signal_type="low_completion",
            severity="moderate",
            value=completion_rate,
            threshold=thresholds["completion_low_threshold"],
            description=f"Assignment completion below threshold ({completion_rate:.1f}% < {thresholds['completion_low_threshold']}%)",
            recommendation="Provide assignment support and time management resources",
        )

    return None


def check_exam_risk(
    exam_score: Optional[float],
    thresholds: Dict[str, float] = None,
) -> Optional[RiskSignal]:
    """Check if exam performance is below threshold.

    Args:
        exam_score: Student's average exam score.
        thresholds: Optional custom thresholds.

    Returns:
        RiskSignal if risk detected, None otherwise.
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    if exam_score is None or pd.isna(exam_score):
        return None

    if exam_score < thresholds["exam_critical_threshold"]:
        return RiskSignal(
            signal_type="low_exam_score",
            severity="high",
            value=exam_score,
            threshold=thresholds["exam_critical_threshold"],
            description=f"Exam performance critically low ({exam_score:.1f} < {thresholds['exam_critical_threshold']})",
            recommendation="Provide academic support and exam preparation resources",
        )
    elif exam_score < thresholds["exam_low_threshold"]:
        return RiskSignal(
            signal_type="low_exam_score",
            severity="moderate",
            value=exam_score,
            threshold=thresholds["exam_low_threshold"],
            description=f"Exam performance below threshold ({exam_score:.1f} < {thresholds['exam_low_threshold']})",
            recommendation="Monitor exam performance and offer tutoring support",
        )

    return None


def check_declining_trend_risk(
    trend_value: Optional[float],
    trend_name: str,
    thresholds: Dict[str, float] = None,
) -> Optional[RiskSignal]:
    """Check if recent performance is declining.

    Args:
        trend_value: Trend value (negative indicates decline).
        trend_name: Name of the trend (e.g., "attendance_trend").
        thresholds: Optional custom thresholds.

    Returns:
        RiskSignal if risk detected, None otherwise.
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    if trend_value is None or pd.isna(trend_value):
        return None

    if trend_value < thresholds["trend_critical_decline"]:
        return RiskSignal(
            signal_type="declining_performance",
            severity="high",
            value=trend_value,
            threshold=thresholds["trend_critical_decline"],
            description=f"Critical decline in {trend_name} ({trend_value:.1f}% < {thresholds['trend_critical_decline']}%)",
            recommendation=f"Immediate intervention - investigate causes of declining {trend_name}",
        )
    elif trend_value < thresholds["trend_decline_threshold"]:
        return RiskSignal(
            signal_type="declining_performance",
            severity="moderate",
            value=trend_value,
            threshold=thresholds["trend_decline_threshold"],
            description=f"Declining trend in {trend_name} ({trend_value:.1f}% < {thresholds['trend_decline_threshold']}%)",
            recommendation=f"Monitor {trend_name} and provide early support",
        )

    return None


def check_submission_risk(
    missing_count: Optional[int] = None,
    late_count: Optional[int] = None,
    late_rate: Optional[float] = None,
    thresholds: Dict[str, float] = None,
) -> List[RiskSignal]:
    """Check for repeated missing or late submissions.

    Args:
        missing_count: Number of missing submissions.
        late_count: Number of late submissions.
        late_rate: Rate of late submissions (0-1).
        thresholds: Optional custom thresholds.

    Returns:
        List of RiskSignals detected.
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    signals: List[RiskSignal] = []

    # Check missing submissions
    if missing_count is not None and missing_count >= thresholds["missing_submission_threshold"]:
        severity = "high" if missing_count >= thresholds["missing_submission_threshold"] * 2 else "moderate"
        signals.append(
            RiskSignal(
                signal_type="missing_submissions",
                severity=severity,
                value=float(missing_count),
                threshold=thresholds["missing_submission_threshold"],
                description=f"Multiple missing submissions ({missing_count} >= {thresholds['missing_submission_threshold']})",
                recommendation="Address barriers to submission completion",
            )
        )

    # Check late submissions count
    if late_count is not None and late_count >= thresholds["late_submission_threshold"]:
        severity = "high" if late_count >= thresholds["late_submission_threshold"] * 2 else "moderate"
        signals.append(
            RiskSignal(
                signal_type="late_submissions",
                severity=severity,
                value=float(late_count),
                threshold=thresholds["late_submission_threshold"],
                description=f"Multiple late submissions ({late_count} >= {thresholds['late_submission_threshold']})",
                recommendation="Address time management and planning",
            )
        )

    # Check late submission rate
    if late_rate is not None and late_rate >= thresholds["late_submission_rate_threshold"]:
        severity = "high" if late_rate >= thresholds["late_submission_rate_threshold"] * 1.5 else "moderate"
        signals.append(
            RiskSignal(
                signal_type="high_late_rate",
                severity=severity,
                value=late_rate,
                threshold=thresholds["late_submission_rate_threshold"],
                description=f"High late submission rate ({late_rate:.1%} >= {thresholds['late_submission_rate_threshold']:.0%})",
                recommendation="Improve time management and deadline awareness",
            )
        )

    return signals


def determine_risk_level(
    signal_count: int,
    high_severity_count: int,
) -> str:
    """Determine risk level based on signals.

    Args:
        signal_count: Total number of concerning signals.
        high_severity_count: Number of high-severity signals.

    Returns:
        Risk level: "LOW", "MODERATE", or "ELEVATED".
    """
    # High-severity signals elevate risk level
    if high_severity_count >= 2:
        return "ELEVATED"
    elif high_severity_count == 1 and signal_count >= 2:
        return "ELEVATED"
    elif signal_count >= 4:
        return "ELEVATED"
    elif signal_count >= 2:
        return "MODERATE"
    elif signal_count == 1:
        return "MODERATE"
    else:
        return "LOW"


def assess_student_risk(
    student_id: Any,
    attendance_percentage: Optional[float] = None,
    completion_rate: Optional[float] = None,
    exam_score: Optional[float] = None,
    attendance_trend: Optional[float] = None,
    assignment_trend: Optional[float] = None,
    exam_trend: Optional[float] = None,
    missing_submissions: Optional[int] = None,
    late_submissions: Optional[int] = None,
    late_submission_rate: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> RiskAssessment:
    """Perform complete risk assessment for a student.

    Args:
        student_id: Student identifier.
        attendance_percentage: Attendance percentage.
        completion_rate: Assignment completion rate.
        exam_score: Average exam score.
        attendance_trend: Attendance trend (negative = declining).
        assignment_trend: Assignment trend (negative = declining).
        exam_trend: Exam trend (negative = declining).
        missing_submissions: Count of missing submissions.
        late_submissions: Count of late submissions.
        late_submission_rate: Rate of late submissions (0-1).
        thresholds: Optional custom thresholds.

    Returns:
        RiskAssessment with complete risk evaluation.
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    signals: List[RiskSignal] = []

    # Check attendance
    attendance_signal = check_attendance_risk(attendance_percentage, thresholds)
    if attendance_signal:
        signals.append(attendance_signal)

    # Check completion
    completion_signal = check_completion_risk(completion_rate, thresholds)
    if completion_signal:
        signals.append(completion_signal)

    # Check exam performance
    exam_signal = check_exam_risk(exam_score, thresholds)
    if exam_signal:
        signals.append(exam_signal)

    # Check trends
    if attendance_trend is not None:
        trend_signal = check_declining_trend_risk(attendance_trend, "attendance", thresholds)
        if trend_signal:
            signals.append(trend_signal)

    if assignment_trend is not None:
        trend_signal = check_declining_trend_risk(assignment_trend, "assignment completion", thresholds)
        if trend_signal:
            signals.append(trend_signal)

    if exam_trend is not None:
        trend_signal = check_declining_trend_risk(exam_trend, "exam performance", thresholds)
        if trend_signal:
            signals.append(trend_signal)

    # Check submissions
    submission_signals = check_submission_risk(
        missing_submissions,
        late_submissions,
        late_submission_rate,
        thresholds,
    )
    signals.extend(submission_signals)

    # Determine risk level
    high_severity_count = sum(1 for s in signals if s.severity == "high")
    risk_level = determine_risk_level(len(signals), high_severity_count)

    # Generate reasons
    reasons = [signal.description for signal in signals]
    if not reasons:
        reasons.append("No concerning signals detected")

    # Generate recommendations
    recommendations = [signal.recommendation for signal in signals]
    if not recommendations:
        recommendations.append("Continue current academic support approach")

    # Disclaimer
    disclaimer = (
        "IMPORTANT: This risk assessment is an academic-support indicator based on observable patterns. "
        "It is NOT a definitive prediction of student outcomes. Risk levels should be used to inform "
        "support decisions, not to label or stigmatize students. Individual circumstances, external factors, "
        "and qualitative context should always be considered alongside these quantitative signals."
    )

    logger.info(
        "Risk assessment for student %s: %s (%d signals)",
        student_id,
        risk_level,
        len(signals),
    )

    return RiskAssessment(
        student_id=student_id,
        risk_level=risk_level,
        risk_score=len(signals),
        signals=signals,
        reasons=reasons,
        recommendations=recommendations,
        assessment_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        disclaimer=disclaimer,
    )


def batch_assess_risk(
    student_features_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[RiskAssessment]:
    """Perform risk assessment for multiple students.

    Args:
        student_features_df: DataFrame with student features.
        thresholds: Optional custom thresholds.

    Returns:
        List of RiskAssessment objects.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if "student_id" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'student_id' column")

    assessments: List[RiskAssessment] = []

    for _, row in student_features_df.iterrows():
        assessment = assess_student_risk(
            student_id=row["student_id"],
            attendance_percentage=row.get("attendance_percentage"),
            completion_rate=row.get("assignment_completion_rate"),
            exam_score=row.get("average_exam_score"),
            attendance_trend=row.get("attendance_trend"),
            assignment_trend=row.get("assignment_trend"),
            exam_trend=row.get("exam_trend"),
            missing_submissions=row.get("missing_submission_count"),
            late_submissions=row.get("late_submission_count"),
            late_submission_rate=row.get("late_submission_rate"),
            thresholds=thresholds,
        )
        assessments.append(assessment)

    logger.info("Batch risk assessment completed for %d students", len(assessments))

    return assessments


def get_risk_thresholds() -> Dict[str, float]:
    """Get current risk thresholds.

    Returns:
        Dictionary of threshold values.
    """
    return RISK_THRESHOLDS.copy()


def get_risk_level_descriptions() -> Dict[str, Dict[str, Any]]:
    """Get risk level descriptions.

    Returns:
        Dictionary mapping risk levels to their descriptions.
    """
    return {
        level: {
            "description": info["description"],
            "signal_count_range": info["signal_count_range"],
            "color": info["color"],
        }
        for level, info in RISK_LEVELS.items()
    }


def generate_risk_report_markdown(
    assessment: RiskAssessment,
) -> str:
    """Generate a markdown summary of risk assessment.

    Args:
        assessment: RiskAssessment object.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic Risk Assessment Report",
        "",
        f"**Student ID**: {assessment.student_id}",
        f"**Risk Level**: {assessment.risk_level}",
        f"**Risk Score**: {assessment.risk_score} (concerning signals)",
        f"**Assessment Date**: {assessment.assessment_date}",
        "",
        "---",
        "",
        f"> {assessment.disclaimer}",
        "",
        "---",
        "",
        "## Risk Signals",
        "",
    ]

    if assessment.signals:
        for signal in assessment.signals:
            lines.append(f"### {signal.signal_type.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **Severity**: {signal.severity.upper()}")
            lines.append(f"- **Value**: {signal.value}")
            lines.append(f"- **Threshold**: {signal.threshold}")
            lines.append(f"- **Description**: {signal.description}")
            lines.append(f"- **Recommendation**: {signal.recommendation}")
            lines.append("")
            lines.append("---")
            lines.append("")
    else:
        lines.append("No concerning signals detected.")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("**Reasons for Risk Level**:")
    for reason in assessment.reasons:
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("**Recommendations**:")
    for rec in assessment.recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)
