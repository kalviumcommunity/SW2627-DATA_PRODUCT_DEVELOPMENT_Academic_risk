"""Academic KPI Framework Module.

Defines reusable Key Performance Indicators (KPIs) for academic analytics:
- Total Students
- Average Attendance
- Assignment Completion Rate
- Average Exam Score
- Students Requiring Review
- Low Attendance Rate
- Missing Submission Rate
- Engagement Rate
- Course Performance

Each KPI includes:
- formula
- data source
- meaning
- limitations
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.kpi_framework")


@dataclass
class KPIDefinition:
    """Definition of a KPI with metadata."""

    name: str
    formula: str
    data_source: str
    meaning: str
    limitations: List[str]
    category: str  # 'enrollment', 'engagement', 'performance', 'risk'
    unit: Optional[str] = None
    is_percentage: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class KPIResult:
    """Result of a KPI calculation."""

    kpi_name: str
    value: Optional[float]
    unit: Optional[str]
    is_percentage: bool
    calculation_date: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


# KPI Definitions
KPI_DEFINITIONS: Dict[str, KPIDefinition] = {
    "total_students": KPIDefinition(
        name="Total Students",
        formula="COUNT(DISTINCT student_id)",
        data_source="enrollments table",
        meaning="Total number of unique students enrolled in the course/program",
        limitations=[
            "Does not account for withdrawals or transfers",
            "May include inactive students if not filtered",
            "Counts all enrollments regardless of status",
        ],
        category="enrollment",
        unit="count",
        is_percentage=False,
    ),
    "average_attendance": KPIDefinition(
        name="Average Attendance",
        formula="AVG(attendance_percentage)",
        data_source="student_features table (calculated from attendance records)",
        meaning="Average attendance percentage across all students",
        limitations=[
            "Sensitive to outliers (extremely high or low attendance)",
            "Does not reflect attendance patterns over time",
            "May be skewed by students with few attendance records",
            "Assumes equal importance of all sessions",
        ],
        category="engagement",
        unit="percentage",
        is_percentage=True,
    ),
    "assignment_completion_rate": KPIDefinition(
        name="Assignment Completion Rate",
        formula="AVG(assignment_completion_rate)",
        data_source="student_features table (calculated from submissions)",
        meaning="Average rate of assignment completion across all students",
        limitations=[
            "Does not account for assignment difficulty or weight",
            "Late submissions may be counted as complete",
            "Quality of work not considered",
            "May vary significantly by course/instructor",
        ],
        category="engagement",
        unit="percentage",
        is_percentage=True,
    ),
    "average_exam_score": KPIDefinition(
        name="Average Exam Score",
        formula="AVG(average_exam_score)",
        data_source="student_features table (calculated from exam records)",
        meaning="Average exam score across all students",
        limitations=[
            "Exam difficulty not normalized",
            "May not reflect continuous assessment performance",
            "Outliers can significantly impact average",
            "Different exams may have different scoring scales",
        ],
        category="performance",
        unit="score",
        is_percentage=False,
    ),
    "students_requiring_review": KPIDefinition(
        name="Students Requiring Review",
        formula="COUNT(students WHERE attendance < 70% OR completion < 70% OR exam < 50%)",
        data_source="student_features table",
        meaning="Number of students falling below performance thresholds",
        limitations=[
            "Thresholds are arbitrary and may not suit all contexts",
            "Does not identify specific intervention needs",
            "May miss students with borderline issues",
            "Does not account for improvement trends",
        ],
        category="risk",
        unit="count",
        is_percentage=False,
    ),
    "low_attendance_rate": KPIDefinition(
        name="Low Attendance Rate",
        formula="COUNT(students WHERE attendance < 70%) / total_students * 100",
        data_source="student_features table",
        meaning="Percentage of students with attendance below 70%",
        limitations=[
            "70% threshold is arbitrary",
            "Does not distinguish between slightly below and severely below",
            "May not account for excused absences",
            "Does not reflect recent attendance trends",
        ],
        category="risk",
        unit="percentage",
        is_percentage=True,
    ),
    "missing_submission_rate": KPIDefinition(
        name="Missing Submission Rate",
        formula="COUNT(missing_submissions) / total_assignments * 100",
        data_source="submissions table",
        meaning="Percentage of assignments not submitted",
        limitations=[
            "Does not account for late submissions",
            "May not reflect assignment difficulty",
            "Does not distinguish between excused and unexcused",
            "May be skewed by assignment count",
        ],
        category="risk",
        unit="percentage",
        is_percentage=True,
    ),
    "engagement_rate": KPIDefinition(
        name="Engagement Rate",
        formula="(AVG(attendance) + AVG(completion)) / 2",
        data_source="student_features table",
        meaning="Combined measure of attendance and assignment engagement",
        limitations=[
            "Equal weighting may not reflect actual importance",
            "Does not include exam participation",
            "May mask issues in one area by strong performance in another",
            "Simple average may not capture complex engagement patterns",
        ],
        category="engagement",
        unit="percentage",
        is_percentage=True,
    ),
    "course_performance": KPIDefinition(
        name="Course Performance",
        formula="(AVG(assignment_completion_rate) * 0.4) + (AVG(exam_score) * 0.6)",
        data_source="student_features table",
        meaning="Weighted performance metric combining assignments and exams",
        limitations=[
            "Weights (40/60) are arbitrary and may not suit all courses",
            "Does not account for attendance",
            "May not reflect actual learning outcomes",
            "Different courses may require different weightings",
        ],
        category="performance",
        unit="score",
        is_percentage=False,
    ),
}


def calculate_total_students(
    enrollments_df: pd.DataFrame,
    student_id_column: str = "student_id",
) -> KPIResult:
    """Calculate Total Students KPI.

    Args:
        enrollments_df: DataFrame with enrollment records.
        student_id_column: Name of the student ID column.

    Returns:
        KPIResult with total student count.

    Raises:
        DataValidationError: If student_id column is missing.
    """
    if student_id_column not in enrollments_df.columns:
        raise DataValidationError(f"enrollments_df must contain '{student_id_column}' column")

    value = enrollments_df[student_id_column].nunique()
    definition = KPI_DEFINITIONS["total_students"]

    logger.info("KPI: Total Students = %d", value)

    return KPIResult(
        kpi_name=definition.name,
        value=float(value),
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={"formula": definition.formula, "data_source": definition.data_source},
    )


def calculate_average_attendance(
    student_features_df: pd.DataFrame,
    attendance_column: str = "attendance_percentage",
) -> KPIResult:
    """Calculate Average Attendance KPI.

    Args:
        student_features_df: DataFrame with student features.
        attendance_column: Name of the attendance column.

    Returns:
        KPIResult with average attendance.

    Raises:
        DataValidationError: If attendance column is missing.
    """
    if attendance_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{attendance_column}' column")

    valid_values = student_features_df[attendance_column].dropna()
    value = valid_values.mean() if not valid_values.empty else None
    definition = KPI_DEFINITIONS["average_attendance"]

    logger.info("KPI: Average Attendance = %.2f%%", value if value is not None else 0.0)

    return KPIResult(
        kpi_name=definition.name,
        value=round(float(value), 2) if value is not None else None,
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "student_count": len(valid_values),
        },
    )


def calculate_assignment_completion_rate(
    student_features_df: pd.DataFrame,
    completion_column: str = "assignment_completion_rate",
) -> KPIResult:
    """Calculate Assignment Completion Rate KPI.

    Args:
        student_features_df: DataFrame with student features.
        completion_column: Name of the completion column.

    Returns:
        KPIResult with average completion rate.

    Raises:
        DataValidationError: If completion column is missing.
    """
    if completion_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{completion_column}' column")

    valid_values = student_features_df[completion_column].dropna()
    value = valid_values.mean() if not valid_values.empty else None
    definition = KPI_DEFINITIONS["assignment_completion_rate"]

    logger.info("KPI: Assignment Completion Rate = %.2f%%", value if value is not None else 0.0)

    return KPIResult(
        kpi_name=definition.name,
        value=round(float(value), 2) if value is not None else None,
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "student_count": len(valid_values),
        },
    )


def calculate_average_exam_score(
    student_features_df: pd.DataFrame,
    exam_column: str = "average_exam_score",
) -> KPIResult:
    """Calculate Average Exam Score KPI.

    Args:
        student_features_df: DataFrame with student features.
        exam_column: Name of the exam score column.

    Returns:
        KPIResult with average exam score.

    Raises:
        DataValidationError: If exam column is missing.
    """
    if exam_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{exam_column}' column")

    valid_values = student_features_df[exam_column].dropna()
    value = valid_values.mean() if not valid_values.empty else None
    definition = KPI_DEFINITIONS["average_exam_score"]

    logger.info("KPI: Average Exam Score = %.2f", value if value is not None else 0.0)

    return KPIResult(
        kpi_name=definition.name,
        value=round(float(value), 2) if value is not None else None,
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "student_count": len(valid_values),
        },
    )


def calculate_students_requiring_review(
    student_features_df: pd.DataFrame,
    attendance_column: str = "attendance_percentage",
    completion_column: str = "assignment_completion_rate",
    exam_column: str = "average_exam_score",
    attendance_threshold: float = 70.0,
    completion_threshold: float = 70.0,
    exam_threshold: float = 50.0,
) -> KPIResult:
    """Calculate Students Requiring Review KPI.

    Args:
        student_features_df: DataFrame with student features.
        attendance_column: Name of the attendance column.
        completion_column: Name of the completion column.
        exam_column: Name of the exam score column.
        attendance_threshold: Attendance threshold for review.
        completion_threshold: Completion threshold for review.
        exam_threshold: Exam score threshold for review.

    Returns:
        KPIResult with count of students requiring review.

    Raises:
        DataValidationError: If required columns are missing.
    """
    required_cols = [attendance_column, completion_column, exam_column]
    for col in required_cols:
        if col not in student_features_df.columns:
            raise DataValidationError(f"student_features_df must contain '{col}' column")

    # Identify students below any threshold
    requires_review = student_features_df[
        (student_features_df[attendance_column] < attendance_threshold) |
        (student_features_df[completion_column] < completion_threshold) |
        (student_features_df[exam_column] < exam_threshold)
    ]

    value = len(requires_review)
    definition = KPI_DEFINITIONS["students_requiring_review"]

    logger.info("KPI: Students Requiring Review = %d", value)

    return KPIResult(
        kpi_name=definition.name,
        value=float(value),
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "thresholds": {
                "attendance": attendance_threshold,
                "completion": completion_threshold,
                "exam": exam_threshold,
            },
        },
    )


def calculate_low_attendance_rate(
    student_features_df: pd.DataFrame,
    attendance_column: str = "attendance_percentage",
    threshold: float = 70.0,
) -> KPIResult:
    """Calculate Low Attendance Rate KPI.

    Args:
        student_features_df: DataFrame with student features.
        attendance_column: Name of the attendance column.
        threshold: Attendance threshold for low attendance.

    Returns:
        KPIResult with low attendance rate.

    Raises:
        DataValidationError: If attendance column is missing.
    """
    if attendance_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{attendance_column}' column")

    total_students = len(student_features_df)
    if total_students == 0:
        value = 0.0
    else:
        low_attendance = student_features_df[student_features_df[attendance_column] < threshold]
        value = (len(low_attendance) / total_students) * 100.0

    definition = KPI_DEFINITIONS["low_attendance_rate"]

    logger.info("KPI: Low Attendance Rate = %.2f%%", value)

    return KPIResult(
        kpi_name=definition.name,
        value=round(value, 2),
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "threshold": threshold,
            "total_students": total_students,
        },
    )


def calculate_missing_submission_rate(
    submissions_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    student_id_column: str = "student_id",
    assignment_id_column: str = "assignment_id",
) -> KPIResult:
    """Calculate Missing Submission Rate KPI.

    Args:
        submissions_df: DataFrame with submission records.
        assignments_df: DataFrame with assignment records.
        student_id_column: Name of the student ID column.
        assignment_id_column: Name of the assignment ID column.

    Returns:
        KPIResult with missing submission rate.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if student_id_column not in submissions_df.columns:
        raise DataValidationError(f"submissions_df must contain '{student_id_column}' column")
    if assignment_id_column not in assignments_df.columns:
        raise DataValidationError(f"assignments_df must contain '{assignment_id_column}' column")

    total_assignments = len(assignments_df)
    if total_assignments == 0:
        value = 0.0
    else:
        # Count unique student-assignment pairs
        expected_submissions = len(assignments_df)  # Assuming one per student per assignment
        actual_submissions = len(submissions_df)
        missing = expected_submissions - actual_submissions
        value = (missing / expected_submissions) * 100.0 if expected_submissions > 0 else 0.0

    definition = KPI_DEFINITIONS["missing_submission_rate"]

    logger.info("KPI: Missing Submission Rate = %.2f%%", value)

    return KPIResult(
        kpi_name=definition.name,
        value=round(value, 2),
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "total_assignments": total_assignments,
        },
    )


def calculate_engagement_rate(
    student_features_df: pd.DataFrame,
    attendance_column: str = "attendance_percentage",
    completion_column: str = "assignment_completion_rate",
) -> KPIResult:
    """Calculate Engagement Rate KPI.

    Args:
        student_features_df: DataFrame with student features.
        attendance_column: Name of the attendance column.
        completion_column: Name of the completion column.

    Returns:
        KPIResult with engagement rate.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if attendance_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{attendance_column}' column")
    if completion_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{completion_column}' column")

    avg_attendance = student_features_df[attendance_column].mean()
    avg_completion = student_features_df[completion_column].mean()

    value = (avg_attendance + avg_completion) / 2.0
    definition = KPI_DEFINITIONS["engagement_rate"]

    logger.info("KPI: Engagement Rate = %.2f%%", value if not pd.isna(value) else 0.0)

    return KPIResult(
        kpi_name=definition.name,
        value=round(float(value), 2) if not pd.isna(value) else None,
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "average_attendance": round(float(avg_attendance), 2) if not pd.isna(avg_attendance) else None,
            "average_completion": round(float(avg_completion), 2) if not pd.isna(avg_completion) else None,
        },
    )


def calculate_course_performance(
    student_features_df: pd.DataFrame,
    completion_column: str = "assignment_completion_rate",
    exam_column: str = "average_exam_score",
    assignment_weight: float = 0.4,
    exam_weight: float = 0.6,
) -> KPIResult:
    """Calculate Course Performance KPI.

    Args:
        student_features_df: DataFrame with student features.
        completion_column: Name of the completion column.
        exam_column: Name of the exam score column.
        assignment_weight: Weight for assignment completion (default 0.4).
        exam_weight: Weight for exam score (default 0.6).

    Returns:
        KPIResult with course performance score.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if completion_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{completion_column}' column")
    if exam_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{exam_column}' column")

    avg_completion = student_features_df[completion_column].mean()
    avg_exam = student_features_df[exam_column].mean()

    value = (avg_completion * assignment_weight) + (avg_exam * exam_weight)
    definition = KPI_DEFINITIONS["course_performance"]

    logger.info("KPI: Course Performance = %.2f", value if not pd.isna(value) else 0.0)

    return KPIResult(
        kpi_name=definition.name,
        value=round(float(value), 2) if not pd.isna(value) else None,
        unit=definition.unit,
        is_percentage=definition.is_percentage,
        calculation_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "formula": definition.formula,
            "data_source": definition.data_source,
            "assignment_weight": assignment_weight,
            "exam_weight": exam_weight,
            "average_completion": round(float(avg_completion), 2) if not pd.isna(avg_completion) else None,
            "average_exam": round(float(avg_exam), 2) if not pd.isna(avg_exam) else None,
        },
    )


def get_kpi_definition(kpi_name: str) -> KPIDefinition:
    """Get the definition of a specific KPI.

    Args:
        kpi_name: Name of the KPI.

    Returns:
        KPIDefinition for the specified KPI.

    Raises:
        DataValidationError: If KPI name is invalid.
    """
    if kpi_name not in KPI_DEFINITIONS:
        raise DataValidationError(f"Invalid KPI name: {kpi_name}")

    return KPI_DEFINITIONS[kpi_name]


def get_all_kpi_definitions() -> Dict[str, KPIDefinition]:
    """Get all KPI definitions.

    Returns:
        Dictionary mapping KPI names to their definitions.
    """
    return KPI_DEFINITIONS.copy()


def generate_kpi_report_markdown(
    kpi_results: List[KPIResult],
) -> str:
    """Generate a markdown summary of KPI calculations.

    Args:
        kpi_results: List of KPIResult objects.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic KPI Report",
        "",
        f"**Report Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**KPIs Calculated**: {len(kpi_results)}",
        "",
        "## KPI Results",
        "",
    ]

    for result in kpi_results:
        definition = KPI_DEFINITIONS.get(result.kpi_name.lower().replace(" ", "_"))
        lines.append(f"### {result.kpi_name}")
        lines.append("")
        lines.append("**Value**")
        if result.value is not None:
            if result.is_percentage:
                lines.append(f"- {result.value}%")
            else:
                lines.append(f"- {result.value}")
        else:
            lines.append("- N/A")
        lines.append("")

        lines.append("**Definition**")
        if definition:
            lines.append(f"- **Formula**: {definition.formula}")
            lines.append(f"- **Data Source**: {definition.data_source}")
            lines.append(f"- **Meaning**: {definition.meaning}")
            lines.append(f"- **Category**: {definition.category}")
        lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## KPI Definitions and Limitations")
    lines.append("")

    for kpi_name, definition in KPI_DEFINITIONS.items():
        lines.append(f"### {definition.name}")
        lines.append("")
        lines.append(f"**Formula**: {definition.formula}")
        lines.append(f"**Data Source**: {definition.data_source}")
        lines.append(f"**Meaning**: {definition.meaning}")
        lines.append("")
        lines.append("**Limitations**:")
        for limitation in definition.limitations:
            lines.append(f"- {limitation}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
