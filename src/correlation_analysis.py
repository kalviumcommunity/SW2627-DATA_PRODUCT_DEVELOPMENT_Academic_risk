"""Academic Correlation Analysis Module.

Provides reusable analytical functions for analyzing relationships between:
- attendance and exam performance
- attendance and assignment completion
- assignment completion and exam scores
- recent attendance and risk indicators

IMPORTANT: Correlation does not imply causation. This module identifies statistical
relationships but does not make causal claims about academic outcomes.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.correlation_analysis")


@dataclass
class CorrelationResult:
    """Result of a correlation analysis between two variables."""

    variable_x: str
    variable_y: str
    correlation_coefficient: Optional[float]
    p_value: Optional[float]
    sample_size: int
    method: str  # 'pearson', 'spearman', 'kendall'
    interpretation: str
    strength: str  # 'negligible', 'weak', 'moderate', 'strong'
    direction: str  # 'positive', 'negative', 'none'
    confidence_interval: Optional[Dict[str, float]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class CorrelationMatrix:
    """Correlation matrix for multiple variables."""

    variables: List[str]
    correlation_matrix: List[List[Optional[float]]]
    p_value_matrix: List[List[Optional[float]]]
    method: str
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class CorrelationInsights:
    """Insights derived from correlation analysis."""

    correlation_pair: str
    statistical_significance: bool
    practical_significance: bool
    potential_confounders: List[str]
    limitations: List[str]
    recommendations: List[str]
    causation_warning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


def interpret_correlation_strength(
    coefficient: float,
) -> str:
    """Interpret the strength of a correlation coefficient.

    Args:
        coefficient: Correlation coefficient (typically -1 to 1).

    Returns:
        Strength category string.
    """
    abs_coeff = abs(coefficient)
    if abs_coeff < 0.1:
        return "negligible"
    elif abs_coeff < 0.3:
        return "weak"
    elif abs_coeff < 0.5:
        return "moderate"
    elif abs_coeff < 0.7:
        return "moderate_to_strong"
    else:
        return "strong"


def interpret_correlation_direction(
    coefficient: float,
) -> str:
    """Interpret the direction of a correlation coefficient.

    Args:
        coefficient: Correlation coefficient.

    Returns:
        Direction string.
    """
    if abs(coefficient) < 0.1:
        return "none"
    elif coefficient > 0:
        return "positive"
    else:
        return "negative"


def calculate_pearson_correlation(
    x: pd.Series,
    y: pd.Series,
) -> Tuple[float, float, int]:
    """Calculate Pearson correlation coefficient and p-value.

    Args:
        x: First variable series.
        y: Second variable series.

    Returns:
        Tuple of (correlation_coefficient, p_value, sample_size).
    """
    # Remove NaN values pairwise
    valid_mask = x.notna() & y.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) < 3:
        return np.nan, np.nan, len(x_clean)

    corr, p_value = stats.pearsonr(x_clean, y_clean)
    return float(corr), float(p_value), len(x_clean)


def calculate_spearman_correlation(
    x: pd.Series,
    y: pd.Series,
) -> Tuple[float, float, int]:
    """Calculate Spearman rank correlation coefficient and p-value.

    Args:
        x: First variable series.
        y: Second variable series.

    Returns:
        Tuple of (correlation_coefficient, p_value, sample_size).
    """
    # Remove NaN values pairwise
    valid_mask = x.notna() & y.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) < 3:
        return np.nan, np.nan, len(x_clean)

    corr, p_value = stats.spearmanr(x_clean, y_clean)
    return float(corr), float(p_value), len(x_clean)


def calculate_kendall_correlation(
    x: pd.Series,
    y: pd.Series,
) -> Tuple[float, float, int]:
    """Calculate Kendall's tau correlation coefficient and p-value.

    Args:
        x: First variable series.
        y: Second variable series.

    Returns:
        Tuple of (correlation_coefficient, p_value, sample_size).
    """
    # Remove NaN values pairwise
    valid_mask = x.notna() & y.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) < 3:
        return np.nan, np.nan, len(x_clean)

    corr, p_value = stats.kendalltau(x_clean, y_clean)
    return float(corr), float(p_value), len(x_clean)


def analyze_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str = "pearson",
    variable_x_name: str = "x",
    variable_y_name: str = "y",
) -> CorrelationResult:
    """Perform correlation analysis between two variables.

    Args:
        x: First variable series.
        y: Second variable series.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        variable_x_name: Name of x variable for reporting.
        variable_y_name: Name of y variable for reporting.

    Returns:
        CorrelationResult with analysis details.

    Raises:
        DataValidationError: If method is invalid.
    """
    if method not in ["pearson", "spearman", "kendall"]:
        raise DataValidationError(f"Invalid correlation method: {method}")

    if method == "pearson":
        corr, p_val, n = calculate_pearson_correlation(x, y)
    elif method == "spearman":
        corr, p_val, n = calculate_spearman_correlation(x, y)
    else:  # kendall
        corr, p_val, n = calculate_kendall_correlation(x, y)

    strength = interpret_correlation_strength(corr) if not np.isnan(corr) else "unknown"
    direction = interpret_correlation_direction(corr) if not np.isnan(corr) else "none"

    # Interpret statistical significance
    interpretation = ""
    notes = []

    if np.isnan(corr):
        interpretation = "Insufficient data for correlation analysis"
        notes.append(f"Only {n} valid paired observations")
    elif np.isnan(p_val):
        interpretation = "Could not calculate p-value"
    elif p_val < 0.001:
        interpretation = "Very strong statistical significance (p < 0.001)"
    elif p_val < 0.01:
        interpretation = "Strong statistical significance (p < 0.01)"
    elif p_val < 0.05:
        interpretation = "Statistically significant (p < 0.05)"
    elif p_val < 0.10:
        interpretation = "Marginal statistical significance (p < 0.10)"
    else:
        interpretation = "Not statistically significant"

    # Add practical significance note
    if not np.isnan(corr):
        if strength == "strong":
            notes.append(f"Strong {direction} relationship detected")
        elif strength == "moderate":
            notes.append(f"Moderate {direction} relationship detected")
        elif strength == "weak":
            notes.append(f"Weak {direction} relationship detected")
        else:
            notes.append("Negligible relationship detected")

    logger.info(
        "Correlation analysis: %s vs %s = %.3f (p=%.4f, n=%d, method=%s)",
        variable_x_name,
        variable_y_name,
        corr if not np.isnan(corr) else 0.0,
        p_val if not np.isnan(p_val) else 0.0,
        n,
        method,
    )

    return CorrelationResult(
        variable_x=variable_x_name,
        variable_y=variable_y_name,
        correlation_coefficient=round(float(corr), 3) if not np.isnan(corr) else None,
        p_value=round(float(p_val), 4) if not np.isnan(p_val) else None,
        sample_size=n,
        method=method,
        interpretation=interpretation,
        strength=strength,
        direction=direction,
        notes=notes,
    )


def analyze_attendance_exam_correlation(
    student_features_df: pd.DataFrame,
) -> CorrelationResult:
    """Analyze correlation between attendance and exam performance.

    Args:
        student_features_df: DataFrame with student features including
                           attendance_percentage and average_exam_score.

    Returns:
        CorrelationResult for attendance vs exam scores.
    """
    if "attendance_percentage" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'attendance_percentage'")
    if "average_exam_score" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'average_exam_score'")

    result = analyze_correlation(
        student_features_df["attendance_percentage"],
        student_features_df["average_exam_score"],
        method="pearson",
        variable_x_name="attendance_percentage",
        variable_y_name="average_exam_score",
    )

    logger.info("Attendance vs Exam correlation: r=%.3f", result.correlation_coefficient)
    return result


def analyze_attendance_completion_correlation(
    student_features_df: pd.DataFrame,
) -> CorrelationResult:
    """Analyze correlation between attendance and assignment completion.

    Args:
        student_features_df: DataFrame with student features including
                           attendance_percentage and assignment_completion_rate.

    Returns:
        CorrelationResult for attendance vs completion rate.
    """
    if "attendance_percentage" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'attendance_percentage'")
    if "assignment_completion_rate" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'assignment_completion_rate'")

    result = analyze_correlation(
        student_features_df["attendance_percentage"],
        student_features_df["assignment_completion_rate"],
        method="pearson",
        variable_x_name="attendance_percentage",
        variable_y_name="assignment_completion_rate",
    )

    logger.info("Attendance vs Completion correlation: r=%.3f", result.correlation_coefficient)
    return result


def analyze_completion_exam_correlation(
    student_features_df: pd.DataFrame,
) -> CorrelationResult:
    """Analyze correlation between assignment completion and exam scores.

    Args:
        student_features_df: DataFrame with student features including
                           assignment_completion_rate and average_exam_score.

    Returns:
        CorrelationResult for completion vs exam scores.
    """
    if "assignment_completion_rate" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'assignment_completion_rate'")
    if "average_exam_score" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'average_exam_score'")

    result = analyze_correlation(
        student_features_df["assignment_completion_rate"],
        student_features_df["average_exam_score"],
        method="pearson",
        variable_x_name="assignment_completion_rate",
        variable_y_name="average_exam_score",
    )

    logger.info("Completion vs Exam correlation: r=%.3f", result.correlation_coefficient)
    return result


def analyze_recent_attendance_risk_correlation(
    student_features_df: pd.DataFrame,
    risk_indicator_column: str = "attendance_trend",
) -> CorrelationResult:
    """Analyze correlation between recent attendance trends and risk indicators.

    Args:
        student_features_df: DataFrame with student features.
        risk_indicator_column: Column name for risk indicator (default: attendance_trend).

    Returns:
        CorrelationResult for recent attendance vs risk indicator.
    """
    if risk_indicator_column not in student_features_df.columns:
        raise DataValidationError(f"student_features_df must contain '{risk_indicator_column}'")

    # Use attendance_percentage as the recent attendance measure
    if "attendance_percentage" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'attendance_percentage'")

    result = analyze_correlation(
        student_features_df["attendance_percentage"],
        student_features_df[risk_indicator_column],
        method="spearman",  # Use Spearman for trend data
        variable_x_name="attendance_percentage",
        variable_y_name=risk_indicator_column,
    )

    logger.info(
        "Recent attendance vs %s correlation: r=%.3f",
        risk_indicator_column,
        result.correlation_coefficient,
    )
    return result


def generate_correlation_matrix(
    df: pd.DataFrame,
    variables: List[str],
    method: str = "pearson",
) -> CorrelationMatrix:
    """Generate a correlation matrix for multiple variables.

    Args:
        df: DataFrame with the variables.
        variables: List of column names to include in the matrix.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        CorrelationMatrix with correlation and p-value matrices.
    """
    if not all(var in df.columns for var in variables):
        missing = [var for var in variables if var not in df.columns]
        raise DataValidationError(f"DataFrame missing columns: {missing}")

    n_vars = len(variables)
    corr_matrix = [[None for _ in range(n_vars)] for _ in range(n_vars)]
    p_matrix = [[None for _ in range(n_vars)] for _ in range(n_vars)]

    sample_size = 0

    for i, var_x in enumerate(variables):
        for j, var_y in enumerate(variables):
            if i == j:
                corr_matrix[i][j] = 1.0
                p_matrix[i][j] = 0.0
            else:
                result = analyze_correlation(
                    df[var_x],
                    df[var_y],
                    method=method,
                    variable_x_name=var_x,
                    variable_y_name=var_y,
                )
                corr_matrix[i][j] = result.correlation_coefficient
                p_matrix[i][j] = result.p_value
                if result.sample_size > sample_size:
                    sample_size = result.sample_size

    logger.info("Generated correlation matrix for %d variables using %s method", n_vars, method)

    return CorrelationMatrix(
        variables=variables,
        correlation_matrix=corr_matrix,
        p_value_matrix=p_matrix,
        method=method,
        sample_size=sample_size,
    )


def generate_correlation_insights(
    result: CorrelationResult,
    context: Optional[Dict[str, Any]] = None,
) -> CorrelationInsights:
    """Generate insights from correlation analysis with causation warnings.

    Args:
        result: CorrelationResult from analysis.
        context: Optional context information (sample characteristics, etc.).

    Returns:
        CorrelationInsights with interpretation and recommendations.
    """
    # Statistical significance
    stat_sig = result.p_value is not None and result.p_value < 0.05

    # Practical significance (based on strength)
    practical_sig = result.strength in ["moderate", "moderate_to_strong", "strong"]

    # Potential confounders based on variable types
    confounders: List[str] = []
    var_x_lower = result.variable_x.lower()
    var_y_lower = result.variable_y.lower()

    if "attendance" in var_x_lower or "attendance" in var_y_lower:
        confounders.extend([
            "Student motivation and engagement",
            "External circumstances (work, health, family)",
            "Course difficulty and instructor quality",
            "Prior academic preparation",
        ])

    if "exam" in var_x_lower or "exam" in var_y_lower:
        confounders.extend([
            "Test-taking skills and anxiety",
            "Study time and methods",
            "Sleep and nutrition before exam",
            "Course material alignment with exam",
        ])

    if "assignment" in var_x_lower or "assignment" in var_y_lower:
        confounders.extend([
            "Time management skills",
            "Access to resources and support",
            "Assignment clarity and fairness",
            "Group work dynamics",
        ])

    # Limitations
    limitations: List[str] = [
        "Correlation does not imply causation",
        "Observational data cannot control for all confounding variables",
        "Sample may not be representative of broader population",
    ]

    if result.sample_size < 30:
        limitations.append(f"Small sample size (n={result.sample_size}) reduces statistical power")

    if result.method == "pearson":
        limitations.append("Pearson correlation assumes linear relationship")
    elif result.method == "spearman":
        limitations.append("Spearman correlation assesses monotonic, not necessarily linear, relationships")

    # Recommendations
    recommendations: List[str] = []

    if stat_sig and practical_sig:
        recommendations.append(
            f"Statistically and practically significant {result.direction} relationship detected"
        )
        recommendations.append("Consider this relationship when designing interventions")
    elif stat_sig and not practical_sig:
        recommendations.append(
            "Statistically significant but weak relationship - may not warrant intervention"
        )
    elif not stat_sig:
        recommendations.append("No statistically significant relationship detected")

    recommendations.append("Investigate potential causal mechanisms through controlled studies")
    recommendations.append("Consider qualitative research to understand underlying reasons")

    # Causation warning
    causation_warning = (
        "IMPORTANT: Correlation does not imply causation. A significant correlation between "
        f"{result.variable_x} and {result.variable_y} does not mean that changes in one variable "
        f"cause changes in the other. There may be unmeasured confounding variables, reverse "
        "causality, or the relationship may be spurious. Any interventions based on this "
        "correlation should be tested with experimental or quasi-experimental designs."
    )

    return CorrelationInsights(
        correlation_pair=f"{result.variable_x} vs {result.variable_y}",
        statistical_significance=stat_sig,
        practical_significance=practical_sig,
        potential_confounders=confounders,
        limitations=limitations,
        recommendations=recommendations,
        causation_warning=causation_warning,
    )


def generate_correlation_report_markdown(
    results: List[CorrelationResult],
    insights_list: Optional[List[CorrelationInsights]] = None,
) -> str:
    """Generate a markdown summary of correlation analysis results.

    Args:
        results: List of CorrelationResult objects.
        insights_list: Optional list of CorrelationInsights objects.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic Correlation Analysis Report",
        "",
        f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Correlations Analyzed**: {len(results)}",
        "",
        "## ⚠️ Important Disclaimer",
        "",
        "**Correlation does not imply causation.** This report identifies statistical relationships "
        "between academic variables but does not make causal claims. The observed correlations may "
        "be influenced by unmeasured confounding variables, reverse causality, or other factors. "
        "Any interventions based on these findings should be validated through experimental or "
        "quasi-experimental designs.",
        "",
        "## Correlation Results",
        "",
    ]

    for i, result in enumerate(results, 1):
        lines.append(f"### {i}. {result.variable_x} vs {result.variable_y}")
        lines.append("")

        lines.append("**Statistical Summary**")
        lines.append(f"- **Method**: {result.method}")
        lines.append(f"- **Sample Size**: {result.sample_size:,}")
        lines.append(f"- **Correlation Coefficient**: {result.correlation_coefficient if result.correlation_coefficient is not None else 'N/A'}")
        lines.append(f"- **P-Value**: {result.p_value if result.p_value is not None else 'N/A'}")
        lines.append(f"- **Strength**: {result.strength}")
        lines.append(f"- **Direction**: {result.direction}")
        lines.append("")

        lines.append("**Interpretation**")
        lines.append(f"- {result.interpretation}")
        lines.append("")

        if result.notes:
            lines.append("**Notes**")
            for note in result.notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.append("---")
        lines.append("")

    if insights_list:
        lines.append("## Detailed Insights and Recommendations")
        lines.append("")

        for i, insights in enumerate(insights_list, 1):
            lines.append(f"### {i}. {insights.correlation_pair}")
            lines.append("")

            lines.append("**Significance Assessment**")
            lines.append(f"- **Statistically Significant**: {'Yes' if insights.statistical_significance else 'No'}")
            lines.append(f"- **Practically Significant**: {'Yes' if insights.practical_significance else 'No'}")
            lines.append("")

            if insights.potential_confounders:
                lines.append("**Potential Confounding Variables**")
                for confounder in insights.potential_confounders:
                    lines.append(f"- {confounder}")
                lines.append("")

            if insights.limitations:
                lines.append("**Study Limitations**")
                for limitation in insights.limitations:
                    lines.append(f"- {limitation}")
                lines.append("")

            if insights.recommendations:
                lines.append("**Recommendations**")
                for rec in insights.recommendations:
                    lines.append(f"- 💡 {rec}")
                lines.append("")

            lines.append("**Causation Warning**")
            lines.append(f"> {insights.causation_warning}")
            lines.append("")

            lines.append("---")
            lines.append("")

    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Pearson Correlation**: Measures linear relationship between continuous variables")
    lines.append("- **Spearman Correlation**: Measures monotonic relationship (rank-based)")
    lines.append("- **Kendall's Tau**: Measures ordinal association (rank-based)")
    lines.append("- **P-Value < 0.05**: Generally considered statistically significant")
    lines.append("- **Strength Interpretation**: Negligible (<0.1), Weak (0.1-0.3), Moderate (0.3-0.5), Strong (>0.5)")
    lines.append("")

    return "\n".join(lines)


def partial_correlation(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
) -> Tuple[float, int]:
    """Calculate partial correlation between x and y controlling for z.

    Args:
        x: First variable series.
        y: Second variable series.
        z: Control variable series.

    Returns:
        Tuple of (partial_correlation_coefficient, sample_size).
    """
    # Remove NaN values for all three variables
    valid_mask = x.notna() & y.notna() & z.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    z_clean = z[valid_mask]

    if len(x_clean) < 4:
        return np.nan, len(x_clean)

    # Calculate correlations
    r_xy, _, _ = calculate_pearson_correlation(x_clean, y_clean)
    r_xz, _, _ = calculate_pearson_correlation(x_clean, z_clean)
    r_yz, _, _ = calculate_pearson_correlation(y_clean, z_clean)

    # Partial correlation formula
    denominator = (1 - r_xz**2) * (1 - r_yz**2)
    if denominator <= 0:
        return np.nan, len(x_clean)

    partial_r = (r_xy - r_xz * r_yz) / np.sqrt(denominator)

    return float(partial_r), len(x_clean)
