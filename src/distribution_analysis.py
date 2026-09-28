"""Academic Distribution Analysis Module.

Provides reusable analytical functions for analyzing distributions of:
- attendance
- assignment completion
- assignment scores
- exam scores
- submission delays

Identifies useful academic patterns and anomalies through statistical analysis.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.distribution_analysis")


@dataclass
class DistributionSummary:
    """Statistical summary of a distribution."""

    metric_name: str
    count: int
    missing_count: int
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    q25: Optional[float]
    q75: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class DistributionInsights:
    """Actionable insights derived from distribution analysis."""

    metric_name: str
    distribution_type: str  # 'normal', 'skewed_left', 'skewed_right', 'bimodal', 'uniform', 'unknown'
    risk_thresholds: Dict[str, float]
    outlier_count: int
    outlier_percentage: float
    patterns: List[str]
    anomalies: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class DistributionReport:
    """Complete distribution analysis report."""

    metric_name: str
    summary: DistributionSummary
    insights: DistributionInsights
    histogram_data: Optional[Dict[str, List]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        result = {
            "metric_name": self.metric_name,
            "summary": self.summary.to_dict(),
            "insights": self.insights.to_dict(),
        }
        if self.histogram_data:
            result["histogram_data"] = self.histogram_data
        return result


def calculate_distribution_summary(
    series: pd.Series,
    metric_name: str,
) -> DistributionSummary:
    """Calculate comprehensive statistical summary of a distribution.

    Args:
        series: Pandas Series containing numeric values.
        metric_name: Name of the metric being analyzed.

    Returns:
        DistributionSummary with statistical measures.
    """
    clean_series = pd.to_numeric(series, errors="coerce")
    valid_values = clean_series.dropna()

    count = len(series)
    missing_count = int(series.isna().sum())

    if valid_values.empty:
        return DistributionSummary(
            metric_name=metric_name,
            count=count,
            missing_count=missing_count,
            mean=None,
            median=None,
            std=None,
            min_val=None,
            max_val=None,
            q25=None,
            q75=None,
            skewness=None,
            kurtosis=None,
        )

    mean_val = float(valid_values.mean())
    median_val = float(valid_values.median())
    std_val = float(valid_values.std()) if len(valid_values) > 1 else 0.0
    min_val = float(valid_values.min())
    max_val = float(valid_values.max())
    q25 = float(valid_values.quantile(0.25))
    q75 = float(valid_values.quantile(0.75))

    # Skewness and kurtosis require sufficient data
    skewness = float(stats.skew(valid_values)) if len(valid_values) >= 3 else None
    kurtosis = float(stats.kurtosis(valid_values)) if len(valid_values) >= 4 else None

    return DistributionSummary(
        metric_name=metric_name,
        count=count,
        missing_count=missing_count,
        mean=round(mean_val, 2),
        median=round(median_val, 2),
        std=round(std_val, 2),
        min_val=round(min_val, 2),
        max_val=round(max_val, 2),
        q25=round(q25, 2),
        q75=round(q75, 2),
        skewness=round(skewness, 3) if skewness is not None else None,
        kurtosis=round(kurtosis, 3) if kurtosis is not None else None,
    )


def identify_distribution_type(
    summary: DistributionSummary,
    series: pd.Series,
) -> str:
    """Identify the type of distribution based on statistical properties.

    Args:
        summary: DistributionSummary with statistics.
        series: Original pandas Series for additional checks.

    Returns:
        Distribution type string.
    """
    if summary.skewness is None or summary.kurtosis is None:
        return "unknown"

    skew = summary.skewness
    kurt = summary.kurtosis

    # Normal distribution: skewness near 0, kurtosis near 0
    if abs(skew) < 0.5 and abs(kurt) < 1.0:
        return "normal"

    # Skewed right (positive skew): tail on the right
    if skew > 0.5:
        return "skewed_right"

    # Skewed left (negative skew): tail on the left
    if skew < -0.5:
        return "skewed_left"

    # Bimodal: high kurtosis might indicate multiple peaks
    if kurt > 2.0:
        return "bimodal"

    # Uniform: very low kurtosis
    if kurt < -1.5:
        return "uniform"

    return "unknown"


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5,
) -> Tuple[np.ndarray, float, float]:
    """Detect outliers using the IQR method.

    Args:
        series: Pandas Series with numeric values.
        multiplier: IQR multiplier (default 1.5 for standard outliers).

    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound).
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    if clean_series.empty:
        return np.array([]), 0.0, 0.0

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)

    return outlier_mask, float(lower_bound), float(upper_bound)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """Detect outliers using z-score method.

    Args:
        series: Pandas Series with numeric values.
        threshold: Z-score threshold (default 3.0).

    Returns:
        Tuple of (outlier_mask, threshold_value).
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    if clean_series.empty or len(clean_series) < 3:
        return np.array([]), threshold

    z_scores = np.abs(stats.zscore(clean_series))
    outlier_mask = z_scores > threshold

    return outlier_mask, threshold


def generate_distribution_insights(
    series: pd.Series,
    summary: DistributionSummary,
    metric_name: str,
) -> DistributionInsights:
    """Generate actionable insights from distribution analysis.

    Args:
        series: Pandas Series with the metric values.
        summary: DistributionSummary with statistics.
        metric_name: Name of the metric being analyzed.

    Returns:
        DistributionInsights with patterns, anomalies, and recommendations.
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()

    if clean_series.empty:
        return DistributionInsights(
            metric_name=metric_name,
            distribution_type="unknown",
            risk_thresholds={},
            outlier_count=0,
            outlier_percentage=0.0,
            patterns=["No data available for analysis"],
            anomalies=["Insufficient data"],
            recommendations=["Collect more data"],
        )

    dist_type = identify_distribution_type(summary, series)

    # Detect outliers using IQR method
    outlier_mask, lower_bound, upper_bound = detect_outliers_iqr(series)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round((outlier_count / len(clean_series)) * 100.0, 2) if len(clean_series) > 0 else 0.0

    patterns: List[str] = []
    anomalies: List[str] = []
    recommendations: List[str] = []

    # Risk thresholds based on metric type
    risk_thresholds: Dict[str, float] = {}

    # Metric-specific analysis
    if "attendance" in metric_name.lower():
        risk_thresholds = {"low_risk": 75.0, "moderate_risk": 60.0, "high_risk": 50.0}
        if summary.mean and summary.mean < 70.0:
            patterns.append(f"Low average attendance ({summary.mean}%)")
            anomalies.append("Class-wide attendance below recommended threshold")
            recommendations.append("Review course engagement and attendance policies")
        if dist_type == "skewed_left":
            patterns.append("Most students have high attendance, few with very low attendance")
            recommendations.append("Target interventions at students with lowest attendance")

    elif "completion" in metric_name.lower():
        risk_thresholds = {"low_risk": 80.0, "moderate_risk": 60.0, "high_risk": 40.0}
        if summary.mean and summary.mean < 75.0:
            patterns.append(f"Low average completion rate ({summary.mean}%)")
            anomalies.append("Significant coursework abandonment")
            recommendations.append("Review assignment difficulty and workload")
        if outlier_pct > 10.0:
            anomalies.append(f"High outlier rate ({outlier_pct}%) in completion rates")

    elif "score" in metric_name.lower():
        risk_thresholds = {"failing": 40.0, "warning": 60.0, "good": 75.0}
        if summary.mean and summary.mean < 60.0:
            patterns.append(f"Low average score ({summary.mean})")
            anomalies.append("Class performance below passing threshold")
            recommendations.append("Review teaching methods and assessment difficulty")
        if summary.std and summary.std > 20.0:
            patterns.append("High score variability")
            recommendations.append("Consider differentiated instruction")

    elif "delay" in metric_name.lower() or "late" in metric_name.lower():
        risk_thresholds = {"acceptable": 1.0, "moderate": 3.0, "severe": 7.0}
        if summary.mean and summary.mean > 3.0:
            patterns.append(f"High average delay ({summary.mean} days)")
            anomalies.append("Chronic submission delays")
            recommendations.append("Review deadline policies and time management support")

    # Distribution-specific patterns
    if dist_type == "skewed_right":
        patterns.append("Right-skewed distribution: majority of values clustered at lower end")
    elif dist_type == "skewed_left":
        patterns.append("Left-skewed distribution: majority of values clustered at higher end")
    elif dist_type == "bimodal":
        patterns.append("Bimodal distribution: two distinct performance groups")
        recommendations.append("Investigate factors creating performance dichotomy")
    elif dist_type == "normal":
        patterns.append("Normally distributed: balanced spread around mean")

    # Outlier analysis
    if outlier_count > 0:
        patterns.append(f"{outlier_count} outliers detected ({outlier_pct}% of data)")
        if outlier_pct > 5.0:
            anomalies.append(f"High outlier percentage ({outlier_pct}%)")
            recommendations.append("Investigate causes of extreme values")

    # Missing data analysis
    if summary.missing_count > 0:
        missing_pct = (summary.missing_count / summary.count) * 100.0
        if missing_pct > 10.0:
            anomalies.append(f"High missing data rate ({missing_pct:.1f}%)")
            recommendations.append("Improve data collection processes")

    # Skewness insights
    if summary.skewness:
        if abs(summary.skewness) > 1.0:
            patterns.append(f"Strong skewness ({summary.skewness:.2f}) indicates asymmetric distribution")

    return DistributionInsights(
        metric_name=metric_name,
        distribution_type=dist_type,
        risk_thresholds=risk_thresholds,
        outlier_count=outlier_count,
        outlier_percentage=outlier_pct,
        patterns=patterns if patterns else ["No significant patterns detected"],
        anomalies=anomalies if anomalies else ["No anomalies detected"],
        recommendations=recommendations if recommendations else ["Continue monitoring"],
    )


def analyze_distribution(
    series: pd.Series,
    metric_name: str,
    include_histogram: bool = False,
    bins: int = 20,
) -> DistributionReport:
    """Perform complete distribution analysis on a metric.

    Args:
        series: Pandas Series with metric values.
        metric_name: Name of the metric being analyzed.
        include_histogram: Whether to include histogram data.
        bins: Number of bins for histogram.

    Returns:
        DistributionReport with summary and insights.
    """
    if not isinstance(series, pd.Series):
        raise DataValidationError(f"Expected pandas Series, got: {type(series).__name__}")

    summary = calculate_distribution_summary(series, metric_name)
    insights = generate_distribution_insights(series, summary, metric_name)

    histogram_data = None
    if include_histogram:
        clean_series = pd.to_numeric(series, errors="coerce").dropna()
        if not clean_series.empty:
            hist, bin_edges = np.histogram(clean_series, bins=bins)
            histogram_data = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
            }

    logger.info("Analyzed distribution for '%s': %s values, %s type", metric_name, summary.count, insights.distribution_type)

    return DistributionReport(
        metric_name=metric_name,
        summary=summary,
        insights=insights,
        histogram_data=histogram_data,
    )


def analyze_attendance_distribution(
    attendance_df: pd.DataFrame,
    student_features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, DistributionReport]:
    """Analyze attendance distributions across the dataset.

    Args:
        attendance_df: DataFrame with attendance records.
        student_features_df: Optional DataFrame with student-level attendance features.

    Returns:
        Dictionary mapping metric names to report objects.
    """
    reports: Dict[str, DistributionReport] = {}

    # 1. Analyze student-level attendance percentages if available
    if student_features_df is not None and "attendance_percentage" in student_features_df.columns:
        report = analyze_distribution(
            student_features_df["attendance_percentage"],
            "student_attendance_percentage",
            include_histogram=True,
        )
        reports["student_attendance_percentage"] = report

    # 2. Analyze attendance status distribution
    if "status" in attendance_df.columns:
        status_counts = attendance_df["status"].value_counts()
        logger.info("Attendance status distribution: %s", status_counts.to_dict())

    return reports


def analyze_assignment_completion_distribution(
    student_features_df: pd.DataFrame,
) -> DistributionReport:
    """Analyze assignment completion rate distribution.

    Args:
        student_features_df: DataFrame with student features including completion rates.

    Returns:
        DistributionReport for assignment completion.
    """
    if "assignment_completion_rate" not in student_features_df.columns:
        raise DataValidationError("student_features_df must contain 'assignment_completion_rate' column")

    report = analyze_distribution(
        student_features_df["assignment_completion_rate"],
        "assignment_completion_rate",
        include_histogram=True,
    )

    return report


def analyze_assignment_scores_distribution(
    submissions_df: pd.DataFrame,
) -> DistributionReport:
    """Analyze assignment scores distribution.

    Args:
        submissions_df: DataFrame with submission records including scores.

    Returns:
        DistributionReport for assignment scores.
    """
    if "score" not in submissions_df.columns:
        raise DataValidationError("submissions_df must contain 'score' column")

    clean_scores = pd.to_numeric(submissions_df["score"], errors="coerce")
    report = analyze_distribution(
        clean_scores,
        "assignment_scores",
        include_histogram=True,
    )

    return report


def analyze_exam_scores_distribution(
    exams_df: pd.DataFrame,
) -> DistributionReport:
    """Analyze exam scores distribution.

    Args:
        exams_df: DataFrame with exam records including scores.

    Returns:
        DistributionReport for exam scores.
    """
    if "score" not in exams_df.columns:
        raise DataValidationError("exams_df must contain 'score' column")

    clean_scores = pd.to_numeric(exams_df["score"], errors="coerce")
    report = analyze_distribution(
        clean_scores,
        "exam_scores",
        include_histogram=True,
    )

    return report


def analyze_submission_delays_distribution(
    submissions_df: pd.DataFrame,
) -> DistributionReport:
    """Analyze submission delays distribution.

    Args:
        submissions_df: DataFrame with submission records including delay information.

    Returns:
        DistributionReport for submission delays.
    """
    # Check for days_late or calculate from dates
    if "days_late" in submissions_df.columns:
        delays = pd.to_numeric(submissions_df["days_late"], errors="coerce")
    elif "is_late" in submissions_df.columns:
        # Binary: 1 for late, 0 for on-time
        delays = submissions_df["is_late"].astype(int)
    else:
        raise DataValidationError("submissions_df must contain 'days_late' or 'is_late' column")

    report = analyze_distribution(
        delays,
        "submission_delays",
        include_histogram=True,
    )

    return report


def compare_distributions(
    series1: pd.Series,
    series2: pd.Series,
    name1: str,
    name2: str,
) -> Dict[str, Any]:
    """Compare two distributions using statistical tests.

    Args:
        series1: First distribution series.
        series2: Second distribution series.
        name1: Name of first distribution.
        name2: Name of second distribution.

    Returns:
        Dictionary with comparison statistics and test results.
    """
    clean1 = pd.to_numeric(series1, errors="coerce").dropna()
    clean2 = pd.to_numeric(series2, errors="coerce").dropna()

    if clean1.empty or clean2.empty:
        return {
            "name1": name1,
            "name2": name2,
            "error": "One or both distributions are empty",
        }

    result: Dict[str, Any] = {
        "name1": name1,
        "name2": name2,
        "mean1": round(float(clean1.mean()), 2),
        "mean2": round(float(clean2.mean()), 2),
        "median1": round(float(clean1.median()), 2),
        "median2": round(float(clean2.median()), 2),
        "std1": round(float(clean1.std()), 2) if len(clean1) > 1 else 0.0,
        "std2": round(float(clean2.std()), 2) if len(clean2) > 1 else 0.0,
    }

    # Mann-Whitney U test (non-parametric)
    try:
        u_stat, p_value = stats.mannwhitneyu(clean1, clean2, alternative="two-sided")
        result["mann_whitney_u_statistic"] = round(float(u_stat), 2)
        result["mann_whitney_p_value"] = round(float(p_value), 4)
        result["significantly_different"] = p_value < 0.05
    except Exception as e:
        logger.warning("Mann-Whitney U test failed: %s", e)
        result["mann_whitney_test"] = "failed"

    # Kolmogorov-Smirnov test
    try:
        ks_stat, ks_p = stats.ks_2samp(clean1, clean2)
        result["ks_statistic"] = round(float(ks_stat), 4)
        result["ks_p_value"] = round(float(ks_p), 4)
    except Exception as e:
        logger.warning("KS test failed: %s", e)
        result["ks_test"] = "failed"

    return result


def detect_academic_anomalies(
    student_features_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    """Detect students with anomalous academic patterns.

    Args:
        student_features_df: DataFrame with engineered student features.

    Returns:
        Dictionary mapping anomaly types to lists of affected students.
    """
    anomalies: Dict[str, List[Dict[str, Any]]] = {
        "critical_attendance": [],
        "critical_completion": [],
        "critical_scores": [],
        "severe_delays": [],
        "declining_trends": [],
    }

    if student_features_df.empty:
        return anomalies

    # Critical attendance (< 50%)
    if "attendance_percentage" in student_features_df.columns:
        critical_att = student_features_df[
            student_features_df["attendance_percentage"].notna() &
            (student_features_df["attendance_percentage"] < 50.0)
        ]
        for _, row in critical_att.iterrows():
            anomalies["critical_attendance"].append({
                "student_id": row.get("student_id"),
                "value": row["attendance_percentage"],
            })

    # Critical completion (< 40%)
    if "assignment_completion_rate" in student_features_df.columns:
        critical_comp = student_features_df[
            (student_features_df["assignment_completion_rate"] < 40.0)
        ]
        for _, row in critical_comp.iterrows():
            anomalies["critical_completion"].append({
                "student_id": row.get("student_id"),
                "value": row["assignment_completion_rate"],
            })

    # Critical scores (< 40% average)
    for score_col in ["average_assignment_score", "average_exam_score"]:
        if score_col in student_features_df.columns:
            critical_scores = student_features_df[
                student_features_df[score_col].notna() &
                (student_features_df[score_col] < 40.0)
            ]
            for _, row in critical_scores.iterrows():
                anomalies["critical_scores"].append({
                    "student_id": row.get("student_id"),
                    "metric": score_col,
                    "value": row[score_col],
                })

    # Severe delays (> 5 late submissions)
    if "late_submission_count" in student_features_df.columns:
        severe_delays = student_features_df[
            student_features_df["late_submission_count"] > 5
        ]
        for _, row in severe_delays.iterrows():
            anomalies["severe_delays"].append({
                "student_id": row.get("student_id"),
                "late_count": row["late_submission_count"],
            })

    # Declining trends
    for trend_col in ["attendance_trend", "assignment_trend", "exam_trend"]:
        if trend_col in student_features_df.columns:
            declining = student_features_df[
                student_features_df[trend_col].notna() &
                (student_features_df[trend_col] < -10.0)
            ]
            for _, row in declining.iterrows():
                anomalies["declining_trends"].append({
                    "student_id": row.get("student_id"),
                    "metric": trend_col,
                    "trend_value": row[trend_col],
                })

    logger.info(
        "Detected anomalies: %d attendance, %d completion, %d scores, %d delays, %d trends",
        len(anomalies["critical_attendance"]),
        len(anomalies["critical_completion"]),
        len(anomalies["critical_scores"]),
        len(anomalies["severe_delays"]),
        len(anomalies["declining_trends"]),
    )

    return anomalies


def generate_distribution_summary_markdown(
    reports: Dict[str, DistributionReport],
) -> str:
    """Generate a markdown summary of distribution analysis results.

    Args:
        reports: Dictionary of distribution reports.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic Distribution Analysis Summary",
        "",
        f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Metrics Analyzed**: {len(reports)}",
        "",
        "## Distribution Reports",
        "",
    ]

    for metric_name, report in reports.items():
        lines.append(f"### {metric_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("**Statistical Summary**")
        lines.append(f"- **Count**: {report.summary.count:,}")
        lines.append(f"- **Missing**: {report.summary.missing_count:,}")
        
        if report.summary.mean is not None:
            lines.append(f"- **Mean**: {report.summary.mean}")
            lines.append(f"- **Median**: {report.summary.median}")
            lines.append(f"- **Std Dev**: {report.summary.std}")
            lines.append(f"- **Range**: [{report.summary.min_val}, {report.summary.max_val}]")
            lines.append(f"- **Q1-Q3**: [{report.summary.q25}, {report.summary.q75}]")
            
            if report.summary.skewness is not None:
                lines.append(f"- **Skewness**: {report.summary.skewness}")
            if report.summary.kurtosis is not None:
                lines.append(f"- **Kurtosis**: {report.summary.kurtosis}")

        lines.append("")
        lines.append("**Distribution Type**")
        lines.append(f"- {report.insights.distribution_type}")
        lines.append("")

        lines.append("**Key Patterns**")
        for pattern in report.insights.patterns:
            lines.append(f"- {pattern}")
        lines.append("")

        if report.insights.anomalies and report.insights.anomalies != ["No anomalies detected"]:
            lines.append("**Anomalies Detected**")
            for anomaly in report.insights.anomalies:
                lines.append(f"- ⚠️ {anomaly}")
            lines.append("")

        if report.insights.recommendations and report.insights.recommendations != ["Continue monitoring"]:
            lines.append("**Recommendations**")
            for rec in report.insights.recommendations:
                lines.append(f"- 💡 {rec}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
