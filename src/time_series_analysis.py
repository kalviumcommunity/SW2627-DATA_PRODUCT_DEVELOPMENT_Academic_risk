"""Academic Time-Series Analysis Module.

Provides time-based analysis for:
- attendance trends
- assignment completion trends
- exam performance
- student engagement

Calculates weekly/monthly aggregates and rolling metrics.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.time_series_analysis")


@dataclass
class TimeSeriesMetrics:
    """Metrics for a time series analysis."""

    metric_name: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_periods: int
    mean_value: Optional[float]
    std_value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'unknown'
    trend_slope: Optional[float]
    trend_r_squared: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class AggregatedTimeSeries:
    """Aggregated time series data."""

    metric_name: str
    aggregation_level: str  # 'daily', 'weekly', 'monthly'
    data: pd.DataFrame
    metrics: TimeSeriesMetrics
    rolling_metrics: Optional[Dict[str, pd.Series]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "metric_name": self.metric_name,
            "aggregation_level": self.aggregation_level,
            "metrics": self.metrics.to_dict(),
            "data": self.data.to_dict(orient="records"),
            "rolling_metrics": {
                k: v.to_list() if v is not None else None
                for k, v in (self.rolling_metrics or {}).items()
            },
        }


def ensure_datetime_column(
    df: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Ensure a column is datetime type.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column.

    Returns:
        DataFrame with datetime column.

    Raises:
        DataValidationError: If column is missing or cannot be converted.
    """
    if date_column not in df.columns:
        raise DataValidationError(f"DataFrame must contain '{date_column}' column")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    if df[date_column].isna().all():
        raise DataValidationError(f"Column '{date_column}' could not be converted to datetime")

    return df


def calculate_trend(
    series: pd.Series,
) -> Tuple[str, Optional[float], Optional[float]]:
    """Calculate trend direction, slope, and R-squared using linear regression.

    Args:
        series: Time series values.

    Returns:
        Tuple of (direction, slope, r_squared).
    """
    # Remove NaN values
    valid_series = series.dropna()

    if len(valid_series) < 3:
        return "unknown", None, None

    # Create numeric index for regression
    x = np.arange(len(valid_series))
    y = valid_series.values

    # Linear regression
    try:
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return direction, float(slope), float(r_squared)
    except Exception:
        return "unknown", None, None


def aggregate_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "weekly",
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Aggregate time series data by time period.

    Args:
        df: Input DataFrame with date and value columns.
        date_column: Name of the date column.
        value_column: Name of the value column.
        aggregation: Aggregation level ('daily', 'weekly', 'monthly').
        agg_func: Aggregation function ('mean', 'sum', 'count', 'min', 'max').

    Returns:
        DataFrame with aggregated time series.

    Raises:
        DataValidationError: If parameters are invalid.
    """
    df = ensure_datetime_column(df, date_column)

    if value_column not in df.columns:
        raise DataValidationError(f"DataFrame must contain '{value_column}' column")

    if aggregation not in ["daily", "weekly", "monthly"]:
        raise DataValidationError(f"Invalid aggregation: {aggregation}")

    if agg_func not in ["mean", "sum", "count", "min", "max"]:
        raise DataValidationError(f"Invalid agg_func: {agg_func}")

    df = df.set_index(date_column).sort_index()

    # Aggregate by time period
    if aggregation == "daily":
        grouped = df.resample("D")
    elif aggregation == "weekly":
        grouped = df.resample("W")
    else:  # monthly
        grouped = df.resample("ME")

    if agg_func == "mean":
        aggregated = grouped[value_column].mean()
    elif agg_func == "sum":
        aggregated = grouped[value_column].sum()
    elif agg_func == "count":
        aggregated = grouped[value_column].count()
    elif agg_func == "min":
        aggregated = grouped[value_column].min()
    else:  # max
        aggregated = grouped[value_column].max()

    result = aggregated.reset_index()
    result.columns = [date_column, value_column]

    return result


def calculate_rolling_metrics(
    series: pd.Series,
    windows: List[int] = [7, 14, 30],
) -> Dict[str, pd.Series]:
    """Calculate rolling metrics for a time series.

    Args:
        series: Time series values.
        windows: List of window sizes for rolling calculations.

    Returns:
        Dictionary of rolling metrics.
    """
    rolling_metrics: Dict[str, pd.Series] = {}

    for window in windows:
        if len(series) >= window:
            rolling_metrics[f"rolling_mean_{window}"] = series.rolling(window=window).mean()
            rolling_metrics[f"rolling_std_{window}"] = series.rolling(window=window).std()
            rolling_metrics[f"rolling_min_{window}"] = series.rolling(window=window).min()
            rolling_metrics[f"rolling_max_{window}"] = series.rolling(window=window).max()

    return rolling_metrics


def analyze_attendance_trends(
    attendance_df: pd.DataFrame,
    date_column: str = "date",
    student_id_column: str = "student_id",
    status_column: str = "status",
    aggregation: str = "weekly",
) -> AggregatedTimeSeries:
    """Analyze attendance trends over time.

    Args:
        attendance_df: DataFrame with attendance records.
        date_column: Name of the date column.
        student_id_column: Name of the student ID column.
        status_column: Name of the status column.
        aggregation: Aggregation level ('daily', 'weekly', 'monthly').

    Returns:
        AggregatedTimeSeries with attendance trend analysis.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if date_column not in attendance_df.columns:
        raise DataValidationError(f"attendance_df must contain '{date_column}' column")
    if student_id_column not in attendance_df.columns:
        raise DataValidationError(f"attendance_df must contain '{student_id_column}' column")

    df = ensure_datetime_column(attendance_df, date_column)

    # Calculate attendance rate per period
    if status_column in df.columns:
        # Count present students per period
        df["is_present"] = df[status_column].astype(str).str.lower().isin(["present", "p"])
        daily_attendance = df.groupby(date_column).agg({
            "is_present": "mean",
            student_id_column: "count",
        }).reset_index()
        daily_attendance.columns = [date_column, "attendance_rate", "student_count"]
        value_column = "attendance_rate"
    else:
        # Just count records per period
        daily_attendance = df.groupby(date_column).agg({
            student_id_column: "count",
        }).reset_index()
        daily_attendance.columns = [date_column, "record_count"]
        value_column = "record_count"

    # Aggregate to requested level
    if aggregation != "daily":
        aggregated = aggregate_time_series(
            daily_attendance,
            date_column,
            value_column,
            aggregation=aggregation,
            agg_func="mean",
        )
    else:
        aggregated = daily_attendance

    # Calculate metrics
    values = aggregated[value_column].dropna()
    direction, slope, r_squared = calculate_trend(values)

    metrics = TimeSeriesMetrics(
        metric_name="attendance_trend",
        start_date=aggregated[date_column].min().strftime("%Y-%m-%d") if not aggregated.empty else None,
        end_date=aggregated[date_column].max().strftime("%Y-%m-%d") if not aggregated.empty else None,
        total_periods=len(aggregated),
        mean_value=float(values.mean()) if not values.empty else None,
        std_value=float(values.std()) if len(values) > 1 else None,
        min_value=float(values.min()) if not values.empty else None,
        max_value=float(values.max()) if not values.empty else None,
        trend_direction=direction,
        trend_slope=slope,
        trend_r_squared=r_squared,
    )

    # Calculate rolling metrics
    rolling = calculate_rolling_metrics(aggregated[value_column])

    logger.info(
        "Analyzed attendance trends: %s aggregation, %s direction, slope=%.4f",
        aggregation,
        direction,
        slope if slope is not None else 0.0,
    )

    return AggregatedTimeSeries(
        metric_name="attendance_trend",
        aggregation_level=aggregation,
        data=aggregated,
        metrics=metrics,
        rolling_metrics=rolling,
    )


def analyze_assignment_completion_trends(
    submissions_df: pd.DataFrame,
    date_column: str = "submission_date",
    student_id_column: str = "student_id",
    aggregation: str = "weekly",
) -> AggregatedTimeSeries:
    """Analyze assignment completion trends over time.

    Args:
        submissions_df: DataFrame with submission records.
        date_column: Name of the date column.
        student_id_column: Name of the student ID column.
        aggregation: Aggregation level ('daily', 'weekly', 'monthly').

    Returns:
        AggregatedTimeSeries with completion trend analysis.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if date_column not in submissions_df.columns:
        raise DataValidationError(f"submissions_df must contain '{date_column}' column")
    if student_id_column not in submissions_df.columns:
        raise DataValidationError(f"submissions_df must contain '{student_id_column}' column")

    df = ensure_datetime_column(submissions_df, date_column)

    # Count submissions per period
    daily_submissions = df.groupby(date_column).agg({
        student_id_column: "count",
    }).reset_index()
    daily_submissions.columns = [date_column, "submission_count"]

    # Aggregate to requested level
    if aggregation != "daily":
        aggregated = aggregate_time_series(
            daily_submissions,
            date_column,
            "submission_count",
            aggregation=aggregation,
            agg_func="sum",
        )
    else:
        aggregated = daily_submissions

    # Calculate metrics
    values = aggregated["submission_count"].dropna()
    direction, slope, r_squared = calculate_trend(values)

    metrics = TimeSeriesMetrics(
        metric_name="assignment_completion_trend",
        start_date=aggregated[date_column].min().strftime("%Y-%m-%d") if not aggregated.empty else None,
        end_date=aggregated[date_column].max().strftime("%Y-%m-%d") if not aggregated.empty else None,
        total_periods=len(aggregated),
        mean_value=float(values.mean()) if not values.empty else None,
        std_value=float(values.std()) if len(values) > 1 else None,
        min_value=float(values.min()) if not values.empty else None,
        max_value=float(values.max()) if not values.empty else None,
        trend_direction=direction,
        trend_slope=slope,
        trend_r_squared=r_squared,
    )

    # Calculate rolling metrics
    rolling = calculate_rolling_metrics(aggregated["submission_count"])

    logger.info(
        "Analyzed assignment completion trends: %s aggregation, %s direction",
        aggregation,
        direction,
    )

    return AggregatedTimeSeries(
        metric_name="assignment_completion_trend",
        aggregation_level=aggregation,
        data=aggregated,
        metrics=metrics,
        rolling_metrics=rolling,
    )


def analyze_exam_performance_trends(
    exams_df: pd.DataFrame,
    date_column: str = "exam_date",
    score_column: str = "score",
    aggregation: str = "weekly",
) -> AggregatedTimeSeries:
    """Analyze exam performance trends over time.

    Args:
        exams_df: DataFrame with exam records.
        date_column: Name of the date column.
        score_column: Name of the score column.
        aggregation: Aggregation level ('daily', 'weekly', 'monthly').

    Returns:
        AggregatedTimeSeries with exam performance trend analysis.

    Raises:
        DataValidationError: If required columns are missing.
    """
    if date_column not in exams_df.columns:
        raise DataValidationError(f"exams_df must contain '{date_column}' column")
    if score_column not in exams_df.columns:
        raise DataValidationError(f"exams_df must contain '{score_column}' column")

    df = ensure_datetime_column(exams_df, date_column)
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")

    # Calculate average score per period
    daily_scores = df.groupby(date_column).agg({
        score_column: "mean",
    }).reset_index()
    daily_scores.columns = [date_column, "avg_score"]

    # Aggregate to requested level
    if aggregation != "daily":
        aggregated = aggregate_time_series(
            daily_scores,
            date_column,
            "avg_score",
            aggregation=aggregation,
            agg_func="mean",
        )
    else:
        aggregated = daily_scores

    # Calculate metrics
    values = aggregated["avg_score"].dropna()
    direction, slope, r_squared = calculate_trend(values)

    metrics = TimeSeriesMetrics(
        metric_name="exam_performance_trend",
        start_date=aggregated[date_column].min().strftime("%Y-%m-%d") if not aggregated.empty else None,
        end_date=aggregated[date_column].max().strftime("%Y-%m-%d") if not aggregated.empty else None,
        total_periods=len(aggregated),
        mean_value=float(values.mean()) if not values.empty else None,
        std_value=float(values.std()) if len(values) > 1 else None,
        min_value=float(values.min()) if not values.empty else None,
        max_value=float(values.max()) if not values.empty else None,
        trend_direction=direction,
        trend_slope=slope,
        trend_r_squared=r_squared,
    )

    # Calculate rolling metrics
    rolling = calculate_rolling_metrics(aggregated["avg_score"])

    logger.info(
        "Analyzed exam performance trends: %s aggregation, %s direction, avg_score=%.2f",
        aggregation,
        direction,
        metrics.mean_value if metrics.mean_value else 0.0,
    )

    return AggregatedTimeSeries(
        metric_name="exam_performance_trend",
        aggregation_level=aggregation,
        data=aggregated,
        metrics=metrics,
        rolling_metrics=rolling,
    )


def analyze_student_engagement_trends(
    attendance_df: pd.DataFrame,
    submissions_df: pd.DataFrame,
    date_column: str = "date",
    aggregation: str = "weekly",
) -> AggregatedTimeSeries:
    """Analyze student engagement trends over time.

    Engagement is measured as a combination of attendance and submissions.

    Args:
        attendance_df: DataFrame with attendance records.
        submissions_df: DataFrame with submission records.
        date_column: Name of the date column.
        aggregation: Aggregation level ('daily', 'weekly', 'monthly').

    Returns:
        AggregatedTimeSeries with engagement trend analysis.

    Raises:
        DataValidationError: If required columns are missing.
    """
    # Process attendance
    if not attendance_df.empty and date_column in attendance_df.columns:
        att_df = ensure_datetime_column(attendance_df, date_column)
        daily_attendance = att_df.groupby(date_column).size().reset_index()
        daily_attendance.columns = [date_column, "attendance_count"]
    else:
        daily_attendance = pd.DataFrame(columns=[date_column, "attendance_count"])

    # Process submissions
    if not submissions_df.empty and date_column in submissions_df.columns:
        sub_df = ensure_datetime_column(submissions_df, date_column)
        daily_submissions = sub_df.groupby(date_column).size().reset_index()
        daily_submissions.columns = [date_column, "submission_count"]
    else:
        daily_submissions = pd.DataFrame(columns=[date_column, "submission_count"])

    # Handle empty data case
    if daily_attendance.empty and daily_submissions.empty:
        # Return empty result
        metrics = TimeSeriesMetrics(
            metric_name="student_engagement_trend",
            start_date=None,
            end_date=None,
            total_periods=0,
            mean_value=None,
            std_value=None,
            min_value=None,
            max_value=None,
            trend_direction="unknown",
            trend_slope=None,
            trend_r_squared=None,
        )

        return AggregatedTimeSeries(
            metric_name="student_engagement_trend",
            aggregation_level=aggregation,
            data=pd.DataFrame(columns=[date_column, "engagement_score"]),
            metrics=metrics,
            rolling_metrics=None,
        )

    # Merge and calculate engagement score
    merged = pd.merge(
        daily_attendance,
        daily_submissions,
        on=date_column,
        how="outer",
    ).fillna(0)

    # Simple engagement score: attendance + submissions
    merged["engagement_score"] = merged["attendance_count"] + merged["submission_count"]

    # Aggregate to requested level
    if aggregation != "daily":
        aggregated = aggregate_time_series(
            merged,
            date_column,
            "engagement_score",
            aggregation=aggregation,
            agg_func="mean",
        )
    else:
        aggregated = merged[[date_column, "engagement_score"]]

    # Calculate metrics
    values = aggregated["engagement_score"].dropna()
    direction, slope, r_squared = calculate_trend(values)

    metrics = TimeSeriesMetrics(
        metric_name="student_engagement_trend",
        start_date=aggregated[date_column].min().strftime("%Y-%m-%d") if not aggregated.empty else None,
        end_date=aggregated[date_column].max().strftime("%Y-%m-%d") if not aggregated.empty else None,
        total_periods=len(aggregated),
        mean_value=float(values.mean()) if not values.empty else None,
        std_value=float(values.std()) if len(values) > 1 else None,
        min_value=float(values.min()) if not values.empty else None,
        max_value=float(values.max()) if not values.empty else None,
        trend_direction=direction,
        trend_slope=slope,
        trend_r_squared=r_squared,
    )

    # Calculate rolling metrics
    rolling = calculate_rolling_metrics(aggregated["engagement_score"])

    logger.info(
        "Analyzed student engagement trends: %s aggregation, %s direction",
        aggregation,
        direction,
    )

    return AggregatedTimeSeries(
        metric_name="student_engagement_trend",
        aggregation_level=aggregation,
        data=aggregated,
        metrics=metrics,
        rolling_metrics=rolling,
    )


def compare_time_series(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "Series 1",
    name2: str = "Series 2",
) -> Dict[str, Any]:
    """Compare two time series.

    Args:
        series1: First time series.
        series2: Second time series.
        name1: Name of first series.
        name2: Name of second series.

    Returns:
        Dictionary with comparison metrics.
    """
    # Align series
    aligned = pd.DataFrame({name1: series1, name2: series2}).dropna()

    if aligned.empty:
        return {
            "error": "No overlapping data points",
            "name1": name1,
            "name2": name2,
        }

    # Calculate correlation
    correlation = aligned[name1].corr(aligned[name2])

    # Calculate trends
    direction1, slope1, r2_1 = calculate_trend(aligned[name1])
    direction2, slope2, r2_2 = calculate_trend(aligned[name2])

    return {
        "name1": name1,
        "name2": name2,
        "correlation": float(correlation) if not np.isnan(correlation) else None,
        "name1_trend": direction1,
        "name1_slope": slope1,
        "name1_r_squared": r2_1,
        "name2_trend": direction2,
        "name2_slope": slope2,
        "name2_r_squared": r2_2,
        "overlapping_periods": len(aligned),
    }


def generate_time_series_report_markdown(
    results: List[AggregatedTimeSeries],
) -> str:
    """Generate a markdown summary of time series analysis.

    Args:
        results: List of AggregatedTimeSeries objects.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Academic Time-Series Analysis Report",
        "",
        f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Metrics Analyzed**: {len(results)}",
        "",
        "## Time-Series Results",
        "",
    ]

    for result in results:
        lines.append(f"### {result.metric_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("**Analysis Summary**")
        lines.append(f"- **Aggregation Level**: {result.aggregation_level}")
        lines.append(f"- **Start Date**: {result.metrics.start_date}")
        lines.append(f"- **End Date**: {result.metrics.end_date}")
        lines.append(f"- **Total Periods**: {result.metrics.total_periods}")
        lines.append("")

        lines.append("**Statistical Summary**")
        lines.append(f"- **Mean**: {result.metrics.mean_value}")
        lines.append(f"- **Std Dev**: {result.metrics.std_value}")
        lines.append(f"- **Range**: [{result.metrics.min_value}, {result.metrics.max_value}]")
        lines.append("")

        lines.append("**Trend Analysis**")
        lines.append(f"- **Direction**: {result.metrics.trend_direction}")
        lines.append(f"- **Slope**: {result.metrics.trend_slope}")
        lines.append(f"- **R-Squared**: {result.metrics.trend_r_squared}")
        lines.append("")

        if result.rolling_metrics:
            lines.append("**Rolling Metrics Available**")
            for metric_name in result.rolling_metrics.keys():
                lines.append(f"- {metric_name}")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Trend Direction**: Calculated using linear regression on time-indexed values")
    lines.append("- **Slope**: Rate of change per time period")
    lines.append("- **R-Squared**: Goodness of fit for trend line (0-1)")
    lines.append("- **Rolling Metrics**: Calculated over specified windows (7, 14, 30 periods)")
    lines.append("- **Aggregation**: Data grouped by time period (daily, weekly, monthly)")
    lines.append("")

    return "\n".join(lines)
