"""NumPy Vectorized Academic Calculations and Numerical Transformations.

High-performance, vectorized numerical processing for academic analytics:
- Score transformations (curving, GPA conversions, boundary clipping)
- Trend calculations (vectorized OLS regression slopes, multi-student batch slopes, momentum)
- Numerical normalization (Z-score, Min-Max, Robust IQR scaling, percentile ranks)
- Metric calculations (weighted assessment composites, attendance rates, composite risk indices)

Avoids slow Python loops and handles NaN missingness safely according to academic rules.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.exceptions import DataTransformationError, DataValidationError
from src.logger import get_logger

logger = get_logger("academic_risk.numpy_calculations")


# ---------------------------------------------------------------------------
# Score Transformations
# ---------------------------------------------------------------------------


def vectorized_curve_scores(
    scores: Union[np.ndarray, pd.Series],
    method: str = "sqrt",
    max_score: float = 100.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply vectorized academic score curving across a score array.

    Curving Methods:
    - 'sqrt': Classic square-root curve: 10 * sqrt(score) scaled to max_score.
              (e.g., a score of 64 becomes 10 * 8 = 80; score of 100 remains 100).
    - 'linear': Uniform linear offset: score + offset, capped at max_score.
    - 'max_scale': Scale so the maximum observed score becomes max_score:
                   score * (max_score / max_observed).

    Args:
        scores: Numerical array or Series of scores.
        method: Curving algorithm ('sqrt', 'linear', 'max_scale').
        max_score: Upper boundary ceiling (default 100.0).
        offset: Points to add if method='linear'.

    Returns:
        Curved score array with NaNs preserved, bounded in [0.0, max_score].
    """
    arr = np.asarray(scores, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out

    valid_vals = arr[valid_mask]
    method_clean = method.strip().lower()

    if method_clean == "sqrt":
        # Formula: sqrt(score / max_score) * max_score
        # For max_score=100: sqrt(score) * 10
        clipped_in = np.clip(valid_vals, 0.0, max_score)
        curved = np.sqrt(clipped_in / max_score) * max_score
    elif method_clean == "linear":
        curved = valid_vals + offset
    elif method_clean == "max_scale":
        obs_max = np.nanmax(valid_vals)
        if obs_max > 0:
            scale_factor = max_score / obs_max
            curved = valid_vals * scale_factor
        else:
            curved = valid_vals
    else:
        raise DataValidationError(f"Unsupported curving method: '{method}'")

    # Clip to legitimate academic domain bounds [0.0, max_score]
    curved = np.clip(curved, 0.0, max_score)
    out[valid_mask] = np.round(curved, 2)
    return out


def vectorized_score_to_gpa(
    scores: Union[np.ndarray, pd.Series],
    scale: float = 4.0,
) -> np.ndarray:
    """Convert numerical scores (0-100) into GPA scale (0.0 - 4.0) using vectorized evaluation.

    Standard collegiate scale:
    - >= 93: 4.00
    - >= 90: 3.70
    - >= 87: 3.30
    - >= 83: 3.00
    - >= 80: 2.70
    - >= 77: 2.30
    - >= 73: 2.00
    - >= 70: 1.70
    - >= 67: 1.30
    - >= 60: 1.00
    - < 60: 0.00

    Args:
        scores: Numerical array of percentage scores.
        scale: Maximum GPA scale (default 4.0).

    Returns:
        NumPy array of GPA values with NaNs preserved.
    """
    arr = np.asarray(scores, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out

    vals = arr[valid_mask]

    # Vectorized piecewise threshold evaluation using np.select
    conditions = [
        vals >= 93.0,
        vals >= 90.0,
        vals >= 87.0,
        vals >= 83.0,
        vals >= 80.0,
        vals >= 77.0,
        vals >= 73.0,
        vals >= 70.0,
        vals >= 67.0,
        vals >= 60.0,
    ]
    choices = [
        4.0,
        3.7,
        3.3,
        3.0,
        2.7,
        2.3,
        2.0,
        1.7,
        1.3,
        1.0,
    ]

    gpa_vals = np.select(conditions, choices, default=0.0)
    if scale != 4.0:
        gpa_vals = (gpa_vals / 4.0) * scale

    out[valid_mask] = np.round(gpa_vals, 2)
    return out


def vectorized_clip_scores(
    scores: Union[np.ndarray, pd.Series],
    lower: float = 0.0,
    upper: float = 100.0,
) -> np.ndarray:
    """Clip score values to domain bounds while strictly preserving NaN missingness.

    Args:
        scores: Input scores array.
        lower: Minimum bound (default 0.0).
        upper: Maximum bound (default 100.0).

    Returns:
        Clipped array with NaNs intact.
    """
    arr = np.asarray(scores, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    out[valid_mask] = np.clip(arr[valid_mask], lower, upper)
    return out


# ---------------------------------------------------------------------------
# Trend & Trajectory Calculations
# ---------------------------------------------------------------------------


def vectorized_linear_slope(
    values: Union[np.ndarray, pd.Series],
    time_points: Optional[Union[np.ndarray, pd.Series]] = None,
) -> float:
    """Compute the Ordinary Least Squares (OLS) slope (trajectory) of a 1D sequence over time.

    Formula:
        slope = Cov(t, y) / Var(t) = sum((t - mean_t) * (y - mean_y)) / sum((t - mean_t)^2)

    Args:
        values: 1D sequence of student performance metrics over time.
        time_points: Optional 1D sequence of time indices/weeks. If None, uses 0, 1, ..., N-1.

    Returns:
        Float slope (rate of change per time unit), or NaN if < 2 valid observations.
    """
    y_arr = np.asarray(values, dtype=np.float64)
    valid_mask = ~np.isnan(y_arr)
    valid_y = y_arr[valid_mask]

    n = len(valid_y)
    if n < 2:
        return np.nan

    if time_points is not None:
        t_arr = np.asarray(time_points, dtype=np.float64)[valid_mask]
    else:
        t_arr = np.arange(len(y_arr), dtype=np.float64)[valid_mask]

    t_mean = np.mean(t_arr)
    y_mean = np.mean(valid_y)

    t_diff = t_arr - t_mean
    var_t = np.sum(t_diff ** 2)

    if var_t == 0.0:
        return 0.0

    covariance = np.sum(t_diff * (valid_y - y_mean))
    return float(np.round(covariance / var_t, 4))


def vectorized_batch_linear_slopes(matrix: np.ndarray) -> np.ndarray:
    """Compute linear performance slopes for hundreds of students simultaneously in a single matrix operation.

    Given a 2D matrix of shape (M students, N time_steps), computes each student's
    trajectory slope using vectorized matrix multiplication, eliminating Python loops.

    Args:
        matrix: 2D array of shape (M, N) containing performance metrics across N time points.
                Missing entries should be NaN or interpolated.

    Returns:
        1D array of shape (M,) containing OLS slope for each student.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise DataValidationError(f"Expected 2D matrix (M students, N time points), got ndim={mat.ndim}")

    m_students, n_points = mat.shape
    if n_points < 2:
        return np.full(m_students, np.nan)

    # Time steps: shape (N,)
    t = np.arange(n_points, dtype=np.float64)
    t_mean = np.mean(t)
    t_diff = t - t_mean  # shape (N,)
    denom = np.sum(t_diff ** 2)  # scalar

    if denom == 0.0:
        return np.zeros(m_students, dtype=np.float64)

    # Handle missing values student by student if NaNs are present, else clean matrix dot product
    if np.isnan(mat).any():
        slopes = np.empty(m_students, dtype=np.float64)
        for i in range(m_students):
            slopes[i] = vectorized_linear_slope(mat[i], t)
        return slopes

    # Fully vectorized OLS across all M students simultaneously:
    # y_mean shape (M, 1)
    y_mean = np.mean(mat, axis=1, keepdims=True)
    # y_diff shape (M, N)
    y_diff = mat - y_mean
    # Dot product of each row with t_diff: shape (M,)
    numerators = np.dot(y_diff, t_diff)
    slopes = numerators / denom
    return np.round(slopes, 4)


def vectorized_momentum(
    early_values: Union[np.ndarray, pd.Series],
    recent_values: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Compute academic momentum across an array of students in vectorized fashion.

    Formula:
        momentum = recent_values - early_values

    Args:
        early_values: Array of metrics from early term.
        recent_values: Array of metrics from recent term.

    Returns:
        Array of percentage point changes.
    """
    e = np.asarray(early_values, dtype=np.float64)
    r = np.asarray(recent_values, dtype=np.float64)
    return np.round(r - e, 2)


# ---------------------------------------------------------------------------
# Numerical Normalization & Scaling
# ---------------------------------------------------------------------------


def vectorized_zscore_normalize(
    data: Union[np.ndarray, pd.Series],
    ddof: int = 1,
) -> np.ndarray:
    """Perform vectorized Z-score standardization: (X - mean) / std.

    Safe against std=0 (returns zeros instead of crashing or generating infs).

    Args:
        data: Numerical array.
        ddof: Delta degrees of freedom for std calculation (default 1).

    Returns:
        Z-score normalized array with NaNs preserved.
    """
    arr = np.asarray(data, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out

    valid = arr[valid_mask]
    mean_val = np.mean(valid)
    std_val = np.std(valid, ddof=ddof) if len(valid) > 1 else 0.0

    if std_val == 0.0:
        out[valid_mask] = 0.0
    else:
        out[valid_mask] = np.round((valid - mean_val) / std_val, 4)

    return out


def vectorized_minmax_normalize(
    data: Union[np.ndarray, pd.Series],
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Perform vectorized Min-Max scaling: (X - min) / (max - min) * (upper - lower) + lower.

    Safe against max == min (returns lower bound).

    Args:
        data: Numerical array.
        feature_range: Output range tuple (default (0.0, 1.0)).

    Returns:
        Min-max scaled array with NaNs preserved.
    """
    arr = np.asarray(data, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out

    valid = arr[valid_mask]
    min_val = np.min(valid)
    max_val = np.max(valid)
    range_span = max_val - min_val

    lower, upper = feature_range
    target_span = upper - lower

    if range_span == 0.0:
        out[valid_mask] = lower
    else:
        scaled = ((valid - min_val) / range_span) * target_span + lower
        out[valid_mask] = np.round(scaled, 4)

    return out


def vectorized_robust_scale(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Scale data using robust statistics: (X - median) / IQR.

    Resilient against extreme score outliers and skewed academic metrics.

    Args:
        data: Numerical array.

    Returns:
        Robustly scaled array with NaNs preserved.
    """
    arr = np.asarray(data, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out

    valid = arr[valid_mask]
    median_val = np.median(valid)
    q1 = np.percentile(valid, 25)
    q3 = np.percentile(valid, 75)
    iqr = q3 - q1

    if iqr == 0.0:
        out[valid_mask] = 0.0
    else:
        out[valid_mask] = np.round((valid - median_val) / iqr, 4)

    return out


def vectorized_percentile_rank(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Compute empirical percentile ranks [0.0, 100.0] across students without loops.

    Args:
        data: 1D array of student performance values.

    Returns:
        1D array of percentile ranks with NaNs preserved.
    """
    arr = np.asarray(data, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out

    valid = arr[valid_mask]
    n = len(valid)

    # Sort and rank using argsort
    sort_indices = np.argsort(valid)
    ranks = np.empty(n, dtype=np.float64)
    ranks[sort_indices] = np.arange(1, n + 1)

    pct_ranks = (ranks / n) * 100.0
    out[valid_mask] = np.round(pct_ranks, 2)
    return out


# ---------------------------------------------------------------------------
# Metric Calculations
# ---------------------------------------------------------------------------


def vectorized_attendance_rates(
    present: Union[np.ndarray, pd.Series],
    late: Union[np.ndarray, pd.Series],
    excused: Union[np.ndarray, pd.Series],
    total: Union[np.ndarray, pd.Series],
    late_weight: float = 0.5,
) -> np.ndarray:
    """Vectorized calculation of attendance percentage rates across arrays of student counts.

    Formula:
        rate = ((present + late_weight * late) / (total - excused)) * 100.0

    Args:
        present: Array of present session counts.
        late: Array of late session counts.
        excused: Array of excused session counts.
        total: Array of total session counts.
        late_weight: Multiplier for late attendance (default 0.5).

    Returns:
        Array of percentage rates bounded in [0.0, 100.0] with NaNs preserved.
    """
    p = np.asarray(present, dtype=np.float64)
    l = np.asarray(late, dtype=np.float64)
    e = np.asarray(excused, dtype=np.float64)
    t = np.asarray(total, dtype=np.float64)

    effective_sessions = t - e
    out = np.full_like(p, np.nan)

    # Compute strictly where effective sessions > 0
    valid_mask = effective_sessions > 0
    if np.any(valid_mask):
        numerators = p[valid_mask] + late_weight * l[valid_mask]
        rates = (numerators / effective_sessions[valid_mask]) * 100.0
        out[valid_mask] = np.round(np.clip(rates, 0.0, 100.0), 2)

    return out


def vectorized_weighted_average(
    scores: np.ndarray,
    weights: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Calculate weighted average across scores with dynamic re-normalization for missing evaluations.

    If a student has NaNs in certain components (e.g. final exam not yet administered),
    weights are dynamically re-normalized over the non-null components rather than penalizing
    with zeroes.

    Args:
        scores: NumPy array of scores (e.g. shape (M, K) for M students, K assessment types).
        weights: 1D array of weights of length K (must sum to > 0).
        axis: Axis along which to compute (default -1 or None).

    Returns:
        Weighted average float or array of weighted averages.
    """
    sc = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if sc.ndim == 1:
        if len(sc) != len(w):
            raise DataValidationError(f"Length mismatch between scores ({len(sc)}) and weights ({len(w)})")
        valid = ~np.isnan(sc)
        if not np.any(valid):
            return np.nan
        active_w = w[valid]
        w_sum = np.sum(active_w)
        if w_sum == 0.0:
            return np.nan
        return float(np.round(np.sum(sc[valid] * active_w) / w_sum, 2))

    elif sc.ndim == 2:
        m, k = sc.shape
        if k != len(w):
            raise DataValidationError(f"Expected scores 2nd dimension ({k}) to match weights ({len(w)})")

        valid_mask = ~np.isnan(sc)
        # Broadcast weights across (M, K)
        w_broadcast = np.broadcast_to(w, (m, k))
        masked_w = np.where(valid_mask, w_broadcast, 0.0)
        w_sums = np.sum(masked_w, axis=1)

        masked_sc = np.where(valid_mask, sc, 0.0)
        numerators = np.sum(masked_sc * masked_w, axis=1)

        out = np.full(m, np.nan, dtype=np.float64)
        has_eval = w_sums > 0
        out[has_eval] = np.round(numerators[has_eval] / w_sums[has_eval], 2)
        return out

    else:
        raise DataValidationError(f"Expected 1D or 2D array, got ndim={sc.ndim}")


def vectorized_composite_risk_score(
    attendance_pct: Union[np.ndarray, pd.Series],
    assignment_avg: Union[np.ndarray, pd.Series],
    exam_avg: Union[np.ndarray, pd.Series],
    weights: Tuple[float, float, float] = (0.35, 0.35, 0.30),
) -> np.ndarray:
    """Compute composite academic risk index [0.0, 100.0] using vectorized vector algebra.

    Higher score = Greater Academic Risk:
    - 0.0: Flawless attendance and perfect scores
    - 100.0: Extreme academic disengagement and failure

    Formula:
        Performance = w1 * Att + w2 * Assign + w3 * Exam
        Risk = 100.0 - Performance

    Handles missing components by dynamically re-weighting available evidence.

    Args:
        attendance_pct: Array of attendance percentages.
        assignment_avg: Array of assignment averages.
        exam_avg: Array of exam averages.
        weights: Weights for attendance, assignments, and exams (default 0.35, 0.35, 0.30).

    Returns:
        Array of risk index values [0.0, 100.0].
    """
    att = np.asarray(attendance_pct, dtype=np.float64)
    asn = np.asarray(assignment_avg, dtype=np.float64)
    exm = np.asarray(exam_avg, dtype=np.float64)

    # Stack into (M, 3) matrix
    mat = np.column_stack([att, asn, exm])
    w = np.array(weights, dtype=np.float64)

    # Compute weighted performance
    perf = vectorized_weighted_average(mat, w)
    out = np.full_like(att, np.nan)

    valid_mask = ~np.isnan(perf)
    if np.any(valid_mask):
        risk = 100.0 - perf[valid_mask]
        out[valid_mask] = np.round(np.clip(risk, 0.0, 100.0), 2)

    return out
