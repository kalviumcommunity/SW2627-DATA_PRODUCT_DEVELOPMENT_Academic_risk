"""Unit tests for Concept #13: Academic Outlier Detection & Treatment."""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.outlier_detection import (
    AcademicOutlierReport,
    BatchOutlierReport,
    ColumnOutlierSummary,
    OutlierMethod,
    OutlierTreatment,
    analyze_and_treat_outliers,
    compute_iqr_bounds,
    compute_modified_zscore_bounds,
    compute_zscore_bounds,
    detect_academic_outliers,
    detect_assignment_score_outliers,
    detect_attendance_outliers,
    detect_column_outliers,
    detect_exam_score_outliers,
    detect_submission_delay_outliers,
)


class TestOutlierDetection:
    """Test suite for statistical bounds, entity outlier detection, and explainable treatments."""

    def test_compute_iqr_bounds_standard_and_domain_clipping(self):
        """Test IQR fences calculation and domain constraint clipping."""
        # 10 values from 60 to 80, plus two outliers: 10 and 100
        vals = pd.Series([10.0, 60.0, 65.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 100.0])
        lower, upper, iqr = compute_iqr_bounds(vals, k=1.5, domain_min=0.0, domain_max=100.0)

        assert iqr > 0
        assert lower >= 0.0  # Clipped by domain_min
        assert upper <= 100.0  # Clipped by domain_max
        assert lower < upper

    def test_compute_zscore_bounds_and_zero_std(self):
        """Test Z-score bounds with normal data and zero variance edge case."""
        vals = pd.Series([50.0, 50.0, 50.0, 50.0])
        lower, upper, mean, std = compute_zscore_bounds(vals, threshold=3.0)
        assert std == 0.0
        assert lower == 50.0
        assert upper == 50.0

        normal_vals = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        lower_n, upper_n, mean_n, std_n = compute_zscore_bounds(normal_vals, threshold=2.0)
        assert std_n > 0
        assert lower_n < mean_n < upper_n

    def test_compute_modified_zscore_bounds(self):
        """Test robust Modified Z-score bounds using median and MAD."""
        # Skewed series
        vals = pd.Series([70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 15.0])
        lower, upper, median, mad = compute_modified_zscore_bounds(vals, threshold=3.5)
        assert median == 72.0
        assert lower < 70.0
        assert upper > 75.0

    def test_detect_column_outliers_iqr_and_zscore(self):
        """Test detect_column_outliers outputs and summary statistics."""
        # Normal scores around 70-80, one very low score (15.0), one domain invalid (150.0)
        data = pd.Series([15.0, 70.0, 72.0, 74.0, 75.0, 76.0, 78.0, 80.0, 150.0, None], name="score")
        mask, summary = detect_column_outliers(data, method="iqr", domain_min=0.0, domain_max=100.0)

        assert isinstance(summary, ColumnOutlierSummary)
        assert summary.total_records == 10
        assert summary.non_null_records == 9
        # Both 15.0 and 150.0 should be flagged
        assert summary.outliers_count >= 2
        assert summary.domain_invalid_count == 1  # 150.0 is > 100.0
        assert mask.iloc[0] == True  # 15.0
        assert mask.iloc[8] == True  # 150.0
        assert mask.iloc[1] == False  # 70.0 normal
        assert mask.iloc[9] == False  # None not outlier

    def test_analyze_and_treat_outliers_flag_only_preserves_records(self):
        """Test that treatment='flag_only' retains all rows while annotating risk."""
        df = pd.DataFrame({
            "student_id": ["S01", "S02", "S03", "S04", "S05", "S06"],
            "score": [15.0, 75.0, 76.0, 78.0, 80.0, 98.0],
        })

        out, report = analyze_and_treat_outliers(
            df=df,
            columns=["score"],
            entity_name="test_exams",
            treatment="flag_only",
            domain_bounds={"score": (0.0, 100.0)},
        )

        assert len(out) == 6  # No rows removed!
        assert "score_is_outlier" in out.columns
        assert "score_outlier_type" in out.columns

        # Row 0 (score=15.0) is flagged as low_risk
        assert out.loc[0, "score_is_outlier"] == True
        assert out.loc[0, "score_outlier_type"] == "low_risk"

        # Middle rows are normal
        assert out.loc[1, "score_outlier_type"] == "normal"
        assert report.total_rows == 6
        assert report.outlier_rows_count >= 1

    def test_analyze_and_treat_outliers_cap_winsorizes(self):
        """Test that treatment='cap' bounds extreme values while logging raw value."""
        df = pd.DataFrame({
            "score": [5.0, 70.0, 72.0, 75.0, 78.0, 95.0],
        })
        out, report = analyze_and_treat_outliers(
            df=df,
            columns=["score"],
            treatment="cap",
        )
        assert "raw_score" in out.columns
        assert "score_is_capped" in out.columns
        assert len(out) == 6
        # Score is clipped to lower bound
        lower_b = report.column_summaries["score"].lower_bound
        assert out.loc[0, "score"] == lower_b
        assert out.loc[0, "raw_score"] == 5.0

    def test_analyze_and_treat_outliers_quarantine_separates_domain_invalids(self):
        """Test that treatment='quarantine' moves impossible academic values to audit."""
        df = pd.DataFrame({
            "student_id": ["S01", "S02", "S03"],
            "score": [75.0, -25.0, 180.0],  # -25 and 180 are invalid domain points
        })
        out, report = analyze_and_treat_outliers(
            df=df,
            columns=["score"],
            treatment="quarantine",
            domain_bounds={"score": (0.0, 100.0)},
        )

        # 2 records quarantined
        assert len(out) == 1
        assert out.loc[0, "student_id"] == "S01"
        assert report.quarantined_records is not None
        assert len(report.quarantined_records) == 2
        assert "outlier_quarantine_reason" in report.quarantined_records.columns

    def test_detect_attendance_outliers(self):
        """Test attendance outlier detection."""
        df = pd.DataFrame({
            "student_id": ["S01", "S02", "S03", "S04", "S05"],
            "attendance_pct": [15.0, 85.0, 88.0, 90.0, 92.0],
        })
        out, report = detect_attendance_outliers(df, treatment="flag_only")

        assert "attendance_pct_is_outlier" in out.columns
        assert out.loc[0, "attendance_pct_is_outlier"] == True
        assert out.loc[0, "attendance_pct_outlier_type"] == "low_risk"
        assert report.entity_name == "attendance"

    def test_detect_submission_delay_outliers(self):
        """Test submission latency delay outlier detection."""
        df = pd.DataFrame({
            "student_id": ["S01", "S02", "S03", "S04", "S05"],
            "days_late": [0.0, 0.5, 1.0, 1.5, 45.0],  # 45 days late is an extreme outlier
        })
        out, report = detect_submission_delay_outliers(df, treatment="flag_only")

        assert "days_late_is_outlier" in out.columns
        assert out.loc[4, "days_late_is_outlier"] == True
        assert out.loc[4, "days_late_outlier_type"] == "high_achiever" or out.loc[4, "days_late_outlier_type"] != "normal"

    def test_detect_exam_and_assignment_score_outliers(self):
        """Test exam and assignment score outlier functions."""
        df_exam = pd.DataFrame({"score": [20.0, 75.0, 78.0, 80.0, 82.0]})
        out_e, rep_e = detect_exam_score_outliers(df_exam)
        assert rep_e.entity_name == "exams"
        assert out_e.loc[0, "score_is_outlier"] == True

        df_sub = pd.DataFrame({"score": [10.0, 85.0, 87.0, 90.0, 92.0]})
        out_s, rep_s = detect_assignment_score_outliers(df_sub)
        assert rep_s.entity_name == "submissions"
        assert out_s.loc[0, "score_is_outlier"] == True

    def test_outlier_report_markdown(self):
        """Test markdown generation for AcademicOutlierReport and BatchOutlierReport."""
        df = pd.DataFrame({"score": [10.0, 70.0, 75.0, 80.0, 85.0]})
        _, report = detect_exam_score_outliers(df)

        md = report.to_markdown()
        assert "### Academic Outlier Audit Report: `exams`" in md
        assert "| `score` | IQR |" in md

        batch = {"exams": df}
        _, batch_report = detect_academic_outliers(batch)
        b_md = batch_report.to_markdown()
        assert "## Academic Outlier Detection & Treatment Summary" in b_md
        assert "`exams`" in b_md
