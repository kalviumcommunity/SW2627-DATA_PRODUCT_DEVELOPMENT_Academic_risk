"""Unit tests for Concept #26: Academic Root Cause Analysis."""

import unittest
import numpy as np
import pandas as pd

from src.exceptions import DataValidationError
from src.root_cause_analysis import (
    NON_DIAGNOSTIC_DISCLAIMER,
    ROOT_CAUSE_THRESHOLDS,
    EvidenceItem,
    FactorContribution,
    StudentRootCauseReport,
    CourseRootCauseReport,
    analyze_attendance_evidence,
    analyze_assignments_evidence,
    analyze_submissions_evidence,
    analyze_exams_evidence,
    analyze_recent_trends_evidence,
    analyze_student_root_cause,
    analyze_course_root_cause,
    batch_analyze_students,
    identify_students_for_investigation,
    run_root_cause_investigation,
    generate_student_root_cause_markdown,
    generate_course_root_cause_markdown,
    get_root_cause_thresholds,
)


class TestAnalyzeAttendanceEvidence(unittest.TestCase):
    """Test attendance evidence analysis."""

    def test_critical_attendance(self):
        """Test attendance below critical threshold (<50%)."""
        factor = analyze_attendance_evidence(42.0, total_sessions=20, absent_sessions=11)
        self.assertEqual(factor.category, "attendance")
        self.assertEqual(factor.severity, "CRITICAL")
        self.assertGreater(factor.score_weight, 30.0)
        self.assertEqual(len(factor.evidence_items), 1)
        ev = factor.evidence_items[0]
        self.assertEqual(ev.metric_name, "attendance_percentage")
        self.assertEqual(ev.severity, "CRITICAL")
        self.assertIn("42.0%", ev.finding)

    def test_low_attendance(self):
        """Test attendance between low and critical thresholds (50-70%)."""
        factor = analyze_attendance_evidence(62.5, total_sessions=20, absent_sessions=7)
        self.assertEqual(factor.severity, "HIGH")
        ev = factor.evidence_items[0]
        self.assertEqual(ev.severity, "HIGH")
        self.assertIn("62.5%", ev.finding)

    def test_normal_attendance(self):
        """Test attendance meeting institutional benchmarks (>=70%)."""
        factor = analyze_attendance_evidence(88.0, total_sessions=20, absent_sessions=2)
        self.assertEqual(factor.severity, "LOW")
        self.assertEqual(factor.score_weight, 0.0)
        ev = factor.evidence_items[0]
        self.assertEqual(ev.severity, "NEGLIGIBLE")

    def test_consecutive_absences(self):
        """Test detection of consecutive absence streaks."""
        factor = analyze_attendance_evidence(
            attendance_percentage=72.0,
            consecutive_absences=4,
        )
        self.assertEqual(factor.severity, "MODERATE")
        cons_ev = next(e for e in factor.evidence_items if e.metric_name == "consecutive_absences")
        self.assertEqual(cons_ev.observed_value, 4)
        self.assertEqual(cons_ev.severity, "MODERATE")

    def test_none_and_nan_attendance(self):
        """Test graceful handling of unrecorded attendance."""
        factor_none = analyze_attendance_evidence(None)
        self.assertEqual(factor_none.severity, "NEGLIGIBLE")
        self.assertEqual(len(factor_none.evidence_items), 0)

        factor_nan = analyze_attendance_evidence(float("nan"))
        self.assertEqual(factor_nan.severity, "NEGLIGIBLE")

    def test_custom_thresholds(self):
        """Test with customized thresholds."""
        custom_th = {"attendance_low": 80.0, "attendance_critical": 65.0, "attendance_consecutive_absence_threshold": 2}
        factor = analyze_attendance_evidence(75.0, thresholds=custom_th)
        self.assertEqual(factor.severity, "HIGH")


class TestAnalyzeAssignmentsEvidence(unittest.TestCase):
    """Test assignment and coursework performance evidence analysis."""

    def test_critical_coursework_score(self):
        """Test critically low assignment average (<40)."""
        factor = analyze_assignments_evidence(34.0, total_assignments=5, failing_assignments_count=4)
        self.assertEqual(factor.category, "assignments")
        self.assertEqual(factor.severity, "CRITICAL")
        self.assertGreater(factor.score_weight, 30.0)

    def test_low_coursework_score(self):
        """Test below-threshold assignment average (40-60)."""
        factor = analyze_assignments_evidence(54.0, total_assignments=5, failing_assignments_count=2)
        self.assertEqual(factor.severity, "HIGH")

    def test_satisfactory_coursework_score(self):
        """Test passing coursework marks."""
        factor = analyze_assignments_evidence(82.0, total_assignments=5, failing_assignments_count=0)
        self.assertEqual(factor.severity, "LOW")
        self.assertEqual(factor.score_weight, 0.0)

    def test_missing_coursework_scores(self):
        """Test unrecorded coursework evaluations."""
        factor = analyze_assignments_evidence(None)
        self.assertEqual(factor.severity, "NEGLIGIBLE")
        self.assertEqual(len(factor.evidence_items), 0)


class TestAnalyzeSubmissionsEvidence(unittest.TestCase):
    """Test submission habits and timeliness evidence analysis."""

    def test_critical_missing_submissions(self):
        """Test multiple unsubmitted assignments reaching critical threshold."""
        factor = analyze_submissions_evidence(missing_count=4, total_assignments=6)
        self.assertEqual(factor.category, "submissions")
        self.assertEqual(factor.severity, "CRITICAL")
        ev = factor.evidence_items[0]
        self.assertEqual(ev.metric_name, "missing_submission_count")
        self.assertEqual(ev.severity, "CRITICAL")

    def test_warning_missing_submissions(self):
        """Test 2-3 unsubmitted assignments."""
        factor = analyze_submissions_evidence(missing_count=2, total_assignments=8)
        self.assertEqual(factor.severity, "HIGH")
        ev = factor.evidence_items[0]
        self.assertEqual(ev.severity, "HIGH")

    def test_late_submissions_and_delays(self):
        """Test late submission tracking and latency."""
        factor = analyze_submissions_evidence(
            missing_count=0,
            late_count=4,
            late_rate=0.5,
            average_delay_days=4.5,
        )
        self.assertEqual(factor.severity, "HIGH")
        late_ev = next(e for e in factor.evidence_items if e.metric_name == "late_submission_count")
        self.assertEqual(late_ev.observed_value, 4)
        delay_ev = next(e for e in factor.evidence_items if e.metric_name == "average_submission_delay_days")
        self.assertEqual(delay_ev.observed_value, 4.5)

    def test_clean_submissions(self):
        """Test student with zero missing and zero late coursework."""
        factor = analyze_submissions_evidence(missing_count=0, late_count=0)
        self.assertEqual(factor.severity, "LOW")
        self.assertEqual(factor.score_weight, 0.0)


class TestAnalyzeExamsEvidence(unittest.TestCase):
    """Test summative examination performance evidence analysis."""

    def test_critical_exam_score(self):
        """Test critically low examination average (<40)."""
        factor = analyze_exams_evidence(32.5, total_exams=2)
        self.assertEqual(factor.category, "exams")
        self.assertEqual(factor.severity, "CRITICAL")
        self.assertEqual(factor.evidence_items[0].severity, "CRITICAL")

    def test_low_exam_score(self):
        """Test sub-threshold examination average (40-60)."""
        factor = analyze_exams_evidence(52.0, total_exams=2)
        self.assertEqual(factor.severity, "HIGH")
        self.assertEqual(factor.evidence_items[0].severity, "HIGH")

    def test_satisfactory_exam_score(self):
        """Test solid exam performance."""
        factor = analyze_exams_evidence(78.0, total_exams=2)
        self.assertEqual(factor.severity, "LOW")
        self.assertEqual(factor.score_weight, 0.0)

    def test_exam_vs_assignment_divergence(self):
        """Test discrepancy where coursework is high but exam performance is low."""
        factor = analyze_exams_evidence(
            average_exam_score=52.0,
            average_assignment_score=85.0,
            total_exams=2,
        )
        self.assertEqual(factor.severity, "CRITICAL")
        gap_ev = next(e for e in factor.evidence_items if e.metric_name == "exam_vs_assignment_gap")
        self.assertEqual(gap_ev.observed_value, 33.0)
        self.assertEqual(gap_ev.severity, "HIGH")


class TestAnalyzeRecentTrendsEvidence(unittest.TestCase):
    """Test temporal trajectories and recent performance momentum."""

    def test_critical_declines(self):
        """Test steep downward trend in attendance and coursework."""
        factor = analyze_recent_trends_evidence(
            attendance_trend=-24.0,
            assignment_trend=-22.0,
            exam_trend=-15.0,
        )
        self.assertEqual(factor.category, "recent_trends")
        self.assertEqual(factor.severity, "CRITICAL")
        att_ev = next(e for e in factor.evidence_items if e.metric_name == "attendance_trend")
        self.assertEqual(att_ev.severity, "CRITICAL")

    def test_moderate_decline(self):
        """Test moderate downward trajectory."""
        factor = analyze_recent_trends_evidence(attendance_trend=-12.0)
        self.assertEqual(factor.severity, "HIGH")

    def test_positive_and_stable_trends(self):
        """Test improving or stable academic trajectory."""
        factor = analyze_recent_trends_evidence(
            attendance_trend=4.0,
            assignment_trend=8.0,
        )
        self.assertEqual(factor.severity, "LOW")
        self.assertEqual(factor.score_weight, 0.0)


class TestAnalyzeStudentRootCause(unittest.TestCase):
    """Test individual student root cause investigation workflows."""

    def test_attendance_driven_review(self):
        """Test investigation for student flagged primarily due to attendance."""
        features = {
            "attendance_percentage": 48.0,
            "average_assignment_score": 78.0,
            "missing_submission_count": 0,
            "average_exam_score": 74.0,
            "attendance_trend": -15.0,
        }
        report = analyze_student_root_cause("STU_101", student_features=features)

        self.assertEqual(report.student_id, "STU_101")
        self.assertEqual(report.overall_review_indicator, "ELEVATED")
        self.assertIn("Attendance Engagement", report.primary_factor)
        self.assertGreater(len(report.all_evidence), 0)
        self.assertTrue(any("attendance" in r.lower() for r in report.supportive_recommendations))
        self.assertIn("ETHICAL NOTICE", report.non_diagnostic_disclaimer)

    def test_submission_driven_review(self):
        """Test investigation for student with coursework neglect."""
        features = {
            "attendance_percentage": 88.0,
            "average_assignment_score": 55.0,
            "missing_submission_count": 4,
            "late_submission_count": 3,
            "average_exam_score": 70.0,
        }
        report = analyze_student_root_cause("STU_102", student_features=features)

        self.assertEqual(report.overall_review_indicator, "ELEVATED")
        self.assertIn("Submission Timeliness", report.primary_factor)
        self.assertTrue(any("submission" in r.lower() for r in report.supportive_recommendations))

    def test_exam_driven_review(self):
        """Test investigation for student with exam struggles despite good attendance."""
        features = {
            "attendance_percentage": 92.0,
            "average_assignment_score": 82.0,
            "missing_submission_count": 0,
            "average_exam_score": 38.0,
        }
        report = analyze_student_root_cause("STU_103", student_features=features)

        self.assertEqual(report.overall_review_indicator, "ELEVATED")
        self.assertIn("Summative Assessment Performance", report.primary_factor)
        self.assertTrue(any("exam" in r.lower() for r in report.supportive_recommendations))

    def test_low_risk_student(self):
        """Test investigation for student with solid indicators."""
        features = {
            "attendance_percentage": 90.0,
            "average_assignment_score": 85.0,
            "missing_submission_count": 0,
            "late_submission_count": 0,
            "average_exam_score": 88.0,
            "attendance_trend": 2.0,
        }
        report = analyze_student_root_cause("STU_104", student_features=features)

        self.assertEqual(report.overall_review_indicator, "LOW")
        self.assertIsNone(report.primary_factor)
        self.assertEqual(len(report.secondary_factors), 0)

    def test_granular_records_inspection(self):
        """Test investigation using raw detailed dataframes."""
        att_df = pd.DataFrame({
            "student_id": ["STU_105", "STU_105", "STU_105", "STU_105"],
            "status": ["present", "absent", "absent", "absent"],
        })
        sub_df = pd.DataFrame({
            "student_id": ["STU_105", "STU_105"],
            "assignment_id": [1, 2],
            "score": [45.0, 50.0],
        })
        exam_df = pd.DataFrame({
            "student_id": ["STU_105"],
            "score": [42.0],
        })

        report = analyze_student_root_cause(
            student_id="STU_105",
            attendance_records=att_df,
            submission_records=sub_df,
            exam_records=exam_df,
        )

        self.assertEqual(report.student_id, "STU_105")
        self.assertEqual(report.overall_review_indicator, "ELEVATED")
        self.assertIsNotNone(report.primary_factor)

    def test_to_dict_serialization(self):
        """Test complete serialization of student report."""
        features = {"attendance_percentage": 55.0, "average_exam_score": 50.0}
        report = analyze_student_root_cause("STU_106", student_features=features)
        data = report.to_dict()

        self.assertIsInstance(data, dict)
        self.assertEqual(data["student_id"], "STU_106")
        self.assertIn("category_evaluations", data)
        self.assertIn("all_evidence", data)
        self.assertIn("non_diagnostic_disclaimer", data)


class TestAnalyzeCourseRootCause(unittest.TestCase):
    """Test course and cohort root cause investigation workflows."""

    def test_course_with_assignment_bottleneck(self):
        """Test detecting specific bottleneck assignment causing elevated student review."""
        course_students = pd.DataFrame({
            "student_id": [f"S{i}" for i in range(10)],
            "attendance_percentage": [85.0] * 10,
            "average_assignment_score": [52.0] * 10,
            "average_exam_score": [72.0] * 10,
            "missing_submission_count": [1] * 10,
        })
        subs = pd.DataFrame({
            "assignment_id": [1] * 10 + [2] * 10,
            "student_id": [f"S{i}" for i in range(10)] * 2,
            "score": [80.0] * 10 + [35.0] * 10,  # Assignment 2 has 100% fail rate
        })

        report = analyze_course_root_cause(
            course_id="CS101",
            course_student_features=course_students,
            course_submissions=subs,
        )

        self.assertEqual(report.course_id, "CS101")
        self.assertEqual(report.total_students, 10)
        self.assertIn("Assignment", report.primary_bottleneck)
        self.assertTrue(any("bottleneck coursework (2)" in r.lower() for r in report.curriculum_recommendations))

    def test_course_with_attendance_disengagement(self):
        """Test course with severe cohort-wide attendance decline."""
        course_students = pd.DataFrame({
            "student_id": [f"S{i}" for i in range(10)],
            "attendance_percentage": [45.0] * 10,
            "average_assignment_score": [75.0] * 10,
            "average_exam_score": [72.0] * 10,
            "missing_submission_count": [0] * 10,
            "risk_level": ["ELEVATED"] * 6 + ["LOW"] * 4,
        })

        report = analyze_course_root_cause("MATH201", course_student_features=course_students)
        self.assertEqual(report.course_review_indicator, "ELEVATED")
        self.assertEqual(report.elevated_students_count, 6)
        self.assertEqual(report.elevated_students_percentage, 60.0)
        self.assertIn("Attendance", report.primary_bottleneck)
        self.assertTrue(any("attendance" in r.lower() for r in report.curriculum_recommendations))

    def test_course_validation_errors(self):
        """Test input validation for course analysis."""
        with self.assertRaises(DataValidationError):
            analyze_course_root_cause("CS101", pd.DataFrame())

        with self.assertRaises(DataValidationError):
            analyze_course_root_cause("CS101", pd.DataFrame({"score": [80]}))


class TestBatchOperations(unittest.TestCase):
    """Test batch root cause analysis operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            "student_id": ["S1", "S2", "S3"],
            "course_id": ["C1", "C1", "C2"],
            "attendance_percentage": [40.0, 95.0, 60.0],
            "average_assignment_score": [45.0, 88.0, 70.0],
            "average_exam_score": [42.0, 90.0, 65.0],
            "missing_submission_count": [4, 0, 1],
        })

    def test_batch_analyze_students(self):
        """Test analyzing multiple students in batch."""
        reports = batch_analyze_students(self.df)
        self.assertEqual(len(reports), 3)
        self.assertEqual(reports[0].overall_review_indicator, "ELEVATED")
        self.assertEqual(reports[1].overall_review_indicator, "LOW")

    def test_batch_analyze_filter_elevated(self):
        """Test filtering only elevated students."""
        reports = batch_analyze_students(self.df, filter_elevated_only=True)
        self.assertTrue(all(r.overall_review_indicator == "ELEVATED" for r in reports))

    def test_identify_students_for_investigation(self):
        """Test identifying prioritized students."""
        flagged = identify_students_for_investigation(self.df)
        self.assertIn("S1", flagged["student_id"].values)
        self.assertNotIn("S2", flagged["student_id"].values)

    def test_run_root_cause_investigation(self):
        """Test end-to-end investigation runner."""
        results = run_root_cause_investigation(self.df)
        self.assertEqual(results["total_students_evaluated"], 3)
        self.assertGreaterEqual(results["elevated_students_count"], 1)
        self.assertEqual(len(results["course_reports"]), 2)
        self.assertIn("ETHICAL NOTICE", results["non_diagnostic_disclaimer"])

    def test_missing_student_id_column(self):
        """Test DataValidationError when student_id column is omitted."""
        bad_df = pd.DataFrame({"score": [90]})
        with self.assertRaises(DataValidationError):
            batch_analyze_students(bad_df)
        with self.assertRaises(DataValidationError):
            identify_students_for_investigation(bad_df)


class TestMarkdownReporting(unittest.TestCase):
    """Test markdown report generation."""

    def test_student_markdown_report(self):
        """Test formatting of student markdown report."""
        report = analyze_student_root_cause(
            "STU_200",
            student_features={
                "attendance_percentage": 45.0,
                "average_exam_score": 38.0,
                "missing_submission_count": 3,
            },
        )
        md = generate_student_root_cause_markdown(report)

        self.assertIn("# Academic Root Cause Analysis: Student STU_200", md)
        self.assertIn("Empirical Evidence Table", md)
        self.assertIn("ETHICAL NOTICE & DISCLAIMER", md)
        self.assertIn("Supportive Advising Recommendations", md)

    def test_course_markdown_report(self):
        """Test formatting of course markdown report."""
        df = pd.DataFrame({
            "student_id": ["S1", "S2"],
            "attendance_percentage": [45.0, 50.0],
            "average_exam_score": [40.0, 42.0],
        })
        course_rep = analyze_course_root_cause("CS301", course_student_features=df)
        md = generate_course_root_cause_markdown(course_rep)

        self.assertIn("# Academic Root Cause Analysis: Course CS301", md)
        self.assertIn("Cohort Empirical Evidence Table", md)
        self.assertIn("Curriculum & Pedagogical Recommendations", md)
        self.assertIn("ETHICAL NOTICE & DISCLAIMER", md)


class TestEthicalAndNonDiagnosticCompliance(unittest.TestCase):
    """Test ethical guardrails and non-diagnostic guarantees."""

    def test_disclaimer_content(self):
        """Ensure disclaimer strictly denies diagnostic intent and causal determinism."""
        self.assertIn("does NOT provide personal, psychological, or medical diagnoses", NON_DIAGNOSTIC_DISCLAIMER)
        self.assertIn("nor does it assert definitive causation", NON_DIAGNOSTIC_DISCLAIMER)

    def test_observational_language(self):
        """Ensure findings use strictly empirical, observable phrasing."""
        report = analyze_student_root_cause(
            "STU_ETHIC",
            student_features={"attendance_percentage": 42.0, "missing_submission_count": 4},
        )
        # Verify no medical or psychological terms in finding texts
        prohibited_terms = ["depressed", "adhd", "lazy", "mental", "syndrome", "disorder", "incapable"]
        all_text = " ".join([e.finding.lower() for e in report.all_evidence] + [report.factual_summary.lower()])
        for term in prohibited_terms:
            self.assertNotIn(term, all_text)

    def test_get_root_cause_thresholds(self):
        """Test fetching configuration dictionary."""
        thresholds = get_root_cause_thresholds()
        self.assertIsInstance(thresholds, dict)
        self.assertIn("attendance_low", thresholds)
        self.assertIn("exam_score_low", thresholds)


if __name__ == "__main__":
    unittest.main()
