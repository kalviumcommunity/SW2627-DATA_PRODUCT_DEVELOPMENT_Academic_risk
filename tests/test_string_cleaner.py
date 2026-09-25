"""Unit tests for Concept #11: Academic String Cleaning & Categorical Normalization."""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.string_cleaner import (
    ACADEMIC_ACRONYMS,
    CANONICAL_ATTENDANCE_STATUS,
    CANONICAL_ASSIGNMENT_STATUS,
    CANONICAL_EXAM_TYPES,
    CANONICAL_FACULTY,
    CANONICAL_PROGRAMS,
    StringCleaningAudit,
    academic_title_case,
    clean_academic_strings,
    clean_assignment_status,
    clean_assignments_strings,
    clean_attendance_status,
    clean_attendance_strings,
    clean_course_name,
    clean_courses_strings,
    clean_exam_type,
    clean_exams_strings,
    clean_faculty_name,
    clean_id,
    clean_id_series,
    clean_program_name,
    clean_students_strings,
    clean_submissions_strings,
    clean_whitespace,
)


class TestStringCleaner:
    """Test suite for text normalization, acronym preservation, and ID protection."""

    def test_clean_whitespace(self):
        """Test whitespace stripping and collapse across spaces, tabs, newlines, and nbsp."""
        assert clean_whitespace("  Computer    Science  ") == "Computer Science"
        assert clean_whitespace("Line 1\n\r\tLine 2") == "Line 1 Line 2"
        assert clean_whitespace("Text\u00a0with\u00a0nbsp") == "Text with nbsp"
        assert clean_whitespace("   ") is None
        assert clean_whitespace(None) is None
        assert clean_whitespace(np.nan) is None

    def test_clean_id_protection_and_rules(self):
        """Test that IDs are standardized without incorrect casing alteration or corruption."""
        # Must uppercase, remove quotes, strip whitespace
        assert clean_id("  stu_001  ") == "STU_001"
        assert clean_id("'cs101'") == "CS101"
        assert clean_id('"eng_202"') == "ENG_202"
        # Must strip trailing .0 from floating point string representations
        assert clean_id("1001.0") == "1001"
        assert clean_id("S0123.0") == "S0123"
        # Must NEVER title case IDs (e.g. CS101 must not become Cs101)
        assert clean_id("cs101") == "CS101"
        assert clean_id("it201") == "IT201"
        assert clean_id("exam-mid-01") == "EXAM-MID-01"
        # Null values
        assert clean_id(None) is None
        assert clean_id(np.nan) is None

    def test_academic_title_case_and_acronyms(self):
        """Test intelligent title casing with acronyms and minor words handling."""
        # Basic title casing
        assert academic_title_case("intro to programming") == "Intro to Programming"
        # Acronyms preserved in full uppercase
        assert academic_title_case("machine learning and ai") == "Machine Learning and AI"
        assert academic_title_case("advanced ml and dsa") == "Advanced ML and DSA"
        assert academic_title_case("intro to cs and it") == "Intro to CS and IT"
        # Roman numerals preserved
        assert academic_title_case("calculus ii") == "Calculus II"
        assert academic_title_case("physics iv") == "Physics IV"
        # Underscores converted to spaces
        assert academic_title_case("database_management_systems") == "Database Management Systems"
        # Minor word as first word capitalized
        assert academic_title_case("the art of computing") == "The Art of Computing"

    def test_clean_program_name(self):
        """Test program name normalization against canonical degrees."""
        assert clean_program_name("btech cse") == "B.Tech Computer Science & Engineering"
        assert clean_program_name("B.Tech CS") == "B.Tech Computer Science & Engineering"
        assert clean_program_name("computer science and engineering") == "B.Tech Computer Science & Engineering"
        assert clean_program_name("btech it") == "B.Tech Information Technology"
        assert clean_program_name("bba") == "Bachelor of Business Administration"
        assert clean_program_name("bsc data science") == "B.Sc Data Science"
        # General unlisted program receives proper title casing
        assert clean_program_name("master of robotics and ai") == "Master of Robotics and AI"
        assert clean_program_name(None) is None

    def test_clean_course_name(self):
        """Test course name cleaning with code prefixes and roman numerals."""
        assert clean_course_name("cs101: data structures and algorithms") == "CS101: Data Structures and Algorithms"
        assert clean_course_name("linear_algebra_and_calculus_ii") == "Linear Algebra and Calculus II"
        assert clean_course_name("  software engineering  ") == "Software Engineering"
        assert clean_course_name(None) is None

    def test_clean_faculty_name(self):
        """Test faculty and department name normalization."""
        assert clean_faculty_name("engg") == "Engineering"
        assert clean_faculty_name("faculty of science") == "Sciences"
        assert clean_faculty_name("school of computer science") == "Computer Science"
        assert clean_faculty_name("business administration") == "Business Administration"
        # Honorifics
        assert clean_faculty_name("dr. alice smith") == "Dr. Alice Smith"
        assert clean_faculty_name("prof. bob jones") == "Prof. Bob Jones"
        assert clean_faculty_name(None) is None

    def test_clean_attendance_status(self):
        """Test canonical mapping of attendance statuses."""
        assert clean_attendance_status("P") == "Present"
        assert clean_attendance_status("present") == "Present"
        assert clean_attendance_status("1") == "Present"
        assert clean_attendance_status("A") == "Absent"
        assert clean_attendance_status("absent") == "Absent"
        assert clean_attendance_status("0") == "Absent"
        assert clean_attendance_status("L") == "Late"
        assert clean_attendance_status("tardy") == "Late"
        assert clean_attendance_status("excused") == "Excused"
        assert clean_attendance_status("medical") == "Excused"
        assert clean_attendance_status("unrecorded") == "Unrecorded"
        assert clean_attendance_status("n/a") == "Unrecorded"
        assert clean_attendance_status(None) == "Unrecorded"

    def test_clean_assignment_status(self):
        """Test canonical mapping of assignment/submission statuses."""
        assert clean_assignment_status("submitted") == "Submitted"
        assert clean_assignment_status("turned in") == "Submitted"
        assert clean_assignment_status("turned_in") == "Submitted"
        assert clean_assignment_status("late") == "Late"
        assert clean_assignment_status("submitted late") == "Late"
        assert clean_assignment_status("pending") == "Pending"
        assert clean_assignment_status("in-progress") == "Pending"
        assert clean_assignment_status("missing") == "Not Submitted"
        assert clean_assignment_status("not submitted") == "Not Submitted"
        assert clean_assignment_status("excused") == "Excused"
        assert clean_assignment_status("graded") == "Graded"
        assert clean_assignment_status(None) == "Not Submitted"

    def test_clean_exam_type(self):
        """Test canonical mapping of exam types."""
        assert clean_exam_type("midterm") == "Midterm"
        assert clean_exam_type("mid-term") == "Midterm"
        assert clean_exam_type("midterm 1") == "Midterm 1"
        assert clean_exam_type("mid-term 2") == "Midterm 2"
        assert clean_exam_type("final") == "Final"
        assert clean_exam_type("finals") == "Final"
        assert clean_exam_type("quiz") == "Quiz"
        assert clean_exam_type("quiz 1") == "Quiz 1"
        assert clean_exam_type("lab test") == "Practical"
        assert clean_exam_type("practical") == "Practical"
        assert clean_exam_type("assignment") == "Assignment"
        assert clean_exam_type(None) == "Midterm"

    def test_clean_students_strings_protects_id(self):
        """Test that clean_students_strings cleans text while protecting student_id."""
        df = pd.DataFrame({
            "student_id": ["stu-001", "stu_002.0"],
            "name": ["alice smith", "BOB JONES"],
            "program": ["btech cse", "bba"],
        })
        cleaned, audit = clean_students_strings(df)

        # IDs properly standardized to uppercase without corruption
        assert cleaned.loc[0, "student_id"] == "STU-001"
        assert cleaned.loc[1, "student_id"] == "STU_002"

        # Names title cased
        assert cleaned.loc[0, "name"] == "Alice Smith"
        assert cleaned.loc[1, "name"] == "Bob Jones"

        # Programs mapped to canonical
        assert cleaned.loc[0, "program"] == "B.Tech Computer Science & Engineering"
        assert cleaned.loc[1, "program"] == "Bachelor of Business Administration"

        assert "student_id" in audit.protected_id_columns
        assert audit.total_modifications > 0
        md = audit.to_markdown()
        assert "String Cleaning Audit: `students`" in md

    def test_clean_courses_strings(self):
        """Test clean_courses_strings with course_id protection."""
        df = pd.DataFrame({
            "course_id": ["cs101.0", " math201 "],
            "course_name": ["intro_to_ai", "calculus_ii"],
            "faculty": ["engg", "sci"],
        })
        cleaned, audit = clean_courses_strings(df)

        assert cleaned.loc[0, "course_id"] == "CS101"
        assert cleaned.loc[1, "course_id"] == "MATH201"
        assert cleaned.loc[0, "course_name"] == "Intro to AI"
        assert cleaned.loc[1, "course_name"] == "Calculus II"
        assert cleaned.loc[0, "faculty"] == "Engineering"
        assert cleaned.loc[1, "faculty"] == "Sciences"

    def test_clean_attendance_strings(self):
        """Test clean_attendance_strings cleans status and protects IDs."""
        df = pd.DataFrame({
            "student_id": ["s01", "s02"],
            "course_id": ["cs101", "cs101"],
            "status": ["p", " absent "],
        })
        cleaned, audit = clean_attendance_strings(df)

        assert cleaned.loc[0, "student_id"] == "S01"
        assert cleaned.loc[0, "status"] == "Present"
        assert cleaned.loc[1, "status"] == "Absent"

    def test_clean_assignments_and_submissions_strings(self):
        """Test cleaning assignments titles and submissions statuses."""
        df_assign = pd.DataFrame({
            "assignment_id": ["a01"],
            "course_id": ["cs101"],
            "title": ["homework 1: ai and ml basics"],
        })
        cleaned_a, _ = clean_assignments_strings(df_assign)
        assert cleaned_a.loc[0, "assignment_id"] == "A01"
        assert cleaned_a.loc[0, "title"] == "Homework 1: AI and ML Basics"

        df_sub = pd.DataFrame({
            "assignment_id": ["a01", "a01"],
            "student_id": ["s01", "s02"],
            "status": ["turned_in", "missing"],
        })
        cleaned_s, _ = clean_submissions_strings(df_sub)
        assert cleaned_s.loc[0, "status"] == "Submitted"
        assert cleaned_s.loc[1, "status"] == "Not Submitted"

    def test_clean_exams_strings(self):
        """Test clean_exams_strings normalizes exam types."""
        df_exams = pd.DataFrame({
            "exam_id": ["e01", "e02"],
            "student_id": ["s01", "s02"],
            "course_id": ["cs101", "cs101"],
            "exam_type": ["mid-term 1", "finals"],
        })
        cleaned_e, audit = clean_exams_strings(df_exams)
        assert cleaned_e.loc[0, "exam_id"] == "E01"
        assert cleaned_e.loc[0, "exam_type"] == "Midterm 1"
        assert cleaned_e.loc[1, "exam_type"] == "Final"

    def test_clean_academic_strings_batch(self):
        """Test batch string cleaning across multiple academic datasets."""
        batch = {
            "students": pd.DataFrame({
                "student_id": ["s01"],
                "name": ["alice smith"],
                "program": ["btech cse"],
            }),
            "courses": pd.DataFrame({
                "course_id": ["cs101"],
                "course_name": ["intro_to_cs"],
                "faculty": ["engg"],
            }),
            "attendance": pd.DataFrame({
                "student_id": ["s01"],
                "course_id": ["cs101"],
                "status": ["p"],
            }),
        }

        cleaned_batch, audits = clean_academic_strings(batch)
        assert len(cleaned_batch) == 3
        assert cleaned_batch["students"].loc[0, "student_id"] == "S01"
        assert cleaned_batch["students"].loc[0, "program"] == "B.Tech Computer Science & Engineering"
        assert cleaned_batch["courses"].loc[0, "course_name"] == "Intro to CS"
        assert cleaned_batch["attendance"].loc[0, "status"] == "Present"
        assert len(audits) == 3
