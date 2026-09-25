"""Programmatic academic data dictionary metadata and field specifications."""

from typing import Any, Dict, List, Optional
import pandas as pd

# Comprehensive dictionary containing field specifications for all academic entities
ACADEMIC_DATA_DICTIONARY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "students": {
        "student_id": {
            "meaning": "Unique identifier for student record",
            "type": "VARCHAR(32)",
            "source": "Student Information System (SIS)",
            "expected_values": "Alphanumeric string (e.g. STU-1001)",
            "required": True,
            "analytical_usage": "Primary key; student drill-down, cohort tracking, risk attribution",
        },
        "name": {
            "meaning": "Full legal or preferred student name",
            "type": "VARCHAR(100)",
            "source": "SIS",
            "expected_values": "Non-empty string",
            "required": True,
            "analytical_usage": "Dashboard display, student search, advisor advisory logging",
        },
        "program": {
            "meaning": "Academic degree program or major",
            "type": "VARCHAR(100)",
            "source": "SIS",
            "expected_values": "Categorical program name (e.g. Computer Science, Information Technology)",
            "required": True,
            "analytical_usage": "Program-level filtering, cohort academic health comparisons",
        },
        "year": {
            "meaning": "Current academic year / level of study",
            "type": "INTEGER",
            "source": "SIS",
            "expected_values": "Integer 1 to 4",
            "required": True,
            "analytical_usage": "Cohort segmentation, year-wise engagement trend analysis",
        },
    },
    "courses": {
        "course_id": {
            "meaning": "Unique course code identifier",
            "type": "VARCHAR(32)",
            "source": "Curriculum Catalog / SIS",
            "expected_values": "Alphanumeric code (e.g. CS101, DS202)",
            "required": True,
            "analytical_usage": "Primary key; course-level analytics, subject comparison",
        },
        "course_name": {
            "meaning": "Official descriptive course title",
            "type": "VARCHAR(100)",
            "source": "SIS",
            "expected_values": "Non-empty string (e.g. Data Structures)",
            "required": True,
            "analytical_usage": "Display label across charts, filters, and student profile views",
        },
        "faculty": {
            "meaning": "Assigned faculty instructor member name",
            "type": "VARCHAR(100)",
            "source": "SIS / Faculty Directory",
            "expected_values": "Non-empty instructor name",
            "required": True,
            "analytical_usage": "Instructor filtering, course ownership attribution",
        },
    },
    "enrollments": {
        "student_id": {
            "meaning": "Enrolled student identifier",
            "type": "VARCHAR(32)",
            "source": "SIS Enrollment Records",
            "expected_values": "Foreign key referencing students.student_id",
            "required": True,
            "analytical_usage": "Roster mapping; baseline for expected student deliverables",
        },
        "course_id": {
            "meaning": "Registered course identifier",
            "type": "VARCHAR(32)",
            "source": "SIS Enrollment Records",
            "expected_values": "Foreign key referencing courses.course_id",
            "required": True,
            "analytical_usage": "Computes active course enrollments and cohort size",
        },
    },
    "attendance": {
        "student_id": {
            "meaning": "Student attending the session",
            "type": "VARCHAR(32)",
            "source": "LMS / Classroom RFID",
            "expected_values": "Foreign key referencing students.student_id",
            "required": True,
            "analytical_usage": "Student-level attendance rate aggregation",
        },
        "course_id": {
            "meaning": "Course session identifier",
            "type": "VARCHAR(32)",
            "source": "LMS Attendance",
            "expected_values": "Foreign key referencing courses.course_id",
            "required": True,
            "analytical_usage": "Course-level attendance percentage aggregation",
        },
        "date": {
            "meaning": "Calendar date on which session took place",
            "type": "DATE",
            "source": "LMS Attendance Timestamp",
            "expected_values": "YYYY-MM-DD calendar date",
            "required": True,
            "analytical_usage": "Time-series trend calculation; recent 4-week moving trend",
        },
        "status": {
            "meaning": "Session attendance status classification",
            "type": "VARCHAR(20)",
            "source": "LMS Attendance",
            "expected_values": "Present, Absent, Late, Excused",
            "required": True,
            "analytical_usage": "Attendance percentage: (Present + 0.5 * Late) / Total Sessions; concern < 75%",
        },
    },
    "assignments": {
        "assignment_id": {
            "meaning": "Unique coursework assignment identifier",
            "type": "VARCHAR(32)",
            "source": "LMS Course Syllabus",
            "expected_values": "Alphanumeric code (e.g. ASN-01)",
            "required": True,
            "analytical_usage": "Primary key; identifies continuous assessment items",
        },
        "course_id": {
            "meaning": "Associated course identifier",
            "type": "VARCHAR(32)",
            "source": "LMS",
            "expected_values": "Foreign key referencing courses.course_id",
            "required": True,
            "analytical_usage": "Maps coursework deliverables to specific course syllabus",
        },
        "title": {
            "meaning": "Descriptive title of coursework task",
            "type": "VARCHAR(150)",
            "source": "LMS",
            "expected_values": "Descriptive text string",
            "required": True,
            "analytical_usage": "Display in student profile assignment history table",
        },
        "due_date": {
            "meaning": "Submission deadline date",
            "type": "DATE",
            "source": "LMS Calendar",
            "expected_values": "YYYY-MM-DD",
            "required": True,
            "analytical_usage": "Baseline for timeliness; identifies past-due missing tasks",
        },
    },
    "submissions": {
        "assignment_id": {
            "meaning": "Associated coursework assignment identifier",
            "type": "VARCHAR(32)",
            "source": "LMS Submission Log",
            "expected_values": "Foreign key referencing assignments.assignment_id",
            "required": True,
            "analytical_usage": "Links score to assignment deliverable",
        },
        "student_id": {
            "meaning": "Student submitting coursework",
            "type": "VARCHAR(32)",
            "source": "LMS Submission Log",
            "expected_values": "Foreign key referencing students.student_id",
            "required": True,
            "analytical_usage": "Computes student completion rate: completed / total assignments",
        },
        "submission_date": {
            "meaning": "Date coursework was submitted",
            "type": "DATE",
            "source": "LMS Submission Timestamp",
            "expected_values": "YYYY-MM-DD (nullable if unsubmitted)",
            "required": False,
            "analytical_usage": "Timeliness evaluation (submission_date > due_date = Late)",
        },
        "score": {
            "meaning": "Graded assessment score",
            "type": "FLOAT",
            "source": "Faculty Gradebook",
            "expected_values": "0.0 to 100.0 (nullable if ungraded/unsubmitted)",
            "required": False,
            "analytical_usage": "Assignment average score; unrecorded scores are NOT zero performance",
        },
    },
    "exams": {
        "exam_id": {
            "meaning": "Unique examination identifier",
            "type": "VARCHAR(32)",
            "source": "Assessment Portal",
            "expected_values": "Alphanumeric code (e.g. EXM-01)",
            "required": True,
            "analytical_usage": "Primary key; identifies formal assessment event",
        },
        "student_id": {
            "meaning": "Student taking the examination",
            "type": "VARCHAR(32)",
            "source": "Assessment Portal",
            "expected_values": "Foreign key referencing students.student_id",
            "required": True,
            "analytical_usage": "Student attribution for summative assessment scores",
        },
        "course_id": {
            "meaning": "Course being evaluated",
            "type": "VARCHAR(32)",
            "source": "Assessment Portal",
            "expected_values": "Foreign key referencing courses.course_id",
            "required": True,
            "analytical_usage": "Course-level exam average and grade distribution",
        },
        "exam_type": {
            "meaning": "Category of examination",
            "type": "VARCHAR(50)",
            "source": "Assessment Portal",
            "expected_values": "Midterm, Final, Quiz, Practical",
            "required": True,
            "analytical_usage": "Weighting and examination type segmentation",
        },
        "score": {
            "meaning": "Examination score percentage",
            "type": "FLOAT",
            "source": "Grading Registry",
            "expected_values": "0.0 to 100.0 (nullable if absent)",
            "required": False,
            "analytical_usage": "Exam average score; concern trigger when exam avg < 50% or decline > 15%",
        },
    },
    "interventions": {
        "id": {
            "meaning": "Unique intervention follow-up log record identifier",
            "type": "INTEGER",
            "source": "Dashboard Database",
            "expected_values": "Positive integer",
            "required": True,
            "analytical_usage": "Primary key for logged advisory support interaction",
        },
        "student_id": {
            "meaning": "Target student receiving academic support",
            "type": "VARCHAR(32)",
            "source": "Advisor Form",
            "expected_values": "Foreign key referencing students.student_id",
            "required": True,
            "analytical_usage": "Associates logged action with student profile",
        },
        "date": {
            "meaning": "Date on which follow-up or check-in occurred",
            "type": "DATE",
            "source": "Advisor Form",
            "expected_values": "YYYY-MM-DD",
            "required": True,
            "analytical_usage": "Chronological audit trail of advisor support frequency",
        },
        "type": {
            "meaning": "Categorization of intervention support method",
            "type": "VARCHAR(50)",
            "source": "Advisor Form",
            "expected_values": "Academic check-in, Faculty discussion, Study support, Other",
            "required": True,
            "analytical_usage": "Categorization for institutional support review and reporting",
        },
        "status": {
            "meaning": "Current workflow lifecycle status",
            "type": "VARCHAR(30)",
            "source": "Advisor Form",
            "expected_values": "Open, In Progress, Resolved",
            "required": True,
            "analytical_usage": "Action item tracking; surfaces open follow-ups on advisor dashboard",
        },
        "notes": {
            "meaning": "Qualitative faculty advisory discussion notes",
            "type": "TEXT",
            "source": "Advisor Form",
            "expected_values": "Free-form text (up to 2000 characters)",
            "required": True,
            "analytical_usage": "Contextual documentation of agreed support plan and actions",
        },
    },
}


def get_data_dictionary() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return the complete academic data dictionary specification."""
    return ACADEMIC_DATA_DICTIONARY


def get_entity_fields(entity_name: str) -> List[Dict[str, Any]]:
    """Return a list of field specifications for a given entity."""
    clean_entity = entity_name.strip().lower()
    entity_dict = ACADEMIC_DATA_DICTIONARY.get(clean_entity, {})
    return [{"field_name": k, **v} for k, v in entity_dict.items()]


def get_field_spec(entity_name: str, field_name: str) -> Optional[Dict[str, Any]]:
    """Return field specification for a specific entity and field name."""
    clean_entity = entity_name.strip().lower()
    clean_field = field_name.strip().lower()
    return ACADEMIC_DATA_DICTIONARY.get(clean_entity, {}).get(clean_field)


def get_data_dictionary_dataframe(entity_name: Optional[str] = None) -> pd.DataFrame:
    """Return a flat pandas DataFrame representing the data dictionary for tabular display.

    Args:
        entity_name: Optional filter for a specific entity. If None, returns all entities.

    Returns:
        pd.DataFrame formatted for presentation tables.
    """
    rows = []
    entities_to_process = (
        [entity_name.strip().lower()]
        if entity_name and entity_name.strip().lower() in ACADEMIC_DATA_DICTIONARY
        else list(ACADEMIC_DATA_DICTIONARY.keys())
    )

    for ent in entities_to_process:
        for field_name, details in ACADEMIC_DATA_DICTIONARY[ent].items():
            rows.append({
                "entity": ent,
                "field_name": field_name,
                "type": details["type"],
                "meaning": details["meaning"],
                "source": details["source"],
                "expected_values": details["expected_values"],
                "required": "Yes" if details["required"] else "No",
                "analytical_usage": details["analytical_usage"],
            })

    return pd.DataFrame(rows)
