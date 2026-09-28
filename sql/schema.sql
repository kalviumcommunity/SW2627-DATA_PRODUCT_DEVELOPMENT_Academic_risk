-- ============================================================================
-- Academic Engagement Risk Dashboard - Relational Schema DDL
-- Database: SQLite 3
-- Entities: students, courses, enrollments, attendance, assignments,
--           submissions, exams, interventions
-- ============================================================================

-- Enforce foreign key constraints
PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- 1. Table: students
-- Stores core student demographics, academic programs, and cohort levels.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    program TEXT NOT NULL,
    year INTEGER NOT NULL CHECK (year >= 1 AND year <= 8)
);

-- ----------------------------------------------------------------------------
-- 2. Table: courses
-- Academic course offerings, syllabus titles, and assigned faculty instructors.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    faculty TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- 3. Table: enrollments
-- Associative table linking students with their registered courses.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS enrollments (
    student_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- 4. Table: attendance
-- Session-level attendance records tracking presence, absence, and tardiness.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('Present', 'Absent', 'Late', 'Excused', 'Unrecorded')),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- 5. Table: assignments
-- Formative coursework deliverables and project deadlines.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assignments (
    assignment_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    title TEXT NOT NULL,
    due_date TEXT NOT NULL,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- 6. Table: submissions
-- Student assignment submissions, turnaround dates, and evaluative scores.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS submissions (
    assignment_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    submission_date TEXT,
    score REAL CHECK (score IS NULL OR (score >= 0.0 AND score <= 100.0)),
    PRIMARY KEY (assignment_id, student_id),
    FOREIGN KEY (assignment_id) REFERENCES assignments(assignment_id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- 7. Table: exams
-- Summative evaluations including midterms, finals, quizzes, and practicals.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exams (
    exam_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    exam_type TEXT NOT NULL,
    score REAL CHECK (score IS NULL OR (score >= 0.0 AND score <= 100.0)),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- 8. Table: interventions
-- Academic advisory logs, student check-ins, action plans, and resolutions.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    date TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
    notes TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- ============================================================================
-- PERFORMANCE & LOOKUP INDEXES
-- ============================================================================

-- Enrollments indexes
CREATE INDEX IF NOT EXISTS idx_enrollments_student ON enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_course ON enrollments(course_id);

-- Attendance indexes
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_course ON attendance(course_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_student_course ON attendance(student_id, course_id);

-- Assignments indexes
CREATE INDEX IF NOT EXISTS idx_assignments_course ON assignments(course_id);
CREATE INDEX IF NOT EXISTS idx_assignments_due_date ON assignments(due_date);

-- Submissions indexes
CREATE INDEX IF NOT EXISTS idx_submissions_student ON submissions(student_id);
CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON submissions(assignment_id);

-- Exams indexes
CREATE INDEX IF NOT EXISTS idx_exams_student ON exams(student_id);
CREATE INDEX IF NOT EXISTS idx_exams_course ON exams(course_id);
CREATE INDEX IF NOT EXISTS idx_exams_student_course ON exams(student_id, course_id);

-- Interventions indexes
CREATE INDEX IF NOT EXISTS idx_interventions_student ON interventions(student_id);
CREATE INDEX IF NOT EXISTS idx_interventions_date ON interventions(date);
CREATE INDEX IF NOT EXISTS idx_interventions_status ON interventions(status);
