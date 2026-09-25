# Academic Student Feature Engineering Documentation

This document specifies the mathematical formulations, source datasets, operational definitions, and academic risk significance of every student-level feature engineered to feed the explainable academic risk engine.

---

## 1. Feature Specifications & Mathematical Formulations

| Feature Name | Mathematical Formula | Valid Range | Missingness Rule | Academic Risk Significance |
| :--- | :--- | :--- | :--- | :--- |
| `attendance_percentage` | $\frac{\text{Present} + 0.5 \times \text{Late}}{\text{Sessions} - \text{Excused}} \times 100$ | $[0.0, 100.0]$ | Preserved as `NaN` if 0 sessions | $< 75.0\%$ represents severe chronic absenteeism risk |
| `assignment_completion_rate` | $\frac{\text{Submitted Coursework}}{\text{Total Expected Assignments}} \times 100$ | $[0.0, 100.0]$ | $0.0$ if 0 submitted | $< 70.0\%$ signals coursework neglect and disengagement |
| `average_assignment_score` | $\frac{1}{N_{\text{sub}}} \sum_{i=1}^{N_{\text{sub}}} \text{score}_i$ | $[0.0, 100.0]$ | Preserved as `NaN` if unattempted | $< 60.0\%$ signals conceptual mastery deficiencies |
| `missing_submission_count` | $\max(0, \text{Expected} - \text{Submitted})$ | Integer $\ge 0$ | $0$ if all submitted | $\ge 2$ indicates chronic assignment abandonment |
| `late_submission_count` | $\sum \mathbb{I}(\text{is\_late} = \text{True})$ | Integer $\ge 0$ | $0$ if none late | High counts indicate time management / workload stress |
| `average_exam_score` | $\frac{1}{N_{\text{exams}}} \sum_{i=1}^{N_{\text{exams}}} \text{score}_i$ | $[0.0, 100.0]$ | Preserved as `NaN` if unattempted | $< 50.0\%$ indicates acute summative failure risk |
| `recent_exam_score` | $\text{score of newest chronological exam}$ | $[0.0, 100.0]$ | Preserved as `NaN` if unattempted | Reflects immediate current trajectory |
| `attendance_trend` | $\text{Rate}_{\text{recent half}} - \text{Rate}_{\text{early half}}$ | $[-100.0, +100.0]$ | Preserved as `NaN` if $< 2$ sessions | Negative values ($< -5.0$) indicate disengagement deterioration |
| `assignment_trend` | $\text{Mean}_{\text{recent scores}} - \text{Mean}_{\text{early scores}}$ | $[-100.0, +100.0]$ | Preserved as `NaN` if $< 2$ submissions | Negative values ($< -5.0$) indicate falling academic performance |
| `exam_trend` | $\text{Recent Exam Score} - \text{Initial Exam Score}$ | $[-100.0, +100.0]$ | $0.0$ if 1 exam; `NaN` if 0 exams | Negative values reveal drops between midterms and subsequent exams |

---

## 2. Detailed Component Calculations

### 2.1 Attendance Metrics
- **Data Source**: `attendance` table joined on `student_id`.
- **Status Weighting**:
  - `Present`: Weight $1.0$.
  - `Late`: Weight $0.5$ (arrived late but received instruction).
  - `Excused`: Excluded from both numerator and denominator (authorized medical / institutional leave).
  - `Absent`: Weight $0.0$.
  - `Unrecorded`: Ignored from numerator.
- **Longitudinal Trend**: Class sessions are sorted chronologically. The sessions are partitioned into an early half and a recent half. The percentage-point difference indicates momentum.

### 2.2 Coursework & Assignment Metrics
- **Data Sources**: `submissions`, `assignments`, and `enrollments`.
- **Expected Coursework Determination**: Total expected assignments are determined by summing all coursework associated with the student's actively enrolled courses.
- **Missing vs Late Tracking**:
  - `missing_submission_count`: Unfulfilled required assignments.
  - `late_submission_count`: Coursework submitted past deadline.
- **Score Averaging**: Computed strictly across evaluated submissions. Never converts missing coursework to zero score in the average score metric; missingness is represented cleanly through `missing_submission_count` and `assignment_completion_rate`.

### 2.3 Exam Summative Metrics
- **Data Source**: `exams` table.
- **Recent Exam Selection**: Identifies the exam sitting with the latest timestamp/date for the student.
- **Exam Trend**: Measures performance change between the initial examination (e.g. Midterm 1) and the most recent examination sitting.

---

## 3. Strict Missingness & Academic Integrity Rules
1. **Never Impute Missing Attendance as Zero**: A student with no recorded attendance records is given `NaN`, distinct from a student with 0% attendance.
2. **Never Impute Missing Exams as Zero**: An unattempted exam remains `NaN`.
3. **Decoupled Architecture**: All feature logic resides in `src/feature_engineering.py` for headless unit testing and reuse by downstream risk models.
