Academic Engagement Risk Dashboard

College Project — PRD + Lightweight LLD + Google Stitch UI Prompt

Project Type: College / Semester Data Product Project
Sprint Context: Sprint 1 — End-to-End Data Product: Dataset to Insights
Expected Duration: 4–5 weeks / up to 20 working days
Team Size: 2–3 students
Primary Users: Faculty / Academic Advisor
Project Status: MVP Design

1. Executive Summary

1.1 Problem Statement

A university/college maintains attendance records, assignment submissions, and exam performance data, but these records are usually viewed separately. This makes it difficult to identify engagement patterns that may indicate academic risk early enough for faculty to review them.

The proposed product is a small academic analytics dashboard that combines these datasets and converts them into:

Student engagement metrics

Attendance and submission trends

Academic performance trends

A simple, explainable risk indicator

Student-level drill-downs

Basic faculty/advisor insights

The project is intentionally scoped as a college-level data product, not as a production university platform.

1.2 Goal

Build an end-to-end data product that demonstrates:

Dataset → Cleaning → SQL/Data Analysis → Risk Logic → Insights → Interactive Application

1.3 Goals

Combine attendance, assignments, and exam data.

Clean and validate the dataset.

Generate meaningful student-level and course-level metrics.

Identify simple engagement patterns associated with academic risk.

Provide an explainable risk indicator.

Build an interactive dashboard.

Demonstrate SQL querying and exploratory analysis.

Present insights clearly during the final demo/viva.

1.4 Non-Goals

The project will not:

Make official academic decisions.

Automatically fail/pass students.

Diagnose students.

Build a university-wide production system.

Require real-time streaming.

Require complex microservices.

Build a sophisticated generative-AI chatbot.

Claim that the risk score is a scientifically validated prediction unless it has actually been validated against appropriate historical outcomes.

1.5 Architecture Principles

Keep the architecture simple.

Prefer explainable rules over unnecessary ML complexity.

Separate data cleaning, analysis, risk logic, and UI.

Make every dashboard insight traceable to source data.

Show trends and evidence, not just a score.

Keep AI optional and secondary to the data analysis.

Design for a 20-working-day college project, not enterprise scale.

2. Users, Use Cases & MVP Scope

2.1 Primary User

Faculty / Academic Advisor

The user should be able to:

Open the dashboard.

See an overview of students and risk indicators.

Search/filter students.

Open an individual student.

Understand why the student was surfaced.

View attendance, assignment, and exam trends.

Record a simple follow-up/intervention note.

2.2 Core Use Cases

UC-01 — View Overall Academic Health

The user sees:

Total students

Average attendance

Assignment completion

Average exam score

Students requiring review

UC-02 — Find Students Requiring Review

The user filters by:

Risk level

Course

Year

Attendance

Assignment completion

UC-03 — Investigate a Student

The user views:

Student profile

Attendance

Assignment submissions

Exam performance

Trends

Risk drivers

UC-04 — Record Follow-Up

The user records:

Follow-up type

Date

Notes

Status

3. Data Model & Data Pipeline

3.1 Input Data

The MVP should work with a small, realistic dataset containing:

Student

student_id

name

program

year

Course

course_id

course_name

faculty

Enrollment

student_id

course_id

Attendance

student_id

course_id

date

attendance_status

Assignment

assignment_id

student_id

course_id

due_date

submission_date

score

Exam

exam_id

student_id

course_id

exam_type

score

3.2 Pipeline

Raw CSV / Dataset
       ↓
Data Cleaning
       ↓
Validation
       ↓
Normalized Tables
       ↓
SQL Queries / Aggregations
       ↓
Student Features
       ↓
Risk Logic
       ↓
Dashboard

3.3 Data Quality Checks

At minimum:

Remove duplicate records.

Handle missing values.

Validate score ranges.

Validate attendance values.

Ensure student/course IDs exist.

Handle missing assignment submissions explicitly.

Document assumptions.

Do not interpret missing data as negative performance automatically.

4. Risk & Analytics Logic

4.1 MVP Approach

Use a simple rule-based risk indicator rather than starting with a complex ML model.

Possible signals:

Attendance percentage

Assignment completion percentage

Late/missing submissions

Average exam score

Recent performance trend

4.2 Example Risk Logic

An example configuration:

Attendance < 75%                  → attendance concern
Assignment completion < 70%       → submission concern
Average exam score < 50%          → performance concern
Recent score decline > 15%       → performance trend concern

A student can be surfaced when multiple concerns occur.

Example:

Risk Level: Elevated

Reasons:
- Attendance: 68%
- Assignment completion: 62%
- Exam average: 54%

These thresholds are project assumptions and should be stated in the final presentation. They should not be presented as universally valid academic standards.

4.3 Risk Levels

Use:

Low

Moderate

Elevated

Avoid overly dramatic labels.

4.4 Explainability

Every risk indicator should show the underlying reasons.

Example:

Elevated

Main signals:
↓ Attendance: 88% → 71%
↓ Assignment completion: 92% → 67%
↓ Exam average: 74% → 58%

5. Layer-by-Layer Architecture

Keep the architecture lightweight.

┌─────────────────────────────┐
│        Presentation         │
│ Dashboard / Student Views   │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       Application Layer     │
│ Filters / Metrics / Actions │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Analytics Layer        │
│ SQL / Pandas / Risk Rules   │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│        Data Layer           │
│ CSV / SQLite / PostgreSQL   │
└─────────────────────────────┘

5.1 Presentation Layer

Responsible for:

Dashboard

Filters

Tables

Charts

Student details

Intervention form

5.2 Application Layer

Responsible for:

User selections

Filtering

Metric retrieval

Navigation

Form handling

5.3 Analytics Layer

Responsible for:

Aggregations

Feature calculations

SQL queries

Risk rules

Trend calculations

5.4 Data Layer

Responsible for:

Raw dataset

Clean dataset

Normalized tables

Stored intervention notes

6. Request Flow & User Flow

6.1 Dashboard Flow

User opens application
        ↓
Load cleaned dataset
        ↓
Calculate / load summary metrics
        ↓
Display dashboard

6.2 Student Investigation Flow

Dashboard
   ↓
Search / Filter
   ↓
Select Student
   ↓
Student Profile
   ↓
View Evidence
   ↓
Review Risk Drivers
   ↓
Optional Follow-Up Note

6.3 Risk Calculation Flow

Attendance
Assignments
Exams
    ↓
Feature Calculation
    ↓
Rule Evaluation
    ↓
Risk Level
    ↓
Risk Reasons
    ↓
Dashboard

7. Technology Choices

Choose technologies that the team can realistically complete within the sprint.

Recommended Stack

Data Analysis

Python

Pandas

NumPy

Database / SQL

SQLite for simplest MVP

PostgreSQL if the team already knows it

Visualization / Application

Recommended for this project:

Streamlit

Plotly

This minimizes frontend/backend overhead and keeps the focus on the data product.

Optional ML

scikit-learn

Only use it if the dataset contains an appropriate historical target and the team has enough time to validate it.

Version Control

Git

GitHub

8. AI & Prompt Services

AI is optional in the MVP.

The core risk logic must remain deterministic and explainable.

Optional AI Feature

Use an LLM only to convert already-calculated evidence into a short advisor-friendly summary.

Example input:

Attendance: 71%
Attendance change: -17 percentage points

Assignment completion: 64%

Exam average: 58%
Exam trend: -12 percentage points

Possible output:

The student's recent engagement shows declining attendance and
assignment completion, alongside lower assessment performance.
These signals may warrant a faculty/advisor check-in.

Prompt

You are an academic analytics assistant.

Summarize the supplied academic engagement data for a faculty
member.

Rules:
- Use only the provided data.
- Do not invent information.
- Do not diagnose the student.
- Do not make disciplinary recommendations.
- Do not make claims beyond the evidence.
- Clearly describe observations as observations.

Student evidence:
{{student_metrics}}

Return:
1. Short summary
2. Main observed signals
3. Neutral follow-up consideration

If the LLM is unavailable, the application must still work using the structured metrics.

9. Failure Handling & Edge Cases

Because this is a college MVP, failure handling should be simple but visible.

Missing Data

Display:

Attendance data unavailable

rather than treating it as zero.

Invalid Records

Exclude invalid rows and show a data-quality summary.

Empty Search

Display:

No students match the selected filters.

Missing Student Metrics

Display:

Insufficient data

Dashboard Error

Display a clear error message and allow the user to retry.

AI Failure

Display:

AI summary unavailable.
Structured academic metrics are still available.

10. Database & SQL Design

For the MVP, use a small relational schema.

students
--------
student_id
name
program
year

courses
-------
course_id
course_name
faculty

enrollments
-----------
student_id
course_id

attendance
----------
student_id
course_id
date
status

assignments
-----------
assignment_id
course_id
title
due_date

submissions
-----------
assignment_id
student_id
submission_date
score

exams
-----
exam_id
student_id
course_id
exam_type
score

interventions
-------------
id
student_id
date
type
status
notes

Example SQL Questions

The project should demonstrate queries such as:

-- Average attendance by course
SELECT
    course_id,
    AVG(attendance_percentage) AS avg_attendance
FROM attendance_summary
GROUP BY course_id;

-- Students with low assignment completion
SELECT
    student_id,
    completion_rate
FROM student_assignment_summary
WHERE completion_rate < 70;

11. MVP Screens & UI Requirements

The application should contain 6–7 screens/views, not a large enterprise portal.

Screen 1 — Overview Dashboard

Show:

Total students

Average attendance

Assignment completion

Average exam score

Students requiring review

Risk distribution

Attendance trend

Performance trend

Students requiring review table

Screen 2 — Student Directory

Show:

Search

Risk filter

Course filter

Year filter

Student table

Screen 3 — Student Profile

Show:

Student information

Risk level

Risk drivers

Attendance chart

Assignment chart

Exam performance chart

Recent academic activity

Screen 4 — Risk Details

Show:

Risk level

Evidence

Current values

Previous/baseline values

Trend

Explanation

Screen 5 — Follow-Up / Intervention

Simple form:

Type

Date

Status

Notes

Screen 6 — Course Analytics

Show:

Course enrollment

Attendance

Assignment completion

Exam average

Students requiring review

Screen 7 — About / Methodology

Explain:

Dataset

Metrics

Risk rules

Project assumptions

Limitations

12. 20-Day Implementation Plan

Align implementation with the Sprint 1 structure shown in the course material.

Days 1–3 — Understand the Dataset

Define problem

Inspect columns

Understand relationships

Identify missing data

Define project assumptions

Deliverable: Dataset understanding + initial PRD.

Days 4–6 — Data Cleaning

Remove duplicates

Handle missing values

Normalize columns

Validate ranges

Create clean dataset

Deliverable: Clean dataset.

Days 7–9 — SQL & Exploratory Analysis

Create database tables

Write SQL queries

Calculate summary metrics

Explore attendance patterns

Explore assignment patterns

Explore exam patterns

Deliverable: Analysis notebook + SQL queries.

Days 10–12 — Feature & Risk Logic

Create student metrics

Create trends

Define risk thresholds

Validate sample outputs

Document assumptions

Deliverable: Risk-analysis module.

Days 13–16 — Interactive Application

Build:

Dashboard

Student directory

Student profile

Charts

Filters

Deliverable: Working MVP.

Days 17–18 — Intervention + Polish

Add:

Follow-up form

Methodology page

Empty states

Error handling

UI cleanup

Deliverable: Demo-ready application.

Days 19–20 — Testing & Demo

Test end-to-end flow

Validate numbers

Prepare presentation

Prepare viva questions

Record limitations

Perform final demo

Final deliverable:

Live demo of the data product + individual viva with mentor.

13. Testing, Validation & Success Criteria

Functional Testing

Verify:

Filters work.

Student search works.

Student profiles display correct data.

Charts match source calculations.

Risk levels match documented rules.

Intervention records save correctly.

Data Validation

For sample students, manually verify:

Raw records
     ↓
Calculated metric
     ↓
Displayed dashboard value

No dashboard metric should be trusted until it can be traced back to the dataset.

Success Criteria

The MVP is successful if a reviewer can:

Open the dashboard.

Understand the overall academic picture.

Identify a student requiring review.

Open that student's profile.

See the evidence behind the risk indicator.

Understand the attendance/assignment/exam trends.

Record a follow-up note.

Understand how the risk indicator was calculated.

14. Deployment & Project Structure

14.1 Simple Deployment

For a college project, use a lightweight deployment.

Possible options:

Streamlit Community Cloud

Render

Railway

Local demo on laptop

No Kubernetes or complex cloud infrastructure is required.

14.2 Suggested Repository

academic-risk-dashboard/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
│
├── src/
│   ├── data_cleaning.py
│   ├── features.py
│   ├── risk_engine.py
│   └── analytics.py
│
├── app/
│   ├── dashboard.py
│   ├── student_view.py
│   └── components.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── PRD.md

14.3 Git Workflow

main
 │
 ├── feature/data-cleaning
 ├── feature/analytics
 ├── feature/risk-engine
 └── feature/dashboard

Keep commits small and descriptive.

15. Limitations, Future Scope & End Note

15.1 Current Limitations

This project is an MVP academic data product.

Its risk indicator:

Depends on the quality of the provided dataset.

Uses project-defined thresholds.

Does not establish causation.

Should not be treated as a definitive prediction.

Has not necessarily been validated on a large historical population.

Does not account for every factor affecting academic outcomes.

15.2 Future Scope

If more time/data were available:

Validate the risk rules against historical outcomes.

Compare rule-based scoring with logistic regression/tree models.

Add model evaluation metrics.

Add more historical time-series features.

Add automated data ingestion.

Add student-facing insights.

Add intervention outcome analysis.

Add explainable ML.

Add role-based authentication.

15.3 Final Product Principle

The project should demonstrate a complete data-product lifecycle:

DATA
 ↓
CLEANING
 ↓
SQL
 ↓
EXPLORATION
 ↓
FEATURES
 ↓
RISK LOGIC
 ↓
INSIGHTS
 ↓
INTERACTIVE APPLICATION

The strongest part of the project is not the complexity of the model.

It is the ability to clearly demonstrate:

How raw academic data was transformed into a useful, explainable and interactive product.

Google Stitch Prompt — MVP Screens

Copy everything inside the following prompt into Google Stitch.

Create a polished MVP web application UI called:

"Academic Risk Dashboard"

PROJECT CONTEXT

This is a normal college/semester data-product project, NOT an enterprise university software system.

The product demonstrates an end-to-end data workflow:

Raw academic dataset
→ Data cleaning
→ SQL analysis
→ Student metrics
→ Simple explainable risk rules
→ Interactive dashboard

The application helps a faculty member or academic advisor review academic engagement patterns using:

- Attendance
- Assignment completion
- Assignment scores
- Exam performance
- Recent trends

IMPORTANT:

This is an analytics and educational-support dashboard.

Do NOT present the risk indicator as a definitive prediction.

Do NOT create a chatbot.

Do NOT create an overly complex enterprise admin portal.

Do NOT create dozens of screens.

The MVP should contain approximately 6–7 main views.

VISUAL STYLE

Create a clean modern academic analytics dashboard.

Style:

- Professional
- Minimal
- Modern SaaS dashboard
- Light background
- Clear typography
- Subtle borders
- Soft cards
- Moderate rounded corners
- Plenty of whitespace
- Strong information hierarchy
- Accessible contrast
- Simple charts
- Restrained semantic status colors

Avoid:

- Neon colors
- Futuristic AI visuals
- Excessive gradients
- 3D graphics
- Gaming UI
- Marketing landing-page aesthetics
- Excessive animations

PRIMARY USER

Faculty member / academic advisor.

--------------------------------------------------
GLOBAL NAVIGATION
--------------------------------------------------

Sidebar:

Academic Risk Dashboard

Overview
Students
Courses
Interventions
Methodology

Bottom:

About Project

Top bar:

Search
Current user
Profile icon

--------------------------------------------------
SCREEN 1 — OVERVIEW DASHBOARD
--------------------------------------------------

Page title:

"Academic Overview"

Subtitle:

"Review attendance, assignments and assessment patterns across students."

Top KPI cards:

Total Students
1,248

Average Attendance
82%

Assignment Completion
76%

Average Exam Score
68%

Students Requiring Review
86

Risk distribution card:

"Student Risk Distribution"

Categories:

Low
Moderate
Elevated

Use a simple bar chart.

Trend card:

"Academic Engagement Trend"

Line chart showing:

Attendance
Assignment completion
Exam performance

Time range:

Week 1
Week 2
Week 3
Week 4
Week 5
Week 6

Bottom section:

"Students Requiring Review"

Table columns:

Student
Program
Year
Risk
Attendance
Assignment Completion
Exam Average
Action

Example students:

Aarav Sharma
Computer Science
2
Elevated
71%
64%
58%

Priya Mehta
Information Technology
3
Moderate
78%
73%
65%

Rahul Verma
Mechanical Engineering
2
Elevated
69%
61%
56%

Clicking a row opens the Student Profile.

Filters above table:

Risk
Program
Year
Course

--------------------------------------------------
SCREEN 2 — STUDENT DIRECTORY
--------------------------------------------------

Title:

"Students"

Subtitle:

"Search and review academic engagement metrics."

Large search field:

"Search by student name or ID"

Filters:

Risk
Program
Year
Course

Table:

Student
Student ID
Program
Year
Risk
Attendance
Assignment Completion
Exam Average

Use pagination.

Add empty state:

"No students match your current filters."

--------------------------------------------------
SCREEN 3 — STUDENT PROFILE
--------------------------------------------------

Title:

"Student Profile"

Header:

Aarav Sharma

Student ID:
STU-1024

Program:
Computer Science

Year:
2

Right side:

Risk:
Elevated

Button:

"View Risk Details"

Button:

"Add Follow-Up"

Main section:

"Academic Summary"

Cards:

Attendance
71%

Assignment Completion
64%

Exam Average
58%

Recent Trend
↓ Declining

Section:

"Engagement Trend"

Create a clean line chart.

Show:

Attendance
Assignment completion
Exam performance

Section:

"Attendance"

Show a chart of attendance over recent weeks.

Section:

"Assignments"

Show:

Completed
Missing
Late
Average Score

Use a compact bar/line visualization.

Section:

"Exam Performance"

Show previous exams and scores.

Section:

"Recent Activity"

Timeline:

Assignment submitted
Attendance recorded
Exam completed
Assignment missing

--------------------------------------------------
SCREEN 4 — RISK DETAILS
--------------------------------------------------

Title:

"Risk Details"

Subtitle:

"Evidence supporting the current engagement indicator."

Top card:

Risk Level:
Elevated

Text:

"Recent engagement patterns show multiple areas that may warrant faculty review."

Section:

"Main Signals"

Create evidence cards.

Card 1:

Attendance

71%

Previous:
88%

Change:
-17 percentage points

Card 2:

Assignment Completion

64%

Previous:
91%

Change:
-27 percentage points

Card 3:

Exam Average

58%

Previous:
72%

Change:
-14 percentage points

Section:

"How the indicator was calculated"

Show:

Attendance concern
+
Assignment completion concern
+
Assessment trend concern
=
Elevated review indicator

Add a small note:

"Thresholds are project-defined rules based on the available dataset and are not universal academic standards."

Section:

"Important"

"Risk indicators support human review and should be considered alongside the student's current context."

--------------------------------------------------
SCREEN 5 — FOLLOW-UP
--------------------------------------------------

Use a right-side drawer or modal.

Title:

"Add Follow-Up"

Fields:

Student
Aarav Sharma

Follow-Up Type:

Academic check-in
Faculty discussion
Study support
Other

Date:

Date picker

Status:

Open
In Progress
Resolved

Notes:

Large text area.

Placeholder:

"Add a short note about the follow-up..."

Buttons:

Cancel
Save Follow-Up

After saving, show a success notification.

--------------------------------------------------
SCREEN 6 — COURSE ANALYTICS
--------------------------------------------------

Title:

"Course Analytics"

Subtitle:

"Compare aggregate engagement metrics across courses."

Filters:

Department
Course
Year

KPI cards:

Courses
Average Attendance
Assignment Completion
Average Exam Score

Chart:

"Attendance by Course"

Bar chart.

Chart:

"Assignment Completion by Course"

Bar chart.

Table:

Course
Students
Attendance
Assignment Completion
Exam Average
Students Requiring Review

Example:

Data Structures
86
79%
72%
65%
8

Database Systems
74
84%
81%
71%
4

Operating Systems
81
76%
69%
62%
9

--------------------------------------------------
SCREEN 7 — METHODOLOGY
--------------------------------------------------

Title:

"How This Dashboard Works"

Create a visual flow:

Academic Dataset
↓
Data Cleaning
↓
SQL Analysis
↓
Student Metrics
↓
Risk Rules
↓
Dashboard Insights

Sections:

"Data Used"

Attendance
Assignments
Exams
Student/Course information

"Risk Logic"

Explain that the MVP uses simple rule-based indicators based on:

Attendance
Assignment completion
Exam performance
Recent trends

"Important Limitations"

The indicator is not a definitive prediction.

The thresholds are project assumptions.

Missing or incomplete data can affect results.

The project demonstrates an academic analytics workflow rather than a production university decision system.

--------------------------------------------------
COMPONENTS
--------------------------------------------------

Create reusable UI components:

KPI Card
Risk Badge
Metric Card
Student Table
Filter Bar
Search Input
Line Chart Card
Bar Chart Card
Evidence Card
Timeline
Modal / Drawer
Empty State
Info Callout

--------------------------------------------------
RESPONSIVE DESIGN
--------------------------------------------------

Desktop first.

Target width:
1440px.

Also support:

1024px tablet
375px mobile

On mobile:

- Collapse sidebar.
- Convert tables to cards.
- Stack KPI cards.
- Keep charts readable.
- Keep actions accessible.

--------------------------------------------------
DATA STATES
--------------------------------------------------

Include:

Loading state
Empty state
No search results
Missing metric state
Error state

For missing data use:

"Insufficient data"

Never display missing values as zero.

--------------------------------------------------
UX PRINCIPLE
--------------------------------------------------

The user journey should be extremely clear:

Overview
→ Find Student
→ Open Student
→ Understand Evidence
→ Add Follow-Up

Prioritize evidence over the risk label.

The risk indicator should never dominate the entire interface.

--------------------------------------------------
FINAL DESIGN QUALITY
--------------------------------------------------

Make this look like a strong college data-product project that could be demonstrated during a mentor review or viva.

The UI should be:

- Cohesive
- Realistic
- Easy to implement
- Data-focused
- Explainable
- Professional
- Not over-engineered

Use realistic sample academic data throughout the screens.