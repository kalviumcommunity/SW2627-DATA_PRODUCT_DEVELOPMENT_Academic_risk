# Academic Engagement Risk Dashboard

An explainable academic analytics dashboard designed for faculty members and academic advisors to identify students whose recent academic engagement patterns may warrant supportive review.

---

## 1. Project Overview

Academic institutions collect attendance, assignment submissions, and assessment results, but these signals are often siloed across different systems. The **Academic Engagement Risk Dashboard** integrates these data sources into:

- **Clean & Validated Datasets**: Standardized schemas for students, courses, enrollments, attendance, assignments, submissions, and exams.
- **Student & Course Metrics**: Aggregated attendance percentage, submission completion, score trends, and grade distributions.
- **Explainable Risk Indicators**: Transparent, rule-based indicators (`Low`, `Moderate`, `Elevated`) that highlight actionable underlying factors without deterministic claims or black-box predictions.
- **Faculty-Friendly Insights**: Clean interactive views with drill-down student profiles, activity timelines, and follow-up/intervention logs.

---

## 2. Tech Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Data Manipulation**: Pandas, NumPy
- **Database & Querying**: SQLite, SQL
- **Visualization**: Plotly
- **Application Interface**: Streamlit
- **Testing**: pytest
- **Version Control**: Git & GitHub

---

## 3. Repository Structure

```text
academic-risk-dashboard/
│
├── data/
│   ├── raw/             # Unprocessed, immutable source academic data
│   └── processed/       # Validated, cleaned, and normalized datasets
│
├── notebooks/           # Exploratory data analysis and experimental notebooks
│
├── sql/                 # SQL schemas, DDL scripts, and analytical queries
│
├── src/                 # Reusable business logic, cleaning, and risk engine modules
│   └── __init__.py
│
├── app/                 # Streamlit UI presentation and interactive components
│   └── __init__.py
│
├── tests/               # Automated unit, integration, and validation tests
│   ├── __init__.py
│   └── test_environment.py
│
├── docs/                # Project documentation, PRD, and design specifications
│   └── PRD.md
│
├── .gitignore           # Git ignore rules for Python, Streamlit, and OS files
├── pyproject.toml       # Project metadata, build configuration, and tool settings
├── requirements.txt     # Locked production and development dependencies
└── README.md            # Project documentation and setup guide
```

---

## 4. Setup and Installation

### Prerequisites

Ensure Python 3.10 or higher is installed on your system.

```bash
python --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/kalviumcommunity/SW2627-DATA_PRODUCT_DEVELOPMENT_Academic_risk.git
cd SW2627-DATA_PRODUCT_DEVELOPMENT_Academic_risk
```

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running Tests

Run the test suite to verify the environment configuration and directory structure:

```bash
pytest -v
```

Or using the standard library `unittest` runner:

```bash
python -m unittest discover tests
```

---

## 6. Architecture & Core Flow

```text
Raw Academic Dataset
        ↓
  Data Cleaning
        ↓
   Validation
        ↓
Normalized Tables (SQLite)
        ↓
    SQL Analysis
        ↓
  Student Metrics
        ↓
Feature Engineering
        ↓
Explainable Risk Rules
        ↓
Interactive Dashboard (Streamlit)
        ↓
Follow-Up / Intervention Logging
```

---

## 7. Key Principles & Assumptions

1. **Academic Support, Not Prediction**: The risk indicator is a supportive review tool. It does not predict future failure with certainty.
2. **Explainability**: Every surfaced student must clearly display the underlying evidence (e.g. attendance decline, missing assignments, lower exam scores).
3. **No Imputation as Zero**: Missing data is never treated as zero performance. Missing or insufficient records are explicitly flagged.
4. **Decoupled Architecture**: All data processing and risk logic reside in `src/`, separated from the `app/` presentation layer.
