# Git & Team Collaboration Workflow

This document outlines the professional version control and collaboration standards for our 3-member engineering team developing the **Academic Engagement Risk Dashboard**. 

Given the curriculum requirement of implementing **50 discrete concepts**, strict adherence to these branching, commit, pull request, and merge protocols ensures zero merge regressions, clear audit trails, and seamless parallel development.

---

## 1. Branch Architecture

Our repository follows a structured branching model adapted for multi-contributor academic data products:

```text
  main (Protected Production)
   │
   ├── develop (Integration / Team Staging)
   │    │
   │    ├── feature/01-environment-setup
   │    ├── feature/02-github-workflow
   │    ├── feature/03-data-ingestion-schema
   │    └── ... [Concepts 04 - 50]
```

### 1.1 `main` Branch
- **Role**: Stable, production-ready release branch representing fully integrated and verified concepts.
- **Protection Rules**:
  - Direct pushes and force pushes are strictly prohibited.
  - Changes must arrive exclusively via reviewed and approved Pull Requests.
  - All automated tests and validations must pass before merging.

### 1.2 `develop` (Integration Branch)
- **Role**: Shared staging branch where active concept features are integrated and tested together prior to formal milestone releases.
- **Protection Rules**:
  - Concept branches target `develop` (or `main` when working in direct-to-main trunk workflows as specified per sprint).
  - Must remain buildable and green at all times.

### 1.3 Concept Feature Branches (`feature/*`)
- **Role**: Dedicated branch created for each specific concept milestone.
- **Lifespan**: Short-lived (1–2 days max per concept).
- **Isolation**: Exactly one concept per branch. Developers must never bundle multiple concepts into a single branch.

---

## 2. Branch Naming Conventions

All branches must follow a standardized naming pattern with lowercase letters, numbers, and hyphens:

| Branch Type | Format Pattern | Example |
| :--- | :--- | :--- |
| **Concept Feature** | `feature/<2-digit-concept-id>-<short-description>` | `feature/01-environment-setup` |
| **Bug Fix** | `bugfix/<concept-id>-<issue-description>` | `bugfix/05-null-handling-fix` |
| **Documentation** | `docs/<concept-id>-<topic>` | `docs/02-team-workflow-update` |
| **Testing** | `test/<concept-id>-<test-suite>` | `test/12-risk-engine-cases` |

> [!IMPORTANT]
> **50 Concepts Rule**: Branch names for concepts must match the exact concept designation assigned by the project roadmap (e.g. `feature/02-github-workflow`). Never create ambiguous branch names like `test-branch`, `ishita-work`, or `feature-new`.

---

## 3. Commit Message Standards (Conventional Commits)

We enforce the [Conventional Commits](https://www.conventionalcommits.org/) specification to keep the Git history readable and automatable.

### Format
```text
<type>(<optional-scope>): <concise-imperative-description>

[optional body explaining context, rationale, and tradeoffs]

[optional footer referencing issue or PR, e.g., Closes #12]
```

### Allowed Commit Types
- `feat`: A new feature, module, or concept implementation (e.g., `feat: implement student attendance metric calculator`).
- `fix`: A bug fix or correction to existing logic (e.g., `fix: correct exam score percentage scaling`).
- `docs`: Documentation changes only (e.g., `docs: establish github team workflow`).
- `test`: Adding missing tests or correcting existing tests (e.g., `test: add edge cases for unrecorded attendance`).
- `refactor`: Code changes that neither fix a bug nor add a feature (e.g., `refactor: decouple risk rules from pandas pipeline`).
- `style`: Formatting, missing semicolons, whitespace (no code logic changes).
- `chore`: Maintenance tasks, dependency updates, tooling configuration (e.g., `chore: update requirements.txt`).

### Commit Rules
1. **Imperative Mood**: Write in the present imperative ("add feature" not "added feature" or "adds feature").
2. **No Capital First Letter**: Keep subject lowercase unless beginning with a proper noun.
3. **No Trailing Period**: Do not end the subject line with a period.
4. **Atomic Commits**: Each commit should represent one logical unit of work.

---

## 4. 3-Member Team Collaboration & Ownership

To maximize throughput across the 50 concepts while preventing conflicts:

| Team Role | Primary Responsibilities | Secondary / Review Focus |
| :--- | :--- | :--- |
| **Member A (Data Engineer)** | Raw data pipelines, cleaning scripts, database schemas, SQL queries | Pipeline performance, schema normalization |
| **Member B (Analytics / Risk)** | Metric calculation, feature engineering, explainable risk rules | Risk explainability, edge-case validation |
| **Member C (UI / Frontend)** | Streamlit views, Plotly charts, student drill-downs, intervention UI | Advisor usability, presentation consistency |

### Team Interaction Rules
1. **No Overwriting**: Never modify or overwrite a teammate's active branch without prior alignment.
2. **Missing Dependencies Protocol (Rule 24)**: If Concept $N$ depends on an uncompleted module from Concept $N-1$, document and report the blocker immediately. Do not build mock or placeholder modules that will break integration.
3. **Daily Base Synchronization**: Before starting any new concept branch, update your local base branch:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/<concept-id>-<description>
   ```

---

## 5. Pull Request (PR) Workflow

Every concept must go through a formal GitHub Pull Request:

### Step 1: Pre-PR Checklist
Before opening a PR, ensure:
- [ ] Code is formatted and adheres to project style guidelines.
- [ ] No temporary files, `.venv`, or local `.sqlite` databases are committed.
- [ ] Automated tests pass locally (`pytest -v`).
- [ ] Business logic is kept in `src/`, separate from UI code in `app/`.
- [ ] Risk indicators remain explainable and do NOT claim definitive predictive certainty.

### Step 2: Open PR
- **Base Branch**: `main` (or `develop`).
- **Title**: Follow the commit format (e.g., `feat: setup academic risk dashboard environment`).
- **Template**: Fill out the structured PR template (`.github/PULL_REQUEST_TEMPLATE.md`).

### Step 3: Peer Review & Approval
- At least **1 peer review approval** from a team member is required before merging.
- Reviewers verify:
  1. Real implementation (no stubbed placeholder code).
  2. Test coverage for the concept.
  3. Strict adherence to project assumptions (e.g., missing data is not zero).

---

## 6. Merge Workflow & Conflict Resolution

### 6.1 Merge Strategy
- **Preferred Option**: **Squash and Merge** (or Rebase and Merge) on GitHub.
  - Produces a clean, linear commit history on `main` where each of the 50 concepts represents a single atomic commit.
  - Automatically ties the PR number to the commit message (e.g., `feat: setup academic risk dashboard environment (#1)`).

### 6.2 Conflict Resolution Protocol
If conflicts arise with the base branch:
1. Fetch latest upstream changes:
   ```bash
   git fetch origin
   ```
2. Rebase or merge base into your feature branch locally:
   ```bash
   git checkout feature/<concept-id>-<description>
   git merge origin/main
   ```
3. Resolve conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) carefully.
4. Run test suite to verify no regressions:
   ```bash
   pytest -v
   ```
5. Commit the resolution and push to the feature branch.
6. **NEVER force-push (`git push --force`)** to shared branches or without explicit team consent.

---

## 7. 50-Concept Roadmap Management

To successfully implement all 50 concepts:
1. **Sequential Branching**: Feature branches should branch off the latest merged `main`.
2. **Branch Cleanup**: Once a concept PR is merged into `main`, delete the remote feature branch on GitHub to prevent clutter.
3. **Continuous Verification**: After every merged PR, all team members pull the updated `main`:
   ```bash
   git checkout main
   git pull origin main
   ```
