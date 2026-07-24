# Student Retention Prediction — SPEC

## Problem Statement
Predict which students are at risk of leaving the university before completing their degree. Uses multi-year engagement + performance patterns to identify attrition risk from Year 1 onwards.

## Data Sources
- **Student records:** demographics, background, entry qualifications
- **Engagement data:** VLE logins, resource access, attendance
- **Assessment data:** marks, progression, GPA trends
- **Enrollments:** module choices, pass/fail patterns
- **Degree outcomes:** final classification (link to graduate outcomes)

## Target Variables
1. **Retention Risk** (binary): Will they stay or leave
2. **Risk Score** (regression): 0-100 likelihood of dropping out
3. **Risk Category** (ordinal): Low / Medium / High / Critical
4. **Year of Departure** (regression): If leaving, which year

## ML Approach

### Multi-task Model
Single model predicting all targets:
- Binary classification for retention risk
- Regression for risk score
- Ordinal classification for risk category
- Regression for departure year

### Features
- **Demographics:** age, gender, ethnicity, socioeconomic background (IMD, POLAR)
- **Entry profile:** entry grades, qualification type, first-generation student
- **Engagement:** VLE activity trends, attendance rates, module participation
- **Academic:** GPA trajectory, pass/fail rate, module load
- **Temporal:** engagement slope (increasing/decreasing), performance trend

### Model
- XGBoost multi-output (classification + regression)
- Time-series features (rolling averages, trends)

## Files to Create

### `src/models/student_retention/`
1. `data_prep.py` — RetentionFeatureEngineer
2. `multi_task_model.py` — RetentionPredictor
3. `metrics.py` — Evaluation (ROC-AUC, precision-recall, calibration)
4. `fairness.py` — Fairness audit (retention gaps by group)

### Synthetic Data Addition
Update `src/data/synthetic.py`:
- `generate_retention_outcomes()` — Student retention/departure data

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| Retention Risk | ROC-AUC | >0.85 |
| Risk Score | R² | >0.70 |
| Risk Category | Accuracy | >80% |
| Departure Year | MAE | <0.5 years |

## Fairness Metrics
- Retention gap by gender/ethnicity/SES (target: <5%)
- Intervention parity across groups

## Integration
- Uses all existing systems (Early Warning, Enrollment, Degree Outcome)
- Timeline: Key at Year 1 transition, then ongoing monitoring
- Proactive intervention: outreach before module registration

## Priority
High — directly impacts student success, NSS, TEF metrics, and university reputation
