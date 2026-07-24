# Exam Performance Prediction — SPEC

## Problem Statement
Predict individual exam marks before exams occur to:
- Identify students at risk of failing
- Target revision interventions
- Flag students needing extra support
- Inform module-level grade distributions

## Data Sources
- **Student records:** demographics, entry grades, background
- **Engagement data:** VLE activity, attendance, tutorial participation
- **Assessment data:** prior coursework marks, assignments, progress tests
- **Module data:** exam weight, difficulty, pass rates

## Target Variables
1. **Exam Mark** (regression): Predicted exam score 0-100
2. **Pass/Fail** (binary): Will they pass (≥40)?
3. **Grade Class** (ordinal): Fail / Pass / Merit / Distinction
4. **Confidence** (regression): How confident is the prediction?

## ML Approach

### Multi-task Model
Single model predicting all targets:
- Regression for exam mark
- Binary classification for pass/fail
- Ordinal classification for grade
- Uncertainty estimation via ensemble variance

### Features
- **Historical:** prior module marks, coursework average, trend
- **Engagement:** VLE activity, attendance, forum participation
- **Module:** exam weight, average pass rate, difficulty
- **Student:** entry grades, demographics, attendance history

### Model
- XGBoost multi-output with regression + classification heads
- Ensemble for uncertainty (multiple model variants)

## Files to Create

### `src/models/exam_performance/`
1. `data_prep.py` — ExamFeatureEngineer
2. `model.py` — ExamPredictor
3. `metrics.py` — Evaluation (MAE, RMSE, ROC-AUC, calibration)
4. `fairness.py` — Fairness audit (grade gaps by group)

### Synthetic Data Addition
Update `src/data/synthetic.py`:
- `generate_exam_outcomes()` — Exam predictions per student/module

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| Exam Mark | MAE | <8 marks |
| Exam Mark | R² | >0.75 |
| Pass/Fail | ROC-AUC | >0.85 |
| Grade Class | QWK | >0.70 |
| Calibration | Brier Score | <0.10 |

## Fairness Metrics
- Grade gap by gender/ethnicity/SES (target: <5%)
- Pass rate parity across groups

## Integration
- Links to Degree Outcome (final classification)
- Links to Early Warning (engagement signals)
- Links to Module Demand (module difficulty)
- Timeline: Pre-exam (4-6 weeks) → predictions → interventions

## Priority
High — direct student impact, exam board preparation, intervention targeting
