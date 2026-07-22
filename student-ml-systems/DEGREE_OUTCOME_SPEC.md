# Degree Outcome Prediction — Phase 1 Specification

## Objective
Build a production-grade ML system to predict student degree classification (First, 2:1, 2:2, Third, Fail) using assessment history, engagement patterns, and demographics.

## Files to Create
Location: `/home/m1nz/.openclaw/workspace/student-ml-systems/src/models/degree_outcome/`

### Required Files
1. `data_prep.py` — Feature engineering from assessments + engagement
2. `ordinal_classifier.py` — Ordinal regression model
3. `feature_importance.py` — SHAP/explainability
4. `fairness.py` — Bias auditing across protected characteristics

### Synthetic Data Extension
Location: `src/data/synthetic.py`
- Add `generate_degree_outcomes()` function
- Output: `final_classification` column in enrollments

## Features to Engineer

### Temporal Features (from assessments)
- Cumulative GPA over time (by year, semester)
- Mark trend (improving/declining)
- Best/worst subject performance
- Credit-weighted average
- Resit patterns

### Engagement Features
- Attendance trajectory
- VLE engagement trend
- Library usage (if available)
- Academic advisor meetings

### Static Features
- Entry qualifications (tariff points)
- Demographics (for fairness)
- Course difficulty (cohort average)

## Target Variable
```
final_classification:
  - First (70-100%)
  - 2:1 (60-69%)
  - 2:2 (50-59%)
  - Third (40-49%)
  - Fail (<40%)
```

## ML Approach

### Primary: Ordinal Logistic Regression
- Cumulative Link Model (proportional odds assumption)
- sklearn-compatible implementation
- Probability calibration

### Alternative: Threshold-Based
- Train binary classifier for each threshold
- Pass/Fail → 2:2/Third → 2:1/First
- Combine with logical constraints

### Evaluation Metrics
- **Mean Absolute Error** (ordinal distance)
- **Exact Match Accuracy**
- **Within-One Accuracy** (e.g., predicting 2:1 when actual is 2:2)
- **ROC-AUC** per class (One-vs-Rest)
- **Calibration plots**

## Fairness Requirements
- Demographic parity check across Gender, Ethnicity, IMD
- Equalized odds for False Negative rates
- disparate impact ratio ≥ 0.8

## Integration Points
- Can use existing `AssessmentResult` data
- Reuse `EngagementFeatures` from Early Warning
- Add to existing Streamlit dashboard

## Success Criteria
- [ ] Synthetic data generates realistic degree outcomes
- [ ] Features engineered correctly (temporal aggregation)
- [ ] Ordinal model trained with proper CV
- [ ] Fairness audit passes thresholds
- [ ] SHAP explanations generated
- [ ] Integration test passes

## Delegation Tasks

### Task 1: Data Prep + Synthetic Data (Data Engineer)
1. Add `generate_degree_outcomes()` to `src/data/synthetic.py`
2. Create `src/models/degree_outcome/data_prep.py`
3. Engineer temporal features (cumulative GPA, trends)
4. Handle missing data appropriately

### Task 2: Model + Evaluation (ML Engineer)
1. Create `src/models/degree_outcome/ordinal_classifier.py`
2. Implement ordinal logistic regression
3. Add calibration
4. Implement evaluation metrics (MAE, within-one, calibration)
5. Create fairness auditor
6. Add SHAP explanations

### Task 3: Integration (ML Engineer)
1. Create `src/models/degree_outcome/__init__.py`
2. Add tests
3. Verify imports work

## Constraints
- Follow existing code style (type hints, docstrings)
- Reuse existing patterns from enrollment/yield system
- Keep features interpretable
- No data leakage (fit on train only)
