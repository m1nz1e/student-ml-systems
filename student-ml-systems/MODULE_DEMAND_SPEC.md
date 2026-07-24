# Module Demand Forecasting — SPEC

## Problem Statement
Predict module enrollment demand to:
- Identify oversubscribed modules for cap management
- Predict under-subscribed modules at risk of cancellation
- Optimize module scheduling and resource allocation
- Inform curriculum planning

## Data Sources
- **Historical enrollment data:** module choices by year, student demographics
- **Module data:** department, credits, prerequisites, capacity
- **Student data:** year of study, course, demographics, prior performance
- **Link to Course Recommender:** which modules students take together

## Target Variables
1. **Enrollment Count** (regression): Number of students enrolling
2. **Demand Category** (classification): Low / Medium / High / Oversubscribed
3. **Fill Rate** (regression): Percentage of capacity filled

## ML Approach

### Time-Series + Regression
- Historical enrollment patterns by module
- Student-level prediction of module choice (collaborative filtering)
- Aggregate to module-level demand

### Features
- Module characteristics (department, credits, difficulty)
- Historical enrollment trends (3-year rolling average)
- Student demographics (which students prefer which modules)
- Prerequisites completion rate
- Timetable constraints

### Model Options
1. **Gradient Boosting Regressor** (XGBoost for enrollment count)
2. **Classification** for demand category
3. **Ensemble** of both

## Files to Create

### `src/models/module_demand/`
1. `data_prep.py` — ModuleDemandFeatureEngineer
2. `regressor.py` — DemandForecaster
3. `classifier.py` — DemandCategoryClassifier
4. `metrics.py` — Evaluation (MAE, RMSE, accuracy)

### Synthetic Data Addition
Update `src/data/synthetic.py`:
- `generate_module_enrollments()` — Historical enrollment data
- `generate_module_demand()` — Demand predictions per module per year

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| Enrollment Count | MAE | <15 students |
| Enrollment Count | RMSE | <20 students |
| Fill Rate | R² | >0.75 |
| Demand Category | Accuracy | >80% |

## Integration
- Links to Course Recommender (which modules students choose)
- Links to Enrollment Yield (which students accept offers)
- Timeline: Pre-registration (March-April) → Module selection (May-June)

## Priority
Medium — operational efficiency, resource planning, curriculum design
