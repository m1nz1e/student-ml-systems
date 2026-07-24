# Student Wellbeing Score — SPEC

## Problem Statement
Predict student wellbeing indicators before they become critical issues. Uses engagement patterns, attendance, and academic behaviour as proxies for mental health and wellbeing.

## Data Sources
- **Student records:** demographics, background factors
- **Engagement data:** VLE activity patterns, logins, forum activity
- **Attendance:** lecture/seminar attendance rates, lateness
- **Academic:** workload, grade stress, progression
- **Help-seeking:** library access, counselling service use, pastoral meetings

## Target Variables
1. **Wellbeing Score** (regression): 0-100 composite score
2. **Risk Level** (ordinal): Critical / High / Medium / Low
3. **At Risk** (binary): Wellbeing below threshold?
4. **Support Need** (ordinal): 0-3 level of intervention needed

## ML Approach

### Multi-task Model
Single model predicting all targets:
- Regression for wellbeing score
- Ordinal classification for risk level
- Binary classification for at-risk flag
- Ordinal regression for support need

### Features
- **Engagement patterns:** login frequency, time of day, session duration
- **Attendance:** lecture attendance, tutorial participation, lateness
- **Academic pressure:** workload per semester, grade anxiety indicators
- **Behavioural:** forum activity, library visits, social engagement
- **Background:** known risk factors, access to support services

### Model
- XGBoost multi-output (regression + classification)
- Temporal features (declining engagement over time = risk signal)

## Files to Create

### `src/models/wellbeing/`
1. `data_prep.py` — WellbeingFeatureEngineer
2. `model.py` — WellbeingPredictor
3. `metrics.py` — Evaluation (MAE, ROC-AUC, QWK)
4. `fairness.py` — Fairness audit (wellbeing gaps by group)

### Synthetic Data Addition
Update `src/data/synthetic.py`:
- `generate_wellbeing_outcomes()` — Wellbeing scores per student

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| Wellbeing Score | MAE | <10 |
| Wellbeing Score | R² | >0.70 |
| At Risk | ROC-AUC | >0.85 |
| Risk Level | QWK | >0.70 |

## Fairness Metrics
- Wellbeing gap by demographics (target: <5%)
- Support access parity across groups
- No predictive bias for protected characteristics

## Integration
- Uses Early Warning engagement features
- Links to Student Retention (shared risk factors)
- Links to NSS (student satisfaction proxies)
- Timeline: Ongoing monitoring, weekly updates

## Priority
High — student welfare, duty of care, university liability, TEF student outcomes
