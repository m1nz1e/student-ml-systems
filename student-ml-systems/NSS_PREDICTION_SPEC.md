# NSS Prediction — SPEC

## Problem Statement
Predict student satisfaction scores before the NSS (National Student Survey) to:
- Identify at-risk students for early intervention
- Inform course-level improvements
- Support TEF (Teaching Excellence Framework) metrics

## Data Sources
- **NSS data:** 27 questions across 7 themes (Teaching, Assessment, Feedback, Support, Organisation, Learning Resources, Student Voice)
- **Student records:** demographics, engagement, attendance, degree outcome
- **Link to other systems:** uses engagement data from Early Warning + degree predictions

## Target Variables
1. **Overall Satisfaction** (binary): Satisfied vs Dissatisfied
2. **Theme Scores** (7 themes, 0-100 scale): Teaching, Assessment, Feedback, Support, Organisation, Learning Resources, Student Voice
3. **NPS** (Net Promoter Score): Likely to recommend (0-10)

## ML Approach

### Multi-task Learning
Single model predicting all targets simultaneously:
- Architecture: Shared encoder → task-specific heads
- Benefits: Regularization, captures cross-theme relationships

### Features (from other systems + new)
- **From Early Warning:** attendance rate, VLE engagement, risk score
- **From Degree Outcome:** predicted degree class, GPA
- **From Engagement:** login frequency, resource access, forum activity
- **New:** module evaluations submitted, feedback response rate, contact hours

### Model Options
1. **Multi-output XGBoost** (baseline)
2. **Multi-task Neural Network** (PyTorch)
3. **Ordinal regression** for theme scores (0-100)

## Files to Create

### `src/models/nss_prediction/`
1. `data_prep.py` — NSSFeatureEngineer
2. `multi_task_model.py` — Multi-task regressor/classifier
3. `metrics.py` — Evaluation (MAE, R², ROC-AUC for binary)
4. `fairness.py` — Fairness audit (satisfaction gaps by group)

### Synthetic Data Addition
Update `src/data/synthetic.py`:
- `generate_nss_outcomes()` — NSS scores by theme

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| Overall Satisfaction | ROC-AUC | >0.80 |
| Theme Scores | R² | >0.70 |
| Theme Scores | MAE | <10 |
| NPS | Correlation | >0.60 |

## Fairness Metrics
- Satisfaction gap by gender/ethnicity (target: <5%)
- Response rate parity (target: >70% response rate)

## Integration
- Uses Early Warning engagement features
- Links to Degree Outcome predictions
- Timeline: Pre-NSS (April) → NSS collection (April-May) → Results (August)

## Priority
Medium-High — directly impacts TEF, career services, course improvement decisions
