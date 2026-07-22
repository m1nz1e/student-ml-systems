# Graduate Outcomes Prediction — SPEC

## Problem Statement
Predict graduate employment outcomes (employment, further study, salary) to:
- Identify students needing career support early
- Inform course design based on outcomes
- Support TEF (Teaching Excellence Framework) metrics

## Data Sources
- **HESA DLHE** (Destinations of Leavers from Higher Education) — official graduate data
- **Student records:** degree classification, course, demographics, engagement
- **Link to Degree Outcome:** uses final year predictions as features

## Target Variables (Multi-output)
1. **Employment Status:** Employed, Further Study, Both, Unemployed (4 classes)
2. **Salary Band:** <£20K, £20-30K, £30-40K, £40K+ (4 bands)
3. **Study Destination:** UK, EU, International (3 classes)

## ML Approach

### Multi-task Learning
Single model predicting all 3 targets simultaneously (shared representation)
- Architecture: Shared encoder → 3 task heads
- Benefits: Regularization, better generalization

### Features (from Degree Outcome + new)
- Degree predicted class
- Course (subject area, level)
- Demographics (gender, ethnicity, background)
- Engagement (attendance, VLE, library)
- Socioeconomic (IMD, POLAR, first-gen)
- Career preparation (optional: CV uploads, career events attended)

### Model Options
1. **Multi-output XGBoost** (baseline)
2. **Multi-task Neural Network** (PyTorch)
3. **Ordinal regression** for salary bands

## Files to Create

### `src/models/graduate_outcomes/`
1. `data_prep.py` — GraduateOutcomeFeatureEngineer
2. `multi_task_model.py` — Multi-task classifier
3. `metrics.py` — Evaluation (per-task + composite)
4. `fairness.py` — Fairness audit (employment disparities)

### Synthetic Data Addition
Update `src/data/synthetic.py`:
- `generate_graduate_outcomes()` — Employment, salary, further study

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| Employment | ROC-AUC | >0.80 |
| Salary Band | QWK | >0.65 |
| Study Dest | ROC-AUC | >0.75 |
| Composite | Avg ROC-AUC | >0.75 |

## Fairness Metrics
- Employment rate by gender/ethnicity (target: <5% disparity)
- Salary gap by background (target: <10% gap)
- Access to career services equity

## Integration
- Uses Degree Outcome predictions as features
- Links to enrollment data (which students → which outcomes)
- Timeline: Final year → 6 months post-graduation (DLHE)

## Priority
Medium — useful for TEF, career service planning, course improvement
