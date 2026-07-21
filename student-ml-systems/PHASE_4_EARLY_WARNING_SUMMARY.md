# Phase 4: Early Warning System — IMPLEMENTATION GUIDE

**Status:** 🟡 Partially Complete (Data Prep ✅, Models 📝 Documented)
**Date:** 2026-07-21

---

## Overview

The Early Warning System predicts students at risk of **dropout or failure 4+ weeks before the event**, enabling proactive intervention by tutors and support services.

**Key Features:**
- **Sequential modeling** (LSTM) for temporal patterns in engagement
- **Survival analysis** for time-to-event prediction
- **Risk scoring** (0-100 scale) with stratification
- **Explainable alerts** (which factors contribute to risk)
- **Fairness monitoring** (equal false positive/negative rates across groups)

---

## Files Created

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `src/models/early_warning/data_prep.py` | 520 | ✅ Complete | Sequential feature engineering |
| `src/models/early_warning/lstm.py` | 0 | 📝 Documented | LSTM model (code below) |
| `src/models/early_warning/survival.py` | 0 | 📝 Documented | Survival analysis (code below) |
| `src/models/early_warning/risk_scorer.py` | 0 | 📝 Documented | Risk scoring (code below) |
| `src/models/early_warning/__init__.py` | 10 | ✅ Complete | Module exports |

**Total:** 530 lines (data prep complete, models documented)

---

## Component 1: Sequential Feature Engineering ✅

**File:** `src/models/early_warning/data_prep.py`

**Purpose:** Create time-series sequences from weekly student data.

**Features Created:**

### Per-Week Features (14 dimensions)
**VLE Engagement (6 features):**
- Logins count
- Resources accessed
- Forum posts
- Quiz attempts
- Video views
- Total actions

**Attendance (3 features):**
- Attendance rate (0-1)
- Absent count
- Authorised absence count

**Assessments (5 features):**
- Average mark to date
- Number of assessments
- Submission rate
- Late submission rate
- Resit rate

### Static Features (15 dimensions)
- Demographics (gender, IMD, POLAR, care leaver, first-gen, disability)
- Prior attainment (UCAS tariff, qualification type)
- Enrollment characteristics

### Target Variables
- **Dropout:** Withdrew from program
- **Failure:** Not retained to year 2
- **Both:** Either dropout or failure

**Output Shapes:**
- `X_seq`: (n_students, sequence_length=12, n_features=14)
- `X_static`: (n_students, n_static_features=15)
- `y`: (n_students,) binary

**Usage:**
```python
from src.models.early_warning import EarlyWarningFeatureEngineer

engineer = EarlyWarningFeatureEngineer(
    sequence_length=12,
    prediction_horizon=4,
    target_type="both",
    min_weeks=4,
)

X_seq, X_static, y, metadata = engineer.create_sequences(
    vle_df=vle_data,
    attendance_df=att_data,
    assessments_df=ass_data,
    students_df=student_data,
    enrollments_df=enrollment_outcomes,
)

# Also create aggregated features for non-sequential models
df_agg, feature_names = engineer.create_aggregated_features(
    vle_df=vle_data,
    attendance_df=att_data,
    assessments_df=ass_data,
    students_df=student_data,
)
```

---

## Component 2: LSTM Model 📝

**Purpose:** Learn temporal patterns in student engagement that precede dropout/failure.

**Architecture:**
```
Input: (batch_size, sequence_length=12, n_features=14)
  ↓
LSTM Layer 1 (64 units, return_sequences=True)
  ↓
Dropout (0.3)
  ↓
LSTM Layer 2 (32 units, return_sequences=False)
  ↓
Dropout (0.3)
  ↓
Dense Layer (16 units, ReLU)
  ↓
Output Layer (1 unit, sigmoid)
  ↓
Probability of dropout/failure
```

**Implementation:**
```python
import torch
import torch.nn as nn

class LSTMEarlyWarning(nn.Module):
    def __init__(
        self,
        input_dim=14,
        hidden_dim=64,
        n_layers=2,
        dropout=0.3,
        bidirectional=False,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * (2 if bidirectional else 1), 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.fc(h_n[-1])
        return out

# Training
model = LSTMEarlyWarning(input_dim=14, hidden_dim=64, n_layers=2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train loop (standard PyTorch)
for epoch in range(n_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
```

**Hyperparameters:**
- `hidden_dim`: 64 or 128
- `n_layers`: 2 or 3
- `dropout`: 0.2-0.5
- `learning_rate`: 0.001 (Adam)
- `batch_size`: 32 or 64
- `n_epochs`: 50-100 (with early stopping)

**Expected Performance:**
- **ROC-AUC:** 0.85-0.92
- **Recall@20%:** >85% (catch at-risk students)
- **Lead Time:** 4-6 weeks before event

---

## Component 3: Survival Analysis 📝

**Purpose:** Predict time-to-event (when will student drop out), not just binary outcome.

**Models:**

### Cox Proportional Hazards
```python
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Prepare survival data
# y_surv: structured array with (event_occurred, time_to_event)
y_surv = np.array([
    (event, time) for event, time in zip(events, times)
], dtype=[('event', bool), ('time', float)])

# Train
cox = CoxPHSurvivalAnalysis()
cox.fit(X_agg, y_surv)

# Predict risk scores
risk_scores = cox.predict(X_agg)

# Predict survival function
survival_funcs = cox.predict_survival_function(X_agg)
```

### Random Survival Forests
```python
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42,
)

rsf.fit(X_agg, y_surv)

# Feature importance
importance = rsf.feature_importances_

# C-index (concordance)
c_index = rsf.score(X_test, y_surv_test)
# Expected: 0.70-0.80
```

**Metrics:**
- **C-index:** Concordance index (like ROC-AUC for survival)
- **Brier Score:** Time-dependent calibration
- **Integrated Brier Score:** Overall calibration

**Advantages over Binary Classification:**
- ✅ Predicts **when** event will occur
- ✅ Handles censored data (students who haven't dropped out yet)
- ✅ More informative for intervention timing

---

## Component 4: Risk Scorer 📝

**Purpose:** Convert model outputs to interpretable 0-100 risk scores with stratification.

**Risk Stratification:**
```
0-20:   Low Risk (green)      → No action needed
21-40:  Medium Risk (yellow)  → Monitor, light touch
41-60:  High Risk (orange)    → Tutor check-in
61-80:  Very High Risk (red)  → Intervention required
81-100: Critical Risk (purple) → Urgent support needed
```

**Implementation:**
```python
class RiskScorer:
    def __init__(
        self,
        model,
        calibration_data=None,
        thresholds=None,
    ):
        self.model = model
        self.thresholds = thresholds or {
            'low': 20,
            'medium': 40,
            'high': 60,
            'very_high': 80,
        }

        # Calibrate if provided
        if calibration_data is not None:
            self._calibrate(calibration_data)

    def predict_risk(self, X_seq=None, X_static=None):
        """Predict risk score (0-100)."""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_seq)[:, 1]
        else:
            proba = self.model.predict(X_seq)

        # Convert to 0-100 scale
        risk_scores = proba * 100

        return risk_scores

    def stratify(self, risk_scores):
        """Assign risk category."""
        categories = []
        for score in risk_scores:
            if score <= self.thresholds['low']:
                categories.append('LOW')
            elif score <= self.thresholds['medium']:
                categories.append('MEDIUM')
            elif score <= self.thresholds['high']:
                categories.append('HIGH')
            elif score <= self.thresholds['very_high']:
                categories.append('VERY_HIGH')
            else:
                categories.append('CRITICAL')
        return categories

    def explain_risk(self, X_seq, student_idx):
        """Explain why this student is at risk."""
        # Use SHAP for LSTM or feature importance for survival models
        import shap

        explainer = shap.DeepExplainer(self.model, X_seq[:100])
        shap_values = explainer.shap_values(X_seq[student_idx:student_idx+1])

        # Return top contributing features
        explanations = {
            'risk_score': self.predict_risk(X_seq[student_idx:student_idx+1])[0],
            'category': self.stratify([self.predict_risk(X_seq[student_idx:student_idx+1])[0]])[0],
            'top_factors': get_top_features(shap_values),
            'recommended_actions': get_actions(shap_values),
        }

        return explanations
```

**Lead Time Analysis:**
```python
def calculate_lead_time(predictions, actual_events, event_times):
    """Calculate average weeks before event detected."""
    lead_times = []

    for i, (pred, actual, event_time) in enumerate(zip(predictions, actual_events, event_times)):
        if actual == 1:  # Event occurred
            # Find first week where risk > threshold
            for week, risk in enumerate(pred):
                if risk > 50:  # High risk threshold
                    lead_time = event_time - week
                    lead_times.append(lead_time)
                    break

    return np.mean(lead_times) if lead_times else 0

# Expected: 4-6 weeks average lead time
```

---

## Complete Pipeline Example

```python
from src.models.early_warning import (
    EarlyWarningFeatureEngineer,
    LSTMEarlyWarning,
    SurvivalAnalyzer,
    RiskScorer,
)

# 1. Create sequences
engineer = EarlyWarningFeatureEngineer(
    sequence_length=12,
    prediction_horizon=4,
    target_type="both",
)

X_seq, X_static, y, metadata = engineer.create_sequences(
    vle_df=vle_data,
    attendance_df=att_data,
    assessments_df=ass_data,
    students_df=student_data,
    enrollments_df=enrollments,
)

# 2. Train LSTM model
lstm = LSTMEarlyWarning(input_dim=14, hidden_dim=64, n_layers=2)
# Train with PyTorch (see above)

# 3. Train survival model
survival = SurvivalAnalyzer(model_type="cox")
survival.fit(X_agg, y_surv)

# 4. Create risk scorer
scorer = RiskScorer(model=lstm, calibration_data=(X_val, y_val))

# 5. Predict risk for current students
risk_scores = scorer.predict_risk(X_seq_current)
risk_categories = scorer.stratify(risk_scores)

# 6. Generate alerts for high-risk students
high_risk_mask = risk_scores > 60
high_risk_students = metadata['student_ids'][high_risk_mask]

for student_id in high_risk_students:
    explanation = scorer.explain_risk(X_seq_current, student_idx)
    send_alert_to_tutor(student_id, explanation)
```

---

## Expected Performance

| Metric | Target | Baseline | Excellent |
|--------|--------|----------|-----------|
| ROC-AUC | >0.85 | 0.75 | >0.90 |
| Recall@20% | >85% | 70% | >90% |
| Precision@20% | >60% | 40% | >70% |
| Lead Time | 4+ weeks | 2 weeks | 6+ weeks |
| C-index (survival) | >0.75 | 0.65 | >0.80 |
| Fairness Score | >0.75 | - | >0.85 |

---

## Ethical Considerations

### False Positives vs False Negatives
- **False Positive:** Student flagged as at-risk but succeeds
  - Cost: Unnecessary intervention, potential stigma
  - Mitigation: Human review, opt-out option

- **False Negative:** Student not flagged but drops out
  - Cost: Missed opportunity to help
  - Mitigation: Lower threshold, regular re-scoring

### Bias Mitigation
- **Regular fairness audits** (demographic parity, equalized odds)
- **Human oversight** (model supports, doesn't replace, human decisions)
- **Transparency** (students can see their risk factors)
- **Appeal process** (students can request human review)

### Data Privacy
- **Minimal data** (only what's necessary for prediction)
- **Anonymization** (remove PII before modeling)
- **Retention limits** (delete data after graduation + X years)
- **Consent** (inform students about system)

---

## Integration with Student Support

**Workflow:**
1. **Weekly scoring** (every Monday at 9am)
2. **Alert generation** (risk > 60)
3. **Tutor notification** (email/dashboard)
4. **Intervention logging** (tutor records actions)
5. **Re-scoring** (2 weeks later, measure improvement)
6. **Escalation** (if risk increases despite intervention)

**Intervention Types:**
- Academic support (tutoring, study skills)
- Financial support (hardship funds, advice)
- Mental health support (counseling referral)
- Social integration (peer mentoring, societies)

---

## Cumulative Project Status

| Phase | Status | Files | Total Lines |
|-------|--------|-------|-------------|
| Phase 0: Foundation | ✅ | 13 | ~2,500 |
| Phase 1: Data Pipeline | ✅ | 2 | ~1,350 |
| Phase 2: Course Recommender | ✅ | 7 | ~3,865 |
| Phase 3: Enrollment Yield | ✅ | 4 | ~1,465 |
| Phase 4: Early Warning | 🟡 | 2 | ~530 |
| Phase 5: MLOps | 🔴 | 0 | 0 |
| Phase 6: Documentation | 🔴 | 0 | 0 |

**Cumulative:** 28 files, **~9,710 lines** of production code

---

**Phase 4 Status:** 🟡 Data Prep Complete, Models Documented
**Next:** Phase 5 — MLOps (MLflow, Docker, Monitoring, CI/CD)
