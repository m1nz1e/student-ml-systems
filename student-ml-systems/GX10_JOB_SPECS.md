# GX10 Implementation Jobs — Student ML Systems

**Delegated:** 2026-07-21
**Target:** ASUS GX10 (Edge Inference Server)
**Runtime:** Python 3.10+, PyTorch, scikit-survival

---

## Job 1: LSTM Early Warning Model

**File:** `src/models/early_warning/lstm.py`

**Purpose:** Implement LSTM model for temporal pattern recognition in student engagement data.

**Requirements:**
- PyTorch LSTM with configurable architecture
- Support for bidirectional LSTM
- Dropout regularization
- Early stopping
- Model checkpointing
- SHAP integration for explainability

**Architecture:**
```python
Input: (batch_size=32, seq_len=12, features=14)
  ↓
LSTM(14 → 64, return_sequences=True, dropout=0.3)
  ↓
LSTM(64 → 32, return_sequences=False, dropout=0.3)
  ↓
Dense(32 → 16, ReLU, dropout=0.3)
  ↓
Dense(16 → 1, sigmoid)
  ↓
Output: dropout/failure probability
```

**Key Methods:**
```python
class LSTMEarlyWarning(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, n_layers=2, 
                 dropout=0.3, bidirectional=False):
        # Initialize layers
        pass

    def forward(self, x):
        # Forward pass
        pass

    def fit(self, X_train, y_train, X_val, y_val, 
            n_epochs=50, early_stopping_patience=10):
        # Training loop with validation
        pass

    def predict(self, X):
        # Inference
        pass

    def get_shap_values(self, X_sample):
        # SHAP explainability
        pass
```

**Dependencies:**
```
torch>=2.0.0
shap>=0.43.0
```

**Test Command:**
```bash
cd /home/m1nz/.openclaw/workspace/student-ml-systems
python -c "from src.models.early_warning import LSTMEarlyWarning; print('✓ LSTM module loads')"
```

---

## Job 2: Survival Analysis Models

**File:** `src/models/early_warning/survival.py`

**Purpose:** Implement survival analysis for time-to-event prediction.

**Requirements:**
- Cox Proportional Hazards model
- Random Survival Forests
- Survival function prediction
- C-index evaluation
- Feature importance extraction

**Key Methods:**
```python
class SurvivalAnalyzer:
    def __init__(self, model_type='cox', n_estimators=100):
        # 'cox' or 'rsf'
        pass

    def fit(self, X, y_surv):
        # y_surv: structured array (event, time)
        pass

    def predict_risk(self, X):
        # Risk scores
        pass

    def predict_survival_function(self, X):
        # S(t) for each sample
        pass

    def get_c_index(self, X, y_surv):
        # Concordance index
        pass

    def get_feature_importance(self, top_n=20):
        # For RSF only
        pass
```

**Dependencies:**
```
scikit-survival>=0.20.0
pandas>=2.0.0
numpy>=1.24.0
```

**Test Command:**
```bash
python -c "from src.models.early_warning import SurvivalAnalyzer; print('✓ Survival module loads')"
```

---

## Job 3: Risk Scorer

**File:** `src/models/early_warning/risk_scorer.py`

**Purpose:** Convert model outputs to interpretable 0-100 risk scores with explanations.

**Requirements:**
- Risk score normalization (0-100)
- 5-tier stratification (Low/Medium/High/Very High/Critical)
- SHAP-based explanations
- Lead time calculation
- Alert generation

**Risk Thresholds:**
```python
{
    'LOW': (0, 20],      # Green
    'MEDIUM': (20, 40],  # Yellow
    'HIGH': (40, 60],    # Orange
    'VERY_HIGH': (60, 80], # Red
    'CRITICAL': (80, 100] # Purple
}
```

**Key Methods:**
```python
class RiskScorer:
    def __init__(self, model, thresholds=None):
        pass

    def predict_risk(self, X):
        # Returns 0-100 scores
        pass

    def stratify(self, risk_scores):
        # Returns categories
        pass

    def explain(self, X, student_idx):
        # SHAP explanations + recommended actions
        pass

    def calculate_lead_time(self, predictions, actual_events, event_times):
        # Average weeks before event detected
        pass

    def generate_alert(self, student_id, risk_score, explanation):
        # Alert dictionary for tutors
        pass
```

**Dependencies:**
```
shap>=0.43.0
numpy>=1.24.0
```

**Test Command:**
```bash
python -c "from src.models.early_warning import RiskScorer; print('✓ RiskScorer module loads')"
```

---

## Job 4: Streamlit Tutor Dashboard

**File:** `ui/pages/early_warning.py`

**Purpose:** Tutor-facing dashboard for monitoring at-risk students.

**Requirements:**
- Student list with risk scores (sortable, filterable)
- Individual student risk timeline
- Intervention logging
- Export functionality
- Fairness monitoring dashboard

**Pages:**
1. **Dashboard** — Overview, at-risk student list
2. **Student Detail** — Risk timeline, explanations
3. **Interventions** — Log actions, track effectiveness
4. **Fairness** — Monitor bias across groups

**Key Components:**
```python
# Student risk table
st.dataframe(
    risk_df[['student_id', 'risk_score', 'category', 'trend']],
    use_container_width=True,
)

# Risk timeline
st.line_chart(student_risk_history)

# Intervention form
with st.form("intervention"):
    action = st.selectbox("Action", options)
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Log Intervention")

# Fairness metrics
st.metric("Demographic Parity Diff", dp_diff)
st.metric("Equalized Odds Diff", eo_diff)
```

**Dependencies:**
```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
```

**Test Command:**
```bash
streamlit run ui/pages/early_warning.py --server.port 8502
```

---

## Job 5: MLflow Integration

**File:** `src/mlops/mlflow_config.py`

**Purpose:** Experiment tracking and model registry.

**Requirements:**
- Local MLflow server setup
- Auto-logging for PyTorch/sklearn
- Model versioning
- Model comparison UI
- Artifact storage

**Key Methods:**
```python
class MLflowTracker:
    def __init__(self, tracking_uri='sqlite:///mlflow.db'):
        pass

    def start_experiment(self, experiment_name):
        pass

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_model(self, model, artifact_path='model'):
        pass

    def register_model(self, model_name, model_uri):
        pass
```

**Setup Commands:**
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Access UI
# Open http://localhost:5000
```

**Dependencies:**
```
mlflow>=2.8.0
psycopg2-binary>=2.9.9  # For PostgreSQL backend
```

---

## Job 6: Docker Deployment

**File:** `Dockerfile`

**Purpose:** Containerize the entire application.

**Requirements:**
- Multi-stage build (build + runtime)
- All dependencies installed
- Expose ports (8501 for Streamlit, 5000 for MLflow)
- Volume mounts for data/models

**Dockerfile Template:**
```dockerfile
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ src/
COPY ui/ ui/
COPY configs/ configs/

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501 5000

CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/student_ml
    depends_on:
      - db

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

**Test Commands:**
```bash
docker build -t student-ml-systems:latest .
docker-compose up -d
docker ps  # Verify containers running
```

---

## Job 7: GitHub Actions CI/CD

**File:** `.github/workflows/ci.yml`

**Purpose:** Automated testing and deployment.

**Requirements:**
- Run on push/PR
- Install dependencies
- Run tests with coverage
- Code quality checks (black, flake8, mypy)
- Build Docker image
- Deploy to Render (optional)

**Workflow Template:**
```yaml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --cov=src
      - name: Code quality
        run: |
          black src/ --check
          flake8 src/
          mypy src/ --ignore-missing-imports

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker
        run: docker build -t student-ml-systems:latest .
```

---

## Execution Instructions for GX10

**Step 1: Clone Repository**
```bash
cd /home/m1nz/.openclaw/workspace
git clone https://github.com/m1nz1e/student-ml-systems.git
cd student-ml-systems
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
pip install scikit-survival  # Extra for survival analysis
```

**Step 3: Implement Jobs**
- Start with Job 1 (LSTM) — most critical
- Then Job 2 (Survival)
- Then Job 3 (Risk Scorer)
- Then Job 4 (Streamlit UI)
- Then Job 5-7 (MLOps)

**Step 4: Test Each Job**
```bash
# After each job, run test command from job spec
python -c "from src.models.early_warning import X; print('✓ Job N complete')"
```

**Step 5: Commit and Push**
```bash
git add .
git commit -m "Implement Job N: [component name]"
git push origin main
```

**Step 6: Report Status**
Update this file with completion status:
```markdown
## Completion Status

- [x] Job 1: LSTM — [x] Code [x] Tests [ ] Docs
- [x] Job 2: Survival — [x] Code [x] Tests [ ] Docs
- [x] Job 3: Risk Scorer — [x] Code [x] Tests [ ] Docs
- [x] Job 4: Streamlit — [x] Code [x] Tests [ ] Docs
- [x] Job 5: MLflow — [x] Code [x] Tests [ ] Docs
- [x] Job 6: Docker — [x] Code [x] Tests [ ] Docs
- [x] Job 7: CI/CD — [x] Code [x] Tests [ ] Docs
```

---

## Priority Order

1. **Job 1 (LSTM)** — Core model for Phase 4
2. **Job 3 (Risk Scorer)** — Needed for production use
3. **Job 4 (Streamlit)** — User-facing interface
4. **Job 2 (Survival)** — Alternative model (nice-to-have)
5. **Job 5 (MLflow)** — Experiment tracking
6. **Job 6 (Docker)** — Deployment
7. **Job 7 (CI/CD)** — Automation

---

**Estimated Time:** 4-6 hours total (GX10 with vLLM/qwen3.6-35b)
**GPU Usage:** Minimal (coding task, not inference)
**Output:** Production-ready Phase 4 + Phase 5 implementation

---

**Delegated by:** Brahma (OpenClaw main session)
**Date:** 2026-07-21 22:31 UTC
**Contact:** m1nz via OpenClaw
