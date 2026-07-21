# Implementation Order — Methodical Development Plan

**Project:** Student ML Systems
**Approach:** Sequential, ordinal, production-grade
**Status:** In Progress

---

## Phase 0: Foundation ✅ (COMPLETED)

### Completed Tasks
- [x] Project README with full documentation
- [x] requirements.txt (all dependencies)
- [x] pyproject.toml (project config, black, isort, mypy, pytest)
- [x] Makefile (common commands)
- [x] .gitignore
- [x] LICENSE (MIT)
- [x] Directory structure created

### Core Modules Implemented
- [x] `src/data/synthetic.py` — SITS-style synthetic data generator
- [x] `src/evaluation/cross_validation.py` — 3 CV strategies (Stratified, TimeSeries, Grouped)
- [x] `src/evaluation/ranking_metrics.py` — NDCG, MAP, MRR, Precision, Recall, Diversity
- [x] `src/tuning/optuna_tuner.py` — Bayesian optimization with pruning
- [x] All `__init__.py` files for proper imports

---

## Phase 1: Data Pipeline (NEXT)

### 1.1 Feature Store Foundation
**File:** `src/data/feature_store.py`

**Tasks:**
1. Define feature groups (student, course, engagement, performance)
2. Implement feature transformations (normalization, encoding, aggregations)
3. Create feature registry (metadata, descriptions, data types)
4. Build train/test split utilities (temporal, stratified, grouped)

**Dependencies:** synthetic.py

---

### 1.2 Data Validation
**File:** `src/data/validation.py`

**Tasks:**
1. Define data quality rules (Great Expectations style)
2. Implement schema validation (types, ranges, nulls)
3. Create data drift detection (PSI, KL divergence)
4. Build validation reports (HTML/JSON output)

**Dependencies:** feature_store.py

---

### 1.3 SITS Extractor (Real Integration)
**File:** `src/data/sits_extractor.py`

**Tasks:**
1. Implement ODBC connection to SITS
2. Map SITS fields to platform schema
3. Create incremental extraction (CDC)
4. Handle data anonymization (GDPR compliance)

**Dependencies:** validation.py
**Note:** Optional — synthetic data works for portfolio

---

## Phase 2: Course Recommender System

### 2.1 Baseline Models
**File:** `src/models/recommender/baselines.py`

**Tasks:**
1. Random baseline (uniform random recommendations)
2. Popularity baseline (most enrolled courses)
3. Content-based v1 (simple cosine similarity)
4. Evaluate baselines (NDCG@10, Precision@5)

**Dependencies:** feature_store.py, ranking_metrics.py

---

### 2.2 Collaborative Filtering
**File:** `src/models/recommender/collaborative.py`

**Tasks:**
1. Matrix factorization (SVD)
2. LightFM implementation (hybrid CF)
3. Neural collaborative filtering (PyTorch)
4. Hyperparameter tuning with Optuna

**Dependencies:** baselines.py, optuna_tuner.py

---

### 2.3 Content-Based Filtering
**File:** `src/models/recommender/content_based.py`

**Tasks:**
1. TF-IDF vectorization (course descriptions, prerequisites)
2. Cosine similarity matching
3. Semantic similarity (embeddings with pgvector)
4. Feature engineering (qualification match, subject overlap)

**Dependencies:** collaborative.py

---

### 2.4 Hybrid Ensemble
**File:** `src/models/recommender/ensemble.py`

**Tasks:**
1. Weighted combination (CF + content-based)
2. Learning-to-rank (LambdaMART/XGBoost)
3. Diversity injection (MMR — Maximal Marginal Relevance)
4. Cold-start handling (content-based fallback)

**Dependencies:** collaborative.py, content_based.py

---

### 2.5 Explainability
**File:** `src/models/recommender/explainability.py`

**Tasks:**
1. SHAP analysis (global feature importance)
2. LIME (local explanations per recommendation)
3. Counterfactuals (Alibi — "what-if" scenarios)
4. User-facing explanation templates

**Dependencies:** ensemble.py, shap, lime, alibi

---

### 2.6 Course Recommender UI
**File:** `ui/pages/recommender.py`

**Tasks:**
1. Input form (qualifications, grades, interests, career goals)
2. Results display (top 10 courses with match scores)
3. Explainability view (SHAP values, counterfactuals)
4. Comparison tool (side-by-side course comparison)
5. Export functionality (PDF, email)

**Dependencies:** ensemble.py, explainability.py, streamlit, plotly

---

## Phase 3: Enrollment Yield Prediction

### 3.1 Data Preparation
**File:** `src/models/enrollment/data_prep.py`

**Tasks:**
1. Feature engineering (applicant, course, engagement features)
2. Handle class imbalance (SMOTE, class weights)
3. Temporal train/test split
4. Baseline model (logistic regression)

**Dependencies:** feature_store.py, cross_validation.py

---

### 3.2 XGBoost Classifier
**File:** `src/models/enrollment/classifier.py`

**Tasks:**
1. XGBoost implementation
2. Probability calibration (Platt scaling, isotonic)
3. Hyperparameter tuning (Optuna)
4. Evaluation (ROC-AUC, PR-AUC, calibration curves)

**Dependencies:** data_prep.py, optuna_tuner.py

---

### 3.3 Fairness Audit
**File:** `src/models/enrollment/fairness.py`

**Tasks:**
1. Demographic parity difference
2. Equalized odds difference
3. Disparate impact ratio (4/5ths rule)
4. Calibration by protected group
5. Bias mitigation (reweighting, adversarial debiasing)

**Dependencies:** classifier.py, fairness_metrics.py

---

### 3.4 Enrollment Yield UI
**File:** `ui/pages/enrollment.py`

**Tasks:**
1. Admin dashboard (yield predictions by course)
2. Applicant-level viewer (individual predictions)
3. What-if analysis (change features, see impact)
4. Export reports (CSV, PDF)

**Dependencies:** classifier.py, fairness.py, streamlit

---

## Phase 4: Early Warning System

### 4.1 Sequential Data Preparation
**File:** `src/models/early_warning/data_prep.py`

**Tasks:**
1. Create time-series features (weekly aggregates)
2. Handle missing data (forward fill, interpolation)
3. Student-level grouped splits (no leakage)
4. Sequence creation (sliding windows)

**Dependencies:** feature_store.py, cross_validation.py

---

### 4.2 LSTM Model
**File:** `src/models/early_warning/lstm.py`

**Tasks:**
1. PyTorch LSTM implementation
2. Attention mechanism (temporal attention)
3. Multi-task learning (dropout + failure prediction)
4. Hyperparameter tuning (hidden_size, layers, dropout)

**Dependencies:** data_prep.py, optuna_tuner.py, pytorch

---

### 4.3 Survival Analysis
**File:** `src/models/early_warning/survival.py`

**Tasks:**
1. Cox Proportional Hazards model
2. Random Survival Forests
3. Time-to-event prediction
4. Concordance index evaluation

**Dependencies:** lstm.py, scikit-survival

---

### 4.4 Risk Scoring & Interventions
**File:** `src/models/early_warning/risk_scorer.py`

**Tasks:**
1. Risk score calculation (0-100 scale)
2. Risk stratification (low/medium/high/critical)
3. Intervention recommendations (actionable steps)
4. Lead time analysis (weeks before event)

**Dependencies:** lstm.py, survival.py

---

### 4.5 Early Warning UI
**File:** `ui/pages/early_warning.py`

**Tasks:**
1. Tutor dashboard (at-risk student list)
2. Student timeline (risk trajectory over time)
3. Intervention tracker (log actions, measure effectiveness)
4. Fairness monitoring (FPR/FNR by group)

**Dependencies:** risk_scorer.py, streamlit, plotly

---

## Phase 5: MLOps & Infrastructure

### 5.1 MLflow Integration
**File:** `experiments/tracking/mlflow_config.py`

**Tasks:**
1. Set up local MLflow server
2. Experiment tracking (params, metrics, artifacts)
3. Model registry (versioning, stages)
4. Model comparison UI

**Dependencies:** All models

---

### 5.2 Model Serving
**File:** `src/api/main.py`

**Tasks:**
1. FastAPI endpoints (REST API)
2. Request/response schemas (Pydantic)
3. Model loading (MLflow integration)
4. Batch prediction endpoint

**Dependencies:** All models, mlflow, fastapi

---

### 5.3 Monitoring Stack
**Files:**
- `monitoring/prometheus/prometheus.yml`
- `monitoring/grafana/dashboards/*.json`

**Tasks:**
1. Prometheus metrics (latency, error rate, predictions)
2. Grafana dashboards (performance, health, fairness)
3. Alert rules (Alertmanager config)
4. Logging (ELK stack integration)

**Dependencies:** api/main.py, prometheus, grafana

---

### 5.4 Docker Deployment
**Files:**
- `Dockerfile`
- `docker-compose.yml`

**Tasks:**
1. Multi-stage Dockerfile (build + runtime)
2. Docker Compose (app + db + monitoring)
3. Environment configuration (.env templates)
4. Deployment scripts (local, cloud)

**Dependencies:** All components

---

### 5.5 CI/CD Pipeline
**File:** `.github/workflows/ci.yml`

**Tasks:**
1. GitHub Actions workflow
2. Automated testing (pytest)
3. Code quality (black, isort, flake8, mypy)
4. Docker build & push
5. Deployment (Render/AWS)

**Dependencies:** All components

---

## Phase 6: Documentation & Portfolio

### 6.1 API Documentation
**File:** `docs/api.md`

**Tasks:**
1. OpenAPI/Swagger spec
2. Endpoint descriptions
3. Request/response examples
4. Authentication guide

---

### 6.2 Architecture Documentation
**File:** `docs/architecture.md`

**Tasks:**
1. System architecture diagram
2. Component descriptions
3. Data flow diagrams
4. Deployment architecture

---

### 6.3 Compliance Documentation
**Files:**
- `docs/compliance/dpia.md`
- `docs/compliance/ethics.md`

**Tasks:**
1. Data Protection Impact Assessment
2. Ethics review application
3. GDPR compliance checklist
4. Fairness audit reports

---

### 6.4 Portfolio Write-up
**File:** `PORTFOLIO.md`

**Tasks:**
1. Project overview (problem, solution, impact)
2. Technical deep-dive (models, architecture)
3. Results & metrics (evaluation reports)
4. Lessons learned & future work

---

## Current Status

**Completed:** Phase 0 (Foundation) ✅
**Next:** Phase 1.1 (Feature Store)

**Files Created:** 10
**Lines of Code:** ~2,500
**Test Coverage:** 0% (pending tests)

---

## Estimated Timeline

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 1 (Data) | 1-2 weeks | None |
| Phase 2 (Recommender) | 2-3 weeks | Phase 1 |
| Phase 3 (Enrollment) | 1-2 weeks | Phase 1 |
| Phase 4 (Early Warning) | 3-4 weeks | Phase 1 |
| Phase 5 (MLOps) | 2-3 weeks | Phases 2-4 |
| Phase 6 (Docs) | 1 week | All phases |

**Total:** 10-15 weeks (solo developer, part-time)

---

**Last Updated:** 2026-07-21
**Next Milestone:** Feature Store implementation
