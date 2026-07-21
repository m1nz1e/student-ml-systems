# Phase 2.1: Baseline Models — COMPLETE

**Date:** 2026-07-21
**Status:** ✅ Complete
**Files Created:** 3

---

## Overview

Implemented three baseline recommender models for the Course Recommender System. These baselines establish minimum performance thresholds that all advanced models must exceed.

---

## Models Implemented

### 1. RandomRecommender

**Approach:** Uniform random recommendations

**Use Case:**
- Lower bound baseline (any model should beat this)
- A/B testing control group
- Debugging evaluation pipeline

**Characteristics:**
- No training required (stateless)
- Different recommendations per user (random per user)
- Scores: Uniform [0, 1]

**Expected Performance:**
- NDCG@10: ~0.15-0.25 (random)
- Precision@10: ~0.05-0.15 (random)

**Code:**
```python
from src.models.recommender import RandomRecommender

rec = RandomRecommender(seed=42)
rec.fit(courses_df)
recommendations = rec.predict(student_features, n_recommendations=10)
```

---

### 2. PopularityRecommender

**Approach:** Recommend most popular courses (same for all users)

**Use Case:**
- Simple but strong baseline
- Cold-start solution (no user data needed)
- Production fallback

**Popularity Metrics:**
- `enrollment_count` (default) — Most enrolled courses
- `employment_rate` — Best graduate outcomes
- `satisfaction` — Highest student satisfaction

**Characteristics:**
- Requires course popularity data
- Same recommendations for all users
- Scores: Actual popularity values

**Expected Performance:**
- NDCG@10: ~0.30-0.45 (better than random)
- Precision@10: ~0.15-0.25

**Code:**
```python
from src.models.recommender import PopularityRecommender

rec = PopularityRecommender(popularity_metric="enrollment_count")
rec.fit(courses_df, enrollments_df)
recommendations = rec.predict(student_features, n_recommendations=10)
```

---

### 3. ContentBasedRecommender

**Approach:** Cosine similarity between student and course features

**Use Case:**
- Personalized recommendations
- Interpretable (feature-level matching)
- Cold-start for new courses (content-based)

**Feature Matching:**
- **Qualification type** (A-Level, BTEC, IB, etc.)
- **UCAS tariff** (grades vs entry requirements)
- **Contextual indicators** (IMD, POLAR, care leaver)
- **Department alignment** (Engineering, CS, Business, etc.)
- **Outcomes** (employment rate, satisfaction)
- **Course characteristics** (sandwich year, assessment type)

**Weights (default):**
- Qualification match: 30%
- Grade match: 30%
- Subject alignment: 20%
- Outcomes: 20%

**Characteristics:**
- Requires feature engineering (FeatureStore)
- Personalized per user
- Scores: Cosine similarity [0, 1]
- Explainable (feature contributions)

**Expected Performance:**
- NDCG@10: ~0.45-0.60 (reasonable personalization)
- Precision@10: ~0.25-0.35

**Code:**
```python
from src.models.recommender import ContentBasedRecommender

rec = ContentBasedRecommender(
    qualification_weight=0.3,
    grade_weight=0.3,
    subject_weight=0.2,
    outcome_weight=0.2,
)
rec.fit(student_features, course_features)
recommendations = rec.predict(student_features, n_recommendations=10)

# Get explanation for a recommendation
explanation = rec.explain_recommendation(
    student_id="SPR12345678",
    course_id="CRS00123",
)
```

**Example Explanation:**
```json
{
  "student_id": "SPR12345678",
  "course_id": "CRS00123",
  "overall_similarity": 0.87,
  "feature_contributions": [
    {
      "feature": "ucas_tariff",
      "student_value": 0.81,
      "course_value": 0.75,
      "similarity": 0.94
    },
    {
      "feature": "qual_A-Level",
      "student_value": 1.0,
      "course_value": 1.0,
      "similarity": 1.0
    },
    {
      "feature": "employment_rate",
      "student_value": 0.0,
      "course_value": 0.92,
      "similarity": 0.08
    }
  ]
}
```

---

## Evaluation Framework

### Metrics

**Primary:**
- **NDCG@10** (Normalized Discounted Cumulative Gain) — Ranking quality
- **Precision@10** — Relevance of top 10 recommendations

**Secondary (available):**
- MAP@10 (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Recall@10
- Diversity (intra-list distance)
- Coverage (catalog coverage)

### Ground Truth

Ground truth derived from historical enrollments:
- Student actually enrolled in course → positive label
- Student did not enroll → negative label

**Limitation:** Only observes one course per student (the one they chose), not all acceptable alternatives.

### Evaluation Function

```python
from src.models.recommender import evaluate_baselines

results = evaluate_baselines(
    student_features=student_features_df,
    course_features=course_features_df,
    enrollments_df=enrollments_df,
    n_recommendations=10,
)
```

### Expected Results

| Model | NDCG@10 | Precision@10 | Notes |
|-------|---------|--------------|-------|
| Random | 0.15-0.25 | 0.05-0.15 | Lower bound |
| Popularity | 0.30-0.45 | 0.15-0.25 | Strong baseline |
| Content-Based | 0.45-0.60 | 0.25-0.35 | Personalized |
| **LightFM (next)** | **0.60-0.75** | **0.35-0.45** | Target |
| **Hybrid Ensemble (final)** | **0.75+** | **0.45+** | Goal |

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/recommender/baselines.py` | 650 | Three baseline models + evaluation |
| `src/models/recommender/__init__.py` | 10 | Module exports |
| `src/models/__init__.py` | 5 | Parent module exports |

**Total:** 665 lines of production code

---

## Key Features

### RandomRecommender
- ✅ Stateless (no training)
- ✅ Reproducible (seed parameter)
- ✅ Fast (O(1) per user)

### PopularityRecommender
- ✅ Multiple popularity metrics
- ✅ Handles missing enrollment data
- ✅ Fast (pre-computed rankings)

### ContentBasedRecommender
- ✅ Feature-level explainability
- ✅ Configurable weights
- ✅ Cosine similarity matching
- ✅ Explanation generation (`explain_recommendation()`)
- ✅ Handles cold-start courses

### Evaluation
- ✅ NDCG@K implementation
- ✅ Precision@K implementation
- ✅ Automatic ground truth construction
- ✅ Comparative reporting

---

## Next Steps (Phase 2.2)

### Collaborative Filtering

**File:** `src/models/recommender/collaborative.py`

**Models to Implement:**
1. **Matrix Factorization (SVD)** — Classic CF
2. **LightFM** — Hybrid CF (user-item + features)
3. **Neural Collaborative Filtering** — Deep learning approach

**Evaluation Targets:**
- Beat Content-Based baseline by 15-20%
- NDCG@10 > 0.60
- Handle cold-start (new users, new courses)

**Hyperparameter Tuning:**
- Latent factors (32-256)
- Learning rate (0.001-0.1)
- Regularization (1e-6 to 1e-3)
- Loss function (warp, bpr, logistic)

---

## Usage Example (Full Pipeline)

```python
from src.data.synthetic import SITSSyntheticGenerator
from src.data.feature_store import FeatureStore
from src.models.recommender import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    evaluate_baselines,
)

# 1. Generate synthetic data
generator = SITSSyntheticGenerator(n_students=1000, n_courses=100, seed=42)
datasets = generator.generate_all_datasets()

# 2. Engineer features
feature_store = FeatureStore()
student_features = feature_store.engineer_student_features(
    datasets["students"],
    datasets["qualifications"],
)
course_features = feature_store.engineer_course_features(datasets["courses"])

# 3. Evaluate all baselines
results = evaluate_baselines(
    student_features=student_features,
    course_features=course_features,
    enrollments_df=datasets["enrollments"],
    n_recommendations=10,
)

# 4. Review results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  NDCG@10: {metrics['ndcg_at_10']:.4f}")
    print(f"  Precision@10: {metrics['precision_at_10']:.4f}")

# 5. Use best baseline for predictions
best_model = ContentBasedRecommender()
best_model.fit(student_features, course_features)
recommendations = best_model.predict(student_features, n_recommendations=10)

# 6. Get explanation
explanation = best_model.explain_recommendation(
    student_id="SPR12345678",
    course_id="CRS00001",
)
print(explanation)
```

---

## Lessons Learned

1. **Popularity is a strong baseline** — Don't underestimate simple approaches
2. **Content-based needs good features** — Garbage in, garbage out
3. **Explainability matters** — Users want to know WHY a course is recommended
4. **Cold-start is real** — Need hybrid approach for new users/courses
5. **Evaluation is tricky** — Only observe one choice per student (selection bias)

---

**Phase 2.1 Status:** ✅ COMPLETE
**Next:** Phase 2.2 — Collaborative Filtering (LightFM)
