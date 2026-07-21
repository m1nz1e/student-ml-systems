# Student ML Systems - Makefile
# Common commands for development, testing, and deployment

.PHONY: help install dev test lint type-check clean data-generate train evaluate mlflow-ui streamlit docker-build docker-run docs

# Default target
help:
	@echo "Student ML Systems - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install production dependencies"
	@echo "  make dev            - Install development dependencies"
	@echo "  make clean          - Remove build artifacts"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run flake8 linting"
	@echo "  make format         - Format code with black"
	@echo "  make type-check     - Run mypy type checking"
	@echo "  make test           - Run pytest with coverage"
	@echo ""
	@echo "Data:"
	@echo "  make data-generate  - Generate synthetic datasets"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Run training pipeline"
	@echo "  make evaluate       - Run evaluation pipeline"
	@echo "  make mlflow-ui      - Launch MLflow tracking UI"
	@echo ""
	@echo "UI:"
	@echo "  make streamlit      - Launch Streamlit dashboard"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build documentation with MkDocs"
	@echo ""

# Setup
install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf mlruns/
	@echo "✓ Cleaned build artifacts"

# Code Quality
lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black --line-length=100

type-check:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Data
data-generate:
	python -c "from src.data.synthetic import SITSSyntheticGenerator; g = SITSSyntheticGenerator(n_students=10000, n_courses=200); datasets = g.generate_all_datasets(); print('✓ Synthetic data generated')"

# Training
train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# UI
streamlit:
	streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

# Docker
docker-build:
	docker build -t student-ml-systems:latest .

docker-run:
	docker run -p 8501:8501 -p 5000:5000 student-ml-systems:latest

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve --dev-addr 0.0.0.0:8000

# Database setup (PostgreSQL with pgvector)
db-setup:
	@echo "Setting up PostgreSQL database..."
	@echo "1. Create database: CREATE DATABASE student_ml;"
	@echo "2. Enable pgvector: CREATE EXTENSION IF NOT EXISTS vector;"
	@echo "3. Run migrations: python scripts/migrate.py"

# MLflow tracking server
mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Hyperparameter tuning
tune-recommender:
	python scripts/tune_recommender.py

tune-enrollment:
	python scripts/tune_enrollment.py

tune-early-warning:
	python scripts/tune_early_warning.py

# Monitoring stack
monitoring-up:
	docker-compose -f monitoring/docker-compose.yml up -d

monitoring-down:
	docker-compose -f monitoring/docker-compose.yml down

# Full pipeline (data → train → evaluate → deploy)
pipeline: clean data-generate train evaluate
	@echo "✓ Full pipeline completed"
