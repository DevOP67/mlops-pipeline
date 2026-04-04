# ==============================================================================
#  MLOps Pipeline — Developer Makefile
#  Usage:  make <target>
#  On Windows with Git Bash or WSL, use:  bash -c "make <target>"
# ==============================================================================

.PHONY: help setup train serve test lint format clean logs status

PYTHON   := venv/Scripts/python   # Windows path; Linux: venv/bin/python
PIP      := venv/Scripts/pip
UVICORN  := venv/Scripts/uvicorn
MLFLOW   := venv/Scripts/mlflow
PYTEST   := venv/Scripts/pytest
PYTHONPATH := .

help:  ## 📋 Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────────────────────

setup: ## 🐍 Create virtualenv and install all dependencies
	python -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov httpx flake8 black isort
	@echo "✅ Environment ready. Run 'make serve' to start the API."

# ── Pipeline ──────────────────────────────────────────────────────────────────

mlflow-server: ## 📊 Start MLflow tracking UI on http://localhost:5000
	$(MLFLOW) server \
	  --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns \
	  --host 127.0.0.1 \
	  --port 5000

train: ## 🤖 Run the ML training pipeline
	PYTHONPATH=$(PYTHONPATH) MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
	  $(PYTHON) -m src.pipeline.train_pipeline

train-local: ## 🤖 Run training with file-based MLflow (no server needed)
	PYTHONPATH=$(PYTHONPATH) MLFLOW_TRACKING_URI=file:./mlruns \
	  $(PYTHON) -m src.pipeline.train_pipeline

# ── API ───────────────────────────────────────────────────────────────────────

serve: ## 🌐 Start FastAPI dev server on http://localhost:8000
	PYTHONPATH=$(PYTHONPATH) \
	  $(UVICORN) backend.app:app \
	  --host 127.0.0.1 --port 8000 --reload

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## ✅ Run all tests with coverage
	PYTHONPATH=$(PYTHONPATH) MLFLOW_TRACKING_URI=file:./mlruns \
	  $(PYTEST) tests/ --cov=src --cov=backend \
	  --cov-report=term-missing -v

test-unit: ## ✅ Run only unit tests (fast)
	PYTHONPATH=$(PYTHONPATH) MLFLOW_TRACKING_URI=file:./mlruns \
	  $(PYTEST) tests/test_pipeline.py -v -m "not integration"

test-api: ## ✅ Run only API tests
	PYTHONPATH=$(PYTHONPATH) MLFLOW_TRACKING_URI=file:./mlruns \
	  $(PYTEST) tests/test_api.py -v

# ── Code Quality ──────────────────────────────────────────────────────────────

lint: ## 🔍 Run flake8 linter
	$(PYTHON) -m flake8 src/ backend/ tests/ \
	  --max-line-length=120 --exclude=venv,__pycache__

format: ## 🎨 Auto-format code with black + isort
	$(PYTHON) -m black . --exclude "venv/"
	$(PYTHON) -m isort . --skip venv

format-check: ## 🎨 Check formatting without modifying files
	$(PYTHON) -m black --check . --exclude "venv/"
	$(PYTHON) -m isort --check-only . --skip venv

# ── Utilities ─────────────────────────────────────────────────────────────────

logs: ## 📜 Tail the latest deployment log
	@tail -f logs/deploy.log 2>/dev/null || echo "No log file found yet."

clean: ## 🧹 Remove pycache, coverage files, and temp artifacts
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true
	rm -f coverage.xml .coverage
	@echo "✅ Cleaned."

status: ## 📡 Check if API and MLflow are running (Linux/Mac only)
	@curl -sf http://127.0.0.1:8000/ && echo "✅ API is UP" || echo "❌ API is DOWN"
	@curl -sf http://127.0.0.1:5000/ && echo "✅ MLflow is UP" || echo "❌ MLflow is DOWN"
