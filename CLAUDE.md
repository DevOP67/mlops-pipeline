# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

**Windows (full pipeline):**
```
run.bat
```
This creates a venv, installs dependencies, starts the MLflow tracking server (port 5000), runs the training pipeline, then launches the FastAPI app (port 8000).

**Manual steps:**
```bash
pip install -r requirements.txt

# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Run training pipeline
python -m src.pipeline.train_pipeline

# Start API server
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Run tests:**
```bash
pytest tests/
```

## Architecture

The project implements an end-to-end MLOps pipeline for Iris classification with three layers:

### 1. Training Pipeline (`src/`)
Orchestrated by `src/pipeline/train_pipeline.py`, which reads `configs/config.yaml`, sets up an MLflow run, then calls each component in sequence:
- `src/components/preprocessing.py` — loads Iris dataset, splits train/test
- `src/components/model_trainer.py` — builds a scikit-learn `Pipeline` (StandardScaler → RandomForestClassifier)
- `src/components/evaluation.py` — evaluates accuracy, logs params/metrics to MLflow, and **registers the model as `IrisClassifier` in the MLflow registry**

### 2. MLflow Tracking
- Tracking URI: `sqlite:///mlflow.db` (local SQLite)
- Artifacts stored in `mlruns/`
- Model registered under name `"IrisClassifier"` with alias `production`
- Dashboard: http://127.0.0.1:5000

### 3. Serving Layer (`backend/`)
- `backend/app.py` — FastAPI app with a single `/predict` route
- `backend/routes/predict.py` — POST `/predict` accepts `IrisInput` (4 floats), returns predicted class as int
- `backend/services/model_service.py` — lazy-loads the model from MLflow registry (`models:/IrisClassifier@production`) with retry logic (5 attempts); the model must be registered before the API can serve predictions

The API depends on the MLflow server being up and the training pipeline having run at least once to register the model.

## Key Config

`configs/config.yaml` controls experiment name, model hyperparameters (n_estimators, max_depth, random_state), and train/test split ratio. Changes here automatically propagate to MLflow logged parameters.

## Known Issue

The GitHub Actions workflow (`.github/workflows/ml_pipeline.yml`) runs `python -m src.train`, but the correct module path is `python -m src.pipeline.train_pipeline`.
