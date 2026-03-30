@echo off
echo ==============================================
echo Automating MLOps Pipeline Startup (Native Windows)
echo ==============================================

:: 1. Verify Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating Python Virtual Environment...
    python -m venv venv
)

:: 2. Activate Environment
call venv\Scripts\activate.bat

:: 3. Setup Requirements
echo [INFO] Installing dependencies quietly...
pip install -r requirements.txt -q

:: 4. Start MLflow Tracking Backend
echo [INFO] Booting up MLflow Tracking Server...
start "MLflow Server" cmd /c "call venv\Scripts\activate.bat && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"

:: Wait gracefully for MLflow to spin up
timeout /t 3 /nobreak >nul

:: 5. Auto-run the pipeline
echo [INFO] Executing Training Pipeline...
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.pipeline.train_pipeline

:: 6. Launch the App!
echo [INFO] Starting FastAPI App at http://127.0.0.1:8000
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
