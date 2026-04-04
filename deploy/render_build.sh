#!/usr/bin/env bash
# deploy/render_build.sh
# Build-step runner for Render deployment

set -euo pipefail

echo "=============================================="
echo "  🚀 Starting ML Pipeline Build"
echo "=============================================="

# 1. Install dependencies
echo "📦 [1/3] Installing project dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run the Machine Learning Training Pipeline
# This bakes the trained model (model.pkl) into the deployment
echo "🚂 [2/3] Running training pipeline..."
export MLFLOW_TRACKING_URI=file:./mlruns
export PYTHONPATH=.
python -m src.pipeline.train_pipeline

# 3. Final verification
echo "✅ [3/3] Build complete. Artifacts ready."
ls -lh artifacts/models/
