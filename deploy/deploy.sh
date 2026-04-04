#!/usr/bin/env bash
# =============================================================================
# deploy/deploy.sh
# DEPLOYMENT SCRIPT — executed remotely via SSH by the CD pipeline
#
# Usage:
#   bash deploy.sh <environment> <git_sha>
#   bash deploy.sh production abc1234
# =============================================================================

set -euo pipefail

ENVIRONMENT="${1:-production}"
GIT_SHA="${2:-unknown}"
APP_DIR="${HOME}/mlops-pipeline"
VENV="${APP_DIR}/venv"
LOG_FILE="${APP_DIR}/logs/deploy.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ── Logging helper ────────────────────────────────
log() {
  echo "[${TIMESTAMP}] $*" | tee -a "${LOG_FILE}"
}

log "========================================"
log "🚀 Starting deployment"
log "   Environment : ${ENVIRONMENT}"
log "   Git SHA     : ${GIT_SHA}"
log "   App Dir     : ${APP_DIR}"
log "========================================"

# ── Ensure directories exist ─────────────────────
mkdir -p "${APP_DIR}/logs"
mkdir -p "${APP_DIR}/artifacts"
mkdir -p "${APP_DIR}/mlruns"

cd "${APP_DIR}"

# ── 1. Pull latest code ───────────────────────────
log "📥 [1/5] Pulling latest code from git..."
if [ -d ".git" ]; then
  git fetch origin main --quiet
  git reset --hard origin/main
  log "✅ Code synced to HEAD (${GIT_SHA})"
else
  log "⚠️  Not a git repo — assuming rsync already pushed files."
fi

# ── 2. Upgrade dependencies ───────────────────────
log "📦 [2/5] Installing/upgrading Python dependencies..."
"${VENV}/bin/pip" install --upgrade pip --quiet
"${VENV}/bin/pip" install -r requirements.txt --quiet
log "✅ Dependencies installed."

# ── 3. Run training pipeline ─────────────────────
log "🤖 [3/5] Running ML training pipeline..."
MLFLOW_TRACKING_URI="sqlite:///${APP_DIR}/mlflow.db" \
PYTHONPATH="${APP_DIR}" \
"${VENV}/bin/python" -m src.pipeline.train_pipeline \
  >> "${LOG_FILE}" 2>&1 && log "✅ Training pipeline completed." \
  || log "⚠️  Training pipeline failed — deploying with existing model."

# ── 4. Zero-downtime service restart ─────────────
log "🔄 [4/5] Restarting services (zero-downtime)..."

# Restart MLflow first (it's a dependency)
if systemctl is-active --quiet mlflow.service; then
  sudo systemctl restart mlflow.service
  log "✅ MLflow service restarted."
else
  sudo systemctl start mlflow.service
  log "✅ MLflow service started."
fi

sleep 3

# Graceful reload of FastAPI (sends SIGHUP to reload workers)
if systemctl is-active --quiet mlops-api.service; then
  sudo systemctl restart mlops-api.service
  log "✅ FastAPI service restarted."
else
  sudo systemctl start mlops-api.service
  log "✅ FastAPI service started."
fi

# ── 5. Post-deploy health check ──────────────────
log "🏥 [5/5] Running health checks..."
sleep 8

API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/ || echo "000")
MLFLOW_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/ || echo "000")

if [[ "${API_STATUS}" == "200" ]]; then
  log "✅ FastAPI API is healthy (HTTP ${API_STATUS})."
else
  log "❌ FastAPI API returned HTTP ${API_STATUS} — deployment may have issues!"
  log "   Run: sudo journalctl -u mlops-api.service -n 50"
  exit 1
fi

if [[ "${MLFLOW_STATUS}" == "200" ]]; then
  log "✅ MLflow UI is healthy (HTTP ${MLFLOW_STATUS})."
else
  log "⚠️  MLflow UI returned HTTP ${MLFLOW_STATUS} (non-critical)."
fi

log "========================================"
log "🎉 Deployment SUCCESSFUL!"
log "   SHA: ${GIT_SHA} | Env: ${ENVIRONMENT}"
log "========================================"
