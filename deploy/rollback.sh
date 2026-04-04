#!/usr/bin/env bash
# =============================================================================
# deploy/rollback.sh
# ROLLBACK SCRIPT — reverts the server to a specific git commit
#
# Usage:
#   bash rollback.sh <git_sha>
#   bash rollback.sh abc1234
# =============================================================================

set -euo pipefail

TARGET_SHA="${1:?Error: provide a git SHA to roll back to}"
APP_DIR="${HOME}/mlops-pipeline"
VENV="${APP_DIR}/venv"
LOG_FILE="${APP_DIR}/logs/deploy.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[${TIMESTAMP}] $*" | tee -a "${LOG_FILE}"; }

log "========================================"
log "⏪ ROLLBACK initiated"
log "   Target SHA : ${TARGET_SHA}"
log "========================================"

cd "${APP_DIR}"

log "📥 Checking out ${TARGET_SHA}..."
git fetch origin --quiet
git checkout "${TARGET_SHA}" -- .

log "📦 Reinstalling dependencies for this version..."
"${VENV}/bin/pip" install -r requirements.txt --quiet

log "🔄 Restarting services..."
sudo systemctl restart mlops-api.service
sudo systemctl restart mlflow.service

sleep 8

API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/ || echo "000")
if [[ "${API_STATUS}" == "200" ]]; then
  log "✅ Rollback successful — API is healthy at ${TARGET_SHA}."
else
  log "❌ API unhealthy after rollback (HTTP ${API_STATUS}) — MANUAL intervention needed!"
  exit 1
fi

log "========================================"
log "⏪ ROLLBACK complete."
log "========================================"
