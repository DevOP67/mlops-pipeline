#!/usr/bin/env bash
# =============================================================================
# deploy/setup_server.sh
# ONE-TIME server provisioning script for MLOps Pipeline
# Run this ONCE on a fresh Ubuntu 20.04/22.04 VPS/VM
#
# Usage:
#   chmod +x setup_server.sh
#   sudo bash setup_server.sh
# =============================================================================

set -euo pipefail
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
APP_DIR="/home/${DEPLOY_USER}/mlops-pipeline"
PYTHON_VERSION="3.10"

echo "=============================================="
echo "  MLOps Pipeline — Server Setup Script"
echo "=============================================="

# ── 1. System Updates ─────────────────────────────
echo ""
echo "📦 [1/7] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-venv \
  python${PYTHON_VERSION}-dev \
  python3-pip \
  git \
  curl \
  wget \
  rsync \
  nginx \
  ufw \
  htop \
  build-essential

echo "✅ System packages installed."

# ── 2. Create Application Directory ──────────────
echo ""
echo "📁 [2/7] Creating application directories..."
mkdir -p "${APP_DIR}"/{mlruns,artifacts,logs,deploy}
chown -R ${DEPLOY_USER}:${DEPLOY_USER} "${APP_DIR}"
echo "✅ Application directory: ${APP_DIR}"

# ── 3. Python Virtual Environment ────────────────
echo ""
echo "🐍 [3/7] Setting up Python virtual environment..."
sudo -u ${DEPLOY_USER} python${PYTHON_VERSION} -m venv "${APP_DIR}/venv"
echo "✅ Virtual environment created."

# ── 4. Configure Firewall ─────────────────────────
echo ""
echo "🔒 [4/7] Configuring UFW firewall..."
ufw --force enable
ufw allow OpenSSH
ufw allow 8000/tcp comment "FastAPI MLOps API"
ufw allow 5000/tcp comment "MLflow Tracking UI"
ufw allow 80/tcp comment "Nginx HTTP"
ufw allow 443/tcp comment "Nginx HTTPS"
ufw status
echo "✅ Firewall rules applied."

# ── 5. Install systemd Services ──────────────────
echo ""
echo "⚙️  [5/7] Installing systemd services..."

# Copy service files
cp -f "${APP_DIR}/deploy/mlops-api.service" /etc/systemd/system/mlops-api.service
cp -f "${APP_DIR}/deploy/mlflow.service"    /etc/systemd/system/mlflow.service

# Reload systemd and enable services
systemctl daemon-reload
systemctl enable mlops-api.service
systemctl enable mlflow.service

echo "✅ systemd services registered."

# ── 6. Configure Nginx Reverse Proxy ─────────────
echo ""
echo "🌐 [6/7] Configuring Nginx reverse proxy..."
cat > /etc/nginx/sites-available/mlops << 'NGINX_CONF'
# MLOps Pipeline — Nginx Reverse Proxy
upstream mlops_api {
    server 127.0.0.1:8000;
}

upstream mlflow_ui {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    # FastAPI — main API
    location /api/ {
        proxy_pass         http://mlops_api/;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    # MLflow UI
    location /mlflow/ {
        proxy_pass         http://mlflow_ui/;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://mlops_api/;
        access_log off;
    }
}
NGINX_CONF

ln -sf /etc/nginx/sites-available/mlops /etc/nginx/sites-enabled/mlops
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
echo "✅ Nginx configured."

# ── 7. Final Summary ──────────────────────────────
echo ""
echo "=============================================="
echo "  ✅ Server Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Push your SSH public key to: ~/.ssh/authorized_keys"
echo "  2. Add GitHub Secrets:"
echo "     - SSH_PRIVATE_KEY    → your private key"
echo "     - PROD_SERVER_IP     → this server's IP"
echo "     - STAGING_SERVER_IP  → staging server IP"
echo "     - SERVER_USER        → ${DEPLOY_USER}"
echo "  3. Push code to 'main' to trigger CI/CD."
echo ""
echo "  FastAPI API  → http://SERVER_IP:8000"
echo "  MLflow UI    → http://SERVER_IP:5000"
echo "  Via Nginx    → http://SERVER_IP/api/"
echo ""
