#!/usr/bin/env bash
# scripts/setup.sh — One-time setup: install all dependencies and copy .env files
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

echo ""
echo "╔══════════════════════════════════╗"
echo "║  🌿 PlantMD — Project Setup      ║"
echo "╚══════════════════════════════════╝"
echo ""

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[ 1/3 ] Installing backend Node.js packages..."
cd "$ROOT/backend"
[ ! -f ".env" ] && cp .env.example .env && echo "  ✓ backend/.env created"
npm install
echo "  ✓ Backend ready"

# ── Frontend ──────────────────────────────────────────────────────────────────
echo ""
echo "[ 2/3 ] Installing frontend Node.js packages..."
cd "$ROOT/frontend"
[ ! -f ".env" ] && cp .env.example .env && echo "  ✓ frontend/.env created"
npm install
echo "  ✓ Frontend ready"

# ── AI Server ─────────────────────────────────────────────────────────────────
echo ""
echo "[ 3/3 ] Setting up Python AI server..."
cd "$ROOT/ai-server"
[ ! -f ".env" ] && cp .env.example .env && echo "  ✓ ai-server/.env created"

if ! command -v python3 &>/dev/null; then
  echo "  ⚠  python3 not found — skipping venv creation"
else
  if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Created Python virtualenv"
  fi
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  echo "  ✓ Python packages installed"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete! Start the app:                      ║"
echo "║                                                      ║"
echo "║  Terminal 1:  ./scripts/start-ai.sh                 ║"
echo "║  Terminal 2:  ./scripts/start-backend.sh            ║"
echo "║  Terminal 3:  ./scripts/start-frontend.sh           ║"
echo "║                                                      ║"
echo "║  Or with Docker:  docker-compose up --build         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
