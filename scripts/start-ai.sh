#!/usr/bin/env bash
# scripts/start-ai.sh — Start the FastAPI AI inference server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
AI_DIR="$ROOT/ai-server"

echo "🌿 Starting PlantMD AI Server..."

cd "$AI_DIR"

# Activate virtualenv if present
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  echo "✓ Activated virtualenv"
elif [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "✓ Activated .venv"
fi

# Copy .env if not present
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "✓ Created .env from .env.example"
fi

PORT=${PORT:-8000}
echo "   Listening on http://localhost:$PORT"
echo "   Press Ctrl+C to stop"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port "$PORT"
