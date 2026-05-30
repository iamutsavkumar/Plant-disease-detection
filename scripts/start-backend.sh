#!/usr/bin/env bash
# scripts/start-backend.sh — Start the Node.js Express backend
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

echo "🌿 Starting PlantMD Backend..."

cd "$BACKEND_DIR"

# Copy .env if not present
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "✓ Created .env from .env.example"
fi

# Install deps if needed
if [ ! -d "node_modules" ]; then
  echo "Installing npm packages..."
  npm install
fi

PORT=${PORT:-5000}
echo "   Listening on http://localhost:$PORT"
echo "   Press Ctrl+C to stop"
echo ""

npm run dev
