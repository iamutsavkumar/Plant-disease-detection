#!/usr/bin/env bash
# scripts/start-frontend.sh — Start the Vite dev server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../frontend"

echo "🌿 Starting PlantMD Frontend..."

cd "$FRONTEND_DIR"

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

echo "   Dev server: http://localhost:3000"
echo "   Press Ctrl+C to stop"
echo ""

npm run dev
