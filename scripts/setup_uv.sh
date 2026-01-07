#!/bin/bash
set -e

echo "🌊 Setting up Dolphin Parser with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null
then
    echo "❌ UV is not installed. Installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Create a clean virtual environment in the current directory
echo "🧹 Creating a clean virtual environment (.venv-modal)..."
uv venv .venv-modal

# Synchronize dependencies from pyproject.toml
echo "📥 Installing only essential client dependencies..."
# Point uv to the specific venv
export UV_PROJECT_ENVIRONMENT=.venv-modal
uv sync

echo "✅ Clean environment setup complete!"
echo "🚀 To activate, run: source .venv/bin/activate"
echo "💡 To run commands without activating, use: uv run <command>"
