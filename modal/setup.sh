#!/bin/bash
# ============================================
# Dolphin Modal Deployment Setup Script
# ============================================
# This script sets up the Modal CLI and deploys
# the Dolphin PDF parser to Modal cloud.
# ============================================

set -e

echo "🐬 Dolphin PDF Parser - Modal Setup"
echo "===================================="
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "📦 Installing Modal CLI..."
    pip install modal
    echo "✅ Modal installed!"
else
    echo "✅ Modal CLI is already installed"
fi

echo ""

# Check if user is logged in
echo "🔐 Checking Modal authentication..."
if ! modal token status &> /dev/null 2>&1; then
    echo ""
    echo "⚠️  You need to authenticate with Modal."
    echo "   Run: modal token new"
    echo ""
    echo "   This will open a browser window to log in."
    echo "   After logging in, run this script again."
    exit 1
else
    echo "✅ Authenticated with Modal"
fi

echo ""

# Prompt for action
echo "What would you like to do?"
echo "  1) Deploy to production (modal deploy)"
echo "  2) Run in development mode (modal serve)"
echo "  3) Exit"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Deploying to Modal..."
        cd "$(dirname "$0")"
        modal deploy dolphin_modal.py
        echo ""
        echo "✅ Deployment complete!"
        echo "   Your app is now live at the URL shown above."
        ;;
    2)
        echo ""
        echo "🔧 Starting development server..."
        echo "   Press Ctrl+C to stop."
        echo ""
        cd "$(dirname "$0")"
        modal serve dolphin_modal.py
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
