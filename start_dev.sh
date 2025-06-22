#!/bin/bash
# LangGraph Development Startup Script

echo "🚀 Starting LangGraph Development Environment..."

# Install project in development mode
echo "📦 Installing project in development mode..."
python -m pip install -e .

# Clean up any existing cache/checkpoint files
if [ -d ".langgraph_api" ]; then
    echo "🧹 Cleaning up existing LangGraph cache..."
    rm -rf .langgraph_api
fi

# Remove any pickle checkpoint files
if ls .langgraph_checkpoint*.pckl* 1> /dev/null 2>&1; then
    echo "🧹 Cleaning up checkpoint files..."
    rm -f .langgraph_checkpoint*.pckl*
fi

# Set environment variable for blocking calls
export BG_JOB_ISOLATED_LOOPS=true

# Start LangGraph with blocking calls allowed
echo "🎯 Starting LangGraph with development settings..."
langgraph dev --allow-blocking
