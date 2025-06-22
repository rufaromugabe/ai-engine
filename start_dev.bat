@echo off
REM LangGraph Development Startup Script for Windows

echo 🚀 Starting LangGraph Development Environment...

REM Install project in development mode
echo 📦 Installing project in development mode...
python -m pip install -e .

REM Clean up any existing cache/checkpoint files
if exist ".langgraph_api" (
    echo 🧹 Cleaning up existing LangGraph cache...
    rmdir /s /q .langgraph_api
)

REM Remove any pickle checkpoint files
if exist ".langgraph_checkpoint*.pckl*" (
    echo 🧹 Cleaning up checkpoint files...
    del /f /q .langgraph_checkpoint*.pckl* 2>nul
)

REM Set environment variable for blocking calls
set BG_JOB_ISOLATED_LOOPS=true

REM Start LangGraph with blocking calls allowed
echo 🎯 Starting LangGraph with development settings...
langgraph dev --allow-blocking
