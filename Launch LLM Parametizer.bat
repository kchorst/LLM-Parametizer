@echo off
:: LLM Parametizer Launcher
:: Double-click to start the application

title LLM Parametizer

echo Starting LLM Parametizer...
echo.

:: Change to the script's directory
cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10 or later from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Run the application
python main.py

if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
