@echo off
REM Fraud Detection API Startup Script

echo ========================================
echo Fraud Message Detection API
echo ========================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Python found!
echo.

REM Install dependencies
echo Installing dependencies...
cd python
pip install -r fraud_requirements.txt
echo.

REM Start the fraud detection server
echo Starting Fraud Detection API on port 8000...
echo.
python fraud_detector.py

pause
