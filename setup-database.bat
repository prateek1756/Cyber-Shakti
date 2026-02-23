@echo off
REM Database Setup Script for CyberShakti Scam Alert System
REM This script sets up the MySQL database

echo ========================================
echo CyberShakti - Database Setup
echo ========================================
echo.

REM Check if MySQL is installed
where mysql >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: MySQL is not installed or not in PATH
    echo Please install MySQL from: https://dev.mysql.com/downloads/installer/
    echo.
    pause
    exit /b 1
)

echo MySQL found!
echo.

REM Prompt for MySQL root password
set /p MYSQL_PASSWORD="Enter MySQL root password: "
echo.

echo Creating database and tables...
echo.

REM Execute the schema file
mysql -u root -p%MYSQL_PASSWORD% < database\schema.sql

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Database setup complete.
    echo ========================================
    echo.
    echo Database: cybershakti
    echo Tables created:
    echo   - scam_reports
    echo   - admin_users
    echo.
    echo Sample data inserted: 3 scam reports
    echo.
    echo Next steps:
    echo 1. Copy .env.example to .env
    echo 2. Update DB_PASSWORD in .env with your MySQL password
    echo 3. Run: npm run dev
    echo 4. Visit: http://localhost:5173/scam-alerts
    echo.
) else (
    echo.
    echo ========================================
    echo ERROR: Database setup failed
    echo ========================================
    echo.
    echo Please check:
    echo 1. MySQL password is correct
    echo 2. MySQL service is running
    echo 3. You have permissions to create databases
    echo.
)

pause
