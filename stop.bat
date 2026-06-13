@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Stopping...

cd /d "%~dp0"
set "PIDFILE=%~dp0.server.pid"

:: ==========================================
:: 1. Kill by PID file
:: ==========================================
if exist "%PIDFILE%" (
    set /p PID=<"%PIDFILE%"
    tasklist /FI "PID eq !PID!" 2>nul | find "!PID!" >nul 2>&1
    if !errorlevel! equ 0 (
        taskkill /F /PID !PID! >nul 2>&1
        echo   Stopped server (PID: !PID!)
    )
    del "%PIDFILE%" >nul 2>&1
)

:: ==========================================
:: 2. Kill anything on port 8501 (netstat)
:: ==========================================
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| find ":8501" ^| find "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    echo   Stopped PID: %%a (port 8501)
)

:: ==========================================
:: 3. Kill "WWTP AI Server" window
:: ==========================================
taskkill /FI "WINDOWTITLE eq WWTP AI Server" /F >nul 2>&1

:: ==========================================
:: 4. Done
:: ==========================================
echo   ========================================
echo     Server stopped.
echo     To restart: double-click start.bat
echo   ========================================
timeout /t 2 >nul
exit /b 0
