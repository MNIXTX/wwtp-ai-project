@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Starting...

cd /d "%~dp0"

:: ==========================================
:: 1. Check venv
:: ==========================================
if not exist "venv\Scripts\python.exe" (
    echo.
    echo   [Error] Virtual environment not found.
    echo   Please run install.bat first.
    echo.
    pause
    exit /b 1
)

:: ==========================================
:: 2. Check if already running (via PID file)
:: ==========================================
set "PIDFILE=%~dp0.server.pid"
if exist "%PIDFILE%" (
    set /p OLD_PID=<"%PIDFILE%"
    tasklist /FI "PID eq !OLD_PID!" 2>nul | find "!OLD_PID!" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [Info] Server already running (PID: !OLD_PID!)
        start http://127.0.0.1:8501
        timeout /t 3 >nul
        exit /b 0
    )
    del "%PIDFILE%" >nul 2>&1
)

:: ==========================================
:: 3. Port check: is something already on 8501?
:: ==========================================
:: Try netstat first (always works, no admin needed)
netstat -ano 2>nul | find ":8501" | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo   [Info] Port 8501 already in use. Server appears to be running.
    start http://127.0.0.1:8501
    timeout /t 3 >nul
    exit /b 0
)

:: ==========================================
:: 4. Start Streamlit
::    Use python.exe (not pythonw) so crash errors are visible in the console.
::    The console stays open and shows live server logs.
:: ==========================================
echo.
echo   ========================================
echo     Starting WWTP AI Platform...
echo     Console will show server logs.
echo     Keep this window open while using the app.
echo     To stop: Ctrl+C or double-click stop.bat
echo   ========================================
echo.

:: Start Streamlit in this console window (shows all output + errors)
start "WWTP AI Server" "venv\Scripts\python.exe" -m streamlit run ui/app.py ^
    --server.headless=true ^
    --server.port=8501 ^
    --server.address=127.0.0.1 ^
    --browser.gatherUsageStats=false ^
    --server.enableXsrfProtection=false ^
    --server.enableCORS=false

:: Give it a moment to get the PID
timeout /t 3 >nul

:: Capture PID of the python process we just started
for /f "tokens=2" %%a in ('tasklist /FI "WINDOWTITLE eq WWTP AI Server" /FO LIST 2^>nul ^| find "PID:"') do (
    echo %%a > "%PIDFILE%"
)

:: ==========================================
:: 5. Wait for server (netstat fallback)
:: ==========================================
echo   Waiting for server to be ready...
set /a RETRIES=0
:wait_loop
timeout /t 2 >nul
set /a RETRIES+=1

:: Check port with netstat (works without PowerShell/admin)
netstat -ano 2>nul | find ":8501" | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 goto :ready

if !RETRIES! geq 20 (
    echo.
    echo   ========================================
    echo   [Error] Server failed to start (timeout after 40s)
    echo.
    echo   Troubleshooting:
    echo   1. Check the server console window for errors
    echo   2. Run manually to see detailed errors:
    echo      venv\Scripts\python.exe -m streamlit run ui/app.py --server.port=8501
    echo   3. Run diagnostic:
    echo      venv\Scripts\python.exe scripts\diagnose.py
    echo   4. Check logs: logs\app_*.log
    echo   ========================================
    pause
    exit /b 1
)
goto :wait_loop

:ready
start http://127.0.0.1:8501

:: ==========================================
:: 6. Done
:: ==========================================
echo   ========================================
echo     Platform is running!
echo     http://127.0.0.1:8501
echo     To stop: double-click stop.bat
echo   ========================================
timeout /t 3 >nul
exit /b 0
