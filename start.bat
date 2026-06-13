@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Starting...

cd /d "%~dp0"

:: ==========================================
:: UTF-8 mode: prevents GBK decode errors on Chinese Windows
:: ==========================================
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: ==========================================
:: Path-length guard: WeChat / USB / network drives can exceed MAX_PATH
:: ==========================================
for /f %%a in ('powershell -NoProfile -Command "if('%~dp0'.Length -gt 230){'1'}else{'0'}"') do set "PATH_TOO_LONG=%%a"
if "!PATH_TOO_LONG!"=="1" (
    echo.
    echo   [Error] Project path is too long ^(over 230 characters^).
    echo   Current path: %~dp0
    echo.
    echo   Windows MAX_PATH ^(260 chars^) may cause failures when
    echo   Streamlit or Python access nested files.
    echo.
    echo   Move the project folder closer to the drive root, e.g.:
    echo     C:\WWTP_AI_System\
    echo.
    pause
    exit /b 1
)

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
:: 1b. Quick sanity check: can the venv import core packages?
::     Catches broken/copied venvs (e.g. shared via WeChat/USB)
:: ==========================================
"venv\Scripts\python.exe" -c "import numpy, pandas, streamlit" 2>nul
if !errorlevel! neq 0 (
    echo.
    echo   [Error] Virtual environment is broken or incomplete.
    echo.
    echo   Common causes:
    echo   1. install.bat was not run on this machine
    echo   2. The venv was copied from another computer
    echo      ^(venvs are NOT portable - must be recreated^)
    echo.
    echo   Fix: Delete the venv folder and run install.bat:
    echo     rmdir /s /q venv
    echo     install.bat
    echo.
    pause
    exit /b 1
)

:: ==========================================
:: 2. Check if already running
:: ==========================================
set "PIDFILE=%~dp0.server.pid"

:: Fast shortcut: PID file from a previous start.bat run
if exist "%PIDFILE%" (
    set /p OLD_PID=<"%PIDFILE%"
    tasklist /FI "PID eq !OLD_PID!" 2>nul | find "!OLD_PID!" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [Info] Server already running (PID: !OLD_PID!)
        goto :AlreadyRunning
    )
    del "%PIDFILE%" >nul 2>&1
)

:: Primary check: is port 8501 already listening?
netstat -ano 2>nul | find ":8501" | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo   [Info] Port 8501 already in use.
    goto :AlreadyRunning
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
:: Use absolute paths so the server window finds ui/app.py regardless of its CWD
start "WWTP AI Server" "%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0ui\app.py" ^
    --server.headless=true ^
    --server.port=8501 ^
    --server.address=127.0.0.1 ^
    --browser.gatherUsageStats=false ^
    --server.enableXsrfProtection=false ^
    --server.enableCORS=false

:: Give it a moment for the server to bind the port
timeout /t 1 >nul

:: Capture PID from the newly-opened port
:: netstat -ano output: TCP  0.0.0.0:8501  0.0.0.0:0  LISTENING  12345
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| find ":8501" ^| find "LISTENING"') do (
    echo %%a > "%PIDFILE%"
    goto :PidDone
)
:: Fallback: window title
for /f "tokens=2" %%a in ('tasklist /FI "WINDOWTITLE eq WWTP AI Server" /FO LIST 2^>nul ^| find "PID:"') do (
    echo %%a > "%PIDFILE%"
)
:PidDone

:: ==========================================
:: 5. Wait for server (adaptive intervals)
:: ==========================================
echo   Waiting for server to be ready...
set /a RETRIES=0
:wait_loop
:: Adaptive backoff: 1s for first 10 tries, 2s for next 10, 3s after
set /a WAIT=1
if !RETRIES! geq 10 set /a WAIT=2
if !RETRIES! geq 20 set /a WAIT=3
timeout /t !WAIT! >nul
set /a RETRIES+=1

:: Check port with netstat (works without PowerShell/admin)
netstat -ano 2>nul | find ":8501" | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 goto :ready

if !RETRIES! geq 25 (
    echo.
    echo   ========================================
    echo   [Error] Server failed to start (timeout after 55s)
    echo.
    echo   Troubleshooting:
    echo   1. Check the server console window for errors
    echo   2. Run manually to see detailed errors:
    echo      "%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0ui\app.py" --server.port=8501
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

:: ==========================================
:: Shared exit: server is already running
:: ==========================================
:AlreadyRunning
start http://127.0.0.1:8501
timeout /t 3 >nul
exit /b 0
