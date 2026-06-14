@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Stopping...

pushd "%~dp0"
for %%i in ("%~dp0.") do set "PROJECT_ROOT=%%~sfi"
if "!PROJECT_ROOT:~-1!"=="\" set "PROJECT_ROOT=!PROJECT_ROOT:~0,-1%"
set "PIDFILE=!PROJECT_ROOT!\.server.pid"

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
for /f "tokens=*" %%L in ('netstat -ano 2^>nul ^| find ":8501 " ^| find "LISTENING"') do (
    for %%P in (%%L) do set "KILL_PID=%%P"
    if defined KILL_PID (
        taskkill /F /PID !KILL_PID! >nul 2>&1
        echo   Stopped PID: !KILL_PID! (port 8501)
    )
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
