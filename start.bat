@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Starting...

:: Convert to short 8.3 path to avoid issues with Chinese/parentheses/spaces
pushd "%~dp0"
set "PROJECT_ROOT_LONG=%~dp0"
if "!PROJECT_ROOT_LONG:~-1!"=="\" set "PROJECT_ROOT_LONG=!PROJECT_ROOT_LONG:~0,-1%"
for %%i in ("%~dp0.") do set "PROJECT_ROOT=%%~sfi"
if "!PROJECT_ROOT:~-1!"=="\" set "PROJECT_ROOT=!PROJECT_ROOT:~0,-1%"

:: ==========================================
:: UTF-8 mode: prevents GBK decode errors on Chinese Windows
:: ==========================================
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: ==========================================
:: Path-length guard
:: ==========================================
call :StrLen "!PROJECT_ROOT!"
if !STRLEN! gtr 100 (
    echo.
    echo   [Error] Project path is very long (!STRLEN! characters^).
    echo   Current path: !PROJECT_ROOT_LONG!
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
:: 2. Pre-flight checks before launching server
:: ==========================================
echo   Running pre-flight checks...

:: Ensure logs directory exists (needed for error capture below)
if not exist "!PROJECT_ROOT!\logs" mkdir "!PROJECT_ROOT!\logs" >nul 2>&1

:: 2a. Check config.yaml exists
if not exist "config.yaml" (
    echo   [Error] config.yaml not found. The app cannot start without it.
    pause
    exit /b 1
)

:: 2b. Quick import check (wider than just numpy/pandas/streamlit)
"venv\Scripts\python.exe" -c "import numpy, pandas, streamlit, lightgbm, onnxruntime, yaml, loguru" 2>"!PROJECT_ROOT!\logs\startup_error.log"
if !errorlevel! neq 0 (
    echo   [Error] Virtual environment is broken or incomplete.
    echo.
    echo   Common causes:
    echo   1. install.bat was not run on this machine
    echo   2. The venv was copied from another computer
    echo      ^(venvs are NOT portable - must be recreated^)
    echo   3. Some packages failed to install
    echo.
    echo   Error details ^(last 10 lines^):
    if exist "!PROJECT_ROOT!\logs\startup_error.log" (
        powershell -NoProfile -Command "Get-Content '!PROJECT_ROOT!\logs\startup_error.log' -Tail 10" 2>nul
    )
    echo.
    echo   Fix: Delete the venv folder and run install.bat:
    echo     rmdir /s /q venv
    echo     install.bat
    echo.
    pause
    exit /b 1
)

echo   [OK] Pre-flight checks passed.
echo.

:: ==========================================
:: 2. Check if already running
:: ==========================================
set "PIDFILE=!PROJECT_ROOT!\.server.pid"

:: Fast shortcut: PID file from a previous start.bat run
if exist "%PIDFILE%" (
    set /p OLD_PID=<"%PIDFILE%"
    tasklist /FI "PID eq !OLD_PID!" 2>nul | find "!OLD_PID!" >nul 2>&1
    if !errorlevel! equ 0 (
        REM Verify the PID actually belongs to a Streamlit process
        tasklist /FI "PID eq !OLD_PID!" /FO CSV 2>nul | find /i "python" >nul 2>&1
        if !errorlevel! equ 0 (
            echo   [Info] Server already running (PID: !OLD_PID!)
            goto :AlreadyRunning
        )
    )
    REM Stale PID file — clean up
    del "%PIDFILE%" >nul 2>&1
)

:: Primary check: is port 8501 already listening?
netstat -ano 2>nul | find ":8501 " | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo   [Info] Port 8501 already in use — verifying server is responsive...
    REM Quick health check: only redirect if server actually responds
    powershell -NoProfile -Command "try {$r=Invoke-WebRequest http://127.0.0.1:8501 -TimeoutSec 3 -UseBasicParsing; exit 0} catch {exit 1}" >nul 2>&1
    if !errorlevel! equ 0 (
        goto :AlreadyRunning
    ) else (
        echo   [Warn] Port 8501 occupied by non-responsive process.
        :: Kill the stale process so the new instance can bind the port
        :: Use tokens=* to get full line, then extract last token (PID is always last column)
        for /f "tokens=*" %%L in ('netstat -ano 2^>nul ^| find ":8501 " ^| find "LISTENING"') do (
            for %%P in (%%L) do set "STALE_PID=%%P"
            if defined STALE_PID (
                taskkill /F /PID !STALE_PID! >nul 2>&1
                echo   [Info] Killed stale PID !STALE_PID! on port 8501
            )
        )
        timeout /t 1 >nul
    )
)

:: ==========================================
:: 3b. (Optional) Start FastAPI inference server
::     Uncomment to enable dedicated inference backend
:: ==========================================
:: start "WWTP AI API" "!PROJECT_ROOT!\venv\Scripts\python.exe" -m uvicorn api_server:app --host 127.0.0.1 --port 8502
:: timeout /t 2 >nul

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

:: Start Streamlit in a new console window (shows live server logs).
:: CRITICAL: redirect stderr to a crash log so errors are NOT lost
:: if the server window closes before the user can read them.
set "CRASHLOG=!PROJECT_ROOT!\logs\streamlit_crash.log"
:: Write a temp helper batch file to avoid nested-quote parsing issues
:: with paths that contain parentheses, spaces, or Chinese characters.
set "HELPER=%TEMP%\wwtp_start_!RANDOM!.bat"
> "%HELPER%" echo @"!PROJECT_ROOT!\venv\Scripts\python.exe" -m streamlit run "!PROJECT_ROOT!\ui\app.py" --server.headless=true --server.port=8501 --server.address=127.0.0.1 --browser.gatherUsageStats=false --server.enableXsrfProtection=false --server.enableCORS=false --server.fileWatcherType=none --server.maxUploadSize=200 ^> "%CRASHLOG%" 2^>^&1
start "WWTP AI Server" "%HELPER%"

:: Give Streamlit time to start (3s initial — slower machines may need more)
timeout /t 3 /nobreak >nul

:: Quick crash detection: if Streamlit exited immediately, the crash log
:: will have content. Show it NOW rather than waiting 55s for timeout.
if exist "%CRASHLOG%" (
    for %%A in ("%CRASHLOG%") do if %%~zA gtr 0 (
        echo.
        echo   [ERROR] Streamlit crashed on startup!
        echo   --- Crash log ---
        type "%CRASHLOG%" 2>nul
        echo   --- End of crash log ---
        echo.
        echo   Common causes on new computers:
        echo   [*] Missing Visual C++ Redistributable
        echo       Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
        echo   [*] Corrupted package install — re-run install.bat
        echo   [*] config.yaml has syntax errors
        echo.
        pause
        exit /b 1
    )
)

:: ==========================================
:: 5. Wait for server + capture PID (adaptive intervals)
:: ==========================================
echo   Waiting for server to be ready...
set /a RETRIES=0
:wait_loop
:: Adaptive backoff: 1s for first 10 tries, 2s for next 10, 3s after
set /a WAIT=1
if !RETRIES! geq 10 set /a WAIT=2
if !RETRIES! geq 20 set /a WAIT=3
timeout /t !WAIT! /nobreak >nul
set /a RETRIES+=1

:: Check port with netstat (works without PowerShell/admin)
netstat -ano 2>nul | find ":8501 " | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    :: Capture PID on first successful port detection
    if not exist "%PIDFILE%" (
        for /f "tokens=*" %%L in ('netstat -ano 2^>nul ^| find ":8501 " ^| find "LISTENING"') do (
            for %%P in (%%L) do set "NEW_PID=%%P"
            echo !NEW_PID! > "%PIDFILE%" & goto :PidCaptured
        )
    )
    goto :ready
)

if !RETRIES! geq 25 (
    echo.
    echo   ========================================
    echo   [Error] Server failed to start (timeout after 55s)
    echo.
    echo   Troubleshooting:
    echo   1. Check the server console window for errors
    echo      ^(if it closed already, error was captured to the crash log^)
    echo.
    if exist "%CRASHLOG%" (
        for %%A in ("%CRASHLOG%") do if %%~zA gtr 0 (
            echo   --- Streamlit crash log ^(last 20 lines^) ---
            powershell -NoProfile -Command "Get-Content '%CRASHLOG%' -Tail 20" 2>nul
            echo   --- End of crash log ---
            echo.
        )
    )
    echo   2. Run manually to see detailed errors:
    echo      "!PROJECT_ROOT!\venv\Scripts\python.exe" -m streamlit run "!PROJECT_ROOT!\ui\app.py" --server.port=8501
    echo   3. Run diagnostic:
    echo      venv\Scripts\python.exe utils\diagnose.py
    echo   4. Check logs: logs\app_*.log
    echo   5. Crash log: %CRASHLOG%
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
:: Helper: string length (never falls through)
:: ==========================================
goto :EOF
:StrLen
setlocal
set "STR=%~1"
set "STRLEN=0"
if not defined STR goto :StrLenEnd
:StrLenLoop
if "!STR!"=="" goto :StrLenEnd
set "STR=!STR:~1!"
set /a STRLEN+=1
goto :StrLenLoop
:StrLenEnd
endlocal & set "STRLEN=%STRLEN%"
goto :EOF

:: Jump target for PID capture in wait loop (kept outside blocks)
:PidCaptured
goto :ready

:: ==========================================
:: Shared exit: server is already running
:: ==========================================
:AlreadyRunning
:: Ensure PID file exists for stop.bat / future start.bat runs
if not exist "%PIDFILE%" (
    for /f "tokens=*" %%L in ('netstat -ano 2^>nul ^| find ":8501 " ^| find "LISTENING"') do (
        for %%P in (%%L) do set "EXIST_PID=%%P"
        echo !EXIST_PID! > "%PIDFILE%" & goto :AlreadyShow
    )
    :: If we reach here, netstat didn't find the port — server may have just died
    echo   [Error] Server was detected earlier but port 8501 is no longer listening.
    echo   The Streamlit process may have crashed. Check logs\streamlit_crash.log
    pause
    exit /b 1
)
:AlreadyShow
echo   ========================================
echo     Server is already running.
echo     Opening browser: http://127.0.0.1:8501
echo   ========================================
start http://127.0.0.1:8501
timeout /t 2 /nobreak >nul
exit /b 0
