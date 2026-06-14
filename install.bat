@echo off
setlocal enabledelayedexpansion
goto :SkipHelpers

:: ============================================================
:: Helpers (must be at top so call :label always finds them)
:: ============================================================
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

:SkipHelpers
chcp 65001 >nul 2>&1
title WWTP AI System - Smart Installer

echo.
echo ============================================================
echo      WWTP AI - Smart Installer (uv + pip fallback)
echo ============================================================
echo.
echo This script will set up the complete Python environment
echo and install all dependencies automatically.
echo.

:: ============================================================
:: 1. Mode detection — MUST run before long-path guard so
::    /force and /silent take effect during the path check.
:: ============================================================
set "FORCE_MODE="
if /i "%~1"=="/offline" set "FORCE_MODE=OFFLINE"
if /i "%~2"=="/offline" set "FORCE_MODE=OFFLINE"
if /i "%~3"=="/offline" set "FORCE_MODE=OFFLINE"
if /i "%~1"=="/online"  set "FORCE_MODE=ONLINE"
if /i "%~2"=="/online"  set "FORCE_MODE=ONLINE"
if /i "%~3"=="/online"  set "FORCE_MODE=ONLINE"
if not "!FORCE_MODE!"=="" echo [Info] Mode override: !FORCE_MODE!

set "SILENT="
if /i "%~1"=="/silent" set "SILENT=1"
if /i "%~2"=="/silent" set "SILENT=1"
if /i "%~3"=="/silent" set "SILENT=1"
if defined SILENT echo [Info] Silent mode: no interactive prompts

:: /force — skip long-path prompt (combine with /silent for unattended)
if /i "%~1"=="/force" set "FORCE_CONTINUE=1"
if /i "%~2"=="/force" set "FORCE_CONTINUE=1"
if /i "%~3"=="/force" set "FORCE_CONTINUE=1"
if defined FORCE_CONTINUE echo [Info] Force mode: skip path-length prompt

:: ============================================================
:: 2. Detect working directory + long-path guard
::    Use 8.3 short path to avoid issues with Chinese chars,
::    parentheses, and spaces in the project path.
:: ============================================================
pushd "%~dp0"
set "PROJECT_ROOT_LONG=%~dp0"
:: Convert to short path (8.3) — eliminates special characters
for %%i in ("%~dp0.") do set "PROJECT_ROOT=%%~sfi"
:: Remove trailing backslash from short path
if "!PROJECT_ROOT:~-1!"=="\" set "PROJECT_ROOT=!PROJECT_ROOT:~0,-1!"
echo [Info] Project root: !PROJECT_ROOT_LONG!

set "STRLEN=0"
call :StrLen "!PROJECT_ROOT!"
set /a "MAX_SAFE_ROOT=100"
if !STRLEN! gtr !MAX_SAFE_ROOT! (
    echo.
    echo ============================================================
    echo [WARN] Project path is very long (!STRLEN! characters^).
    echo.
    echo   !PROJECT_ROOT!
    echo.
    echo   Windows MAX_PATH limit ^(260 chars^) may cause install failures.
    echo.
    echo   ==^> RECOMMENDED: Move the project to a shorter path, e.g.:
    echo         D:\WWTP_AI_System
    echo         C:\WWTP_AI_System
    echo.
    echo   Avoid: OneDrive, WeChat folders, Desktop, deep nesting.
    echo ============================================================
    echo.
    if not defined FORCE_CONTINUE (
        if not defined SILENT (
            echo   Continue anyway? (y/N, default=N^)
            set /p "CONTINUE_LONGPATH="
        ) else (
            set "CONTINUE_LONGPATH=N"
        )
        if /i not "!CONTINUE_LONGPATH!"=="Y" (
            echo [FATAL] Aborted due to long path. Move project and retry.
            if not defined SILENT pause
            exit /b 1
        )
    )
    echo [Info] Proceeding despite long path...
)

:: ============================================================
:: 3. Early checks
:: ============================================================
if not exist "requirements.txt" (
    echo.
    echo ============================================================
    echo [FATAL] requirements.txt not found!
    echo ============================================================
    if not defined SILENT pause
    exit /b 1
)

:: ============================================================
:: 3b. Early Python detection (before attempting uv/pip installs)
:: ============================================================
set "HAS_PYTHON="
where python >nul 2>&1 && set "HAS_PYTHON=1"
if not defined HAS_PYTHON (
    if exist "venv\Scripts\python.exe" set "HAS_PYTHON=1"
)
if not defined HAS_PYTHON (
    for %%d in (
        "!LOCALAPPDATA!\Programs\Python\Python310"
        "!LOCALAPPDATA!\Programs\Python\Python314"
        "!LOCALAPPDATA!\Programs\Python\Python313"
        "!LOCALAPPDATA!\Programs\Python\Python312"
        "!LOCALAPPDATA!\Programs\Python\Python311"
        "C:\Program Files\Python310"
        "C:\Program Files\Python314"
    ) do (
        if exist "%%~d\python.exe" set "HAS_PYTHON=1"
    )
)

:: ============================================================
:: 4. Detect / install uv
:: ============================================================
echo.
echo [1/7] Checking package manager...
set "HAS_UV="
where uv >nul 2>&1 && set "HAS_UV=1"

if defined HAS_UV (
    for /f "tokens=2" %%v in ('uv --version 2^>nul') do echo [OK] uv %%v detected
    goto :UvReady
)

:: uv not found — attempt to install
set "UV_INSTALLED=0"
if "!FORCE_MODE!"=="OFFLINE" (
    echo [!]  uv not found. Offline mode — will fall back to pip.
    goto :PipFallback
)

echo [Info] uv not found. Installing via official installer...
echo         (one-time setup, ~5 seconds^)

:: Try PowerShell installer (official)
powershell -NoProfile -ExecutionPolicy ByPass -Command "try { irm https://astral.sh/uv/install.ps1 | iex; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    :: Refresh PATH for current session
    set "PATH=!USERPROFILE!\.local\bin;!PATH!"
    where uv >nul 2>&1 && set "HAS_UV=1" && set "UV_INSTALLED=1"
)

if defined HAS_UV (
    echo [OK] uv installed successfully
    goto :UvReady
)

:: PowerShell installer failed — try pip (only if Python is available)
echo [Info] PowerShell installer failed. Trying pip install uv...
if defined HAS_PYTHON (
    python -m pip install uv --quiet 2>&1
    where uv >nul 2>&1 && set "HAS_UV=1" && set "UV_INSTALLED=2"
) else (
    echo [!]  No Python available to install uv via pip.
)

if defined HAS_UV (
    echo [OK] uv installed via pip
    goto :UvReady
)

:: Everything failed — fall back to pip
echo [!]  uv install failed. Falling back to pip.
goto :PipFallback

:UvReady

:: ============================================================
:: 5. Network detection
:: ============================================================
echo.
echo [2/7] Checking network...

if "!FORCE_MODE!"=="OFFLINE" (
    set "HAS_NET=0"
    echo [Info] Skipped (forced offline mode^)
    goto :ModeSet
)
if "!FORCE_MODE!"=="ONLINE" (
    set "HAS_NET=1"
    echo [Info] Skipped (forced online mode^)
    goto :ModeSet
)

set "HAS_NET=0"
for %%h in (pypi.org github.com baidu.com) do (
    if "!HAS_NET!"=="0" ping -n 1 -w 2000 "%%h" >nul 2>&1 && set "HAS_NET=1"
)
if "!HAS_NET!"=="0" (
    curl -s --connect-timeout 5 https://pypi.org >nul 2>&1 && set "HAS_NET=1"
)
if "!HAS_NET!"=="0" (
    powershell -NoProfile -Command "try {(Invoke-WebRequest https://pypi.org -TimeoutSec 5).StatusCode; exit 0} catch {exit 1}" >nul 2>&1 && set "HAS_NET=1"
)

:ModeSet
if "!HAS_NET!"=="1" (
    echo [OK] Network available - Online install mode
    set "MODE=ONLINE"
) else (
    echo [!]  No network - Offline install mode
    set "MODE=OFFLINE"
)

:: ============================================================
:: 6. Offline package validation
:: ============================================================
echo.
echo [3/7] Verifying offline packages...

:: Early check: verify the two largest/critical packages exist.
:: A full audit of all 122 wheels would be slow; missing packages
:: will be caught with a clear error at the pip/uv install step.
set "HAS_OFFLINE=0"
if exist "offline_packages\" (
    dir /b "offline_packages\torch-*.whl" >nul 2>&1 && dir /b "offline_packages\numpy-*.whl" >nul 2>&1 && set "HAS_OFFLINE=1"
)

if "!MODE!"=="OFFLINE" (
    if "!HAS_OFFLINE!"=="0" (
        echo.
        echo ============================================================
        echo [FATAL] Offline mode requires offline_packages\ directory
        echo with torch and numpy .whl files.
        echo ============================================================
        if not defined SILENT pause
        exit /b 1
    )
    echo [OK] Offline packages verified
)

:: ============================================================
:: 7. Python environment (uv handles everything)
:: ============================================================
echo.
echo [4/7] Setting up Python environment...

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: ---- Python discovery via uv ----
set "PYTHON_EXE="

if "!MODE!"=="OFFLINE" (
    :: Offline: use uv to find compatible Python (CP310 or CP314)
    for /f "delims=" %%p in ('uv python find 2^>nul') do (
        if exist "%%p" (
            set "PY_VER="
            for /f "delims=" %%v in ('"%%p" -c "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))" 2^>nul') do set "PY_VER=%%v"
            if "!PY_VER!"=="310" if exist "offline_packages\numpy*cp310*.whl" if exist "offline_packages\torch*cp310*.whl" (
                set "PYTHON_EXE=%%p"
                echo [OK] Found Python 3.10 (offline-compatible^): %%p
                goto :CreateVenv
            )
            if "!PY_VER!"=="314" if exist "offline_packages\numpy*cp314*.whl" if exist "offline_packages\torch*cp314*.whl" (
                set "PYTHON_EXE=%%p"
                echo [OK] Found Python 3.14 (offline-compatible^): %%p
                goto :CreateVenv
            )
        )
    )
    :: If uv found nothing compatible, try common paths with CP verification
    for %%d in (
        "!LOCALAPPDATA!\Programs\Python\Python310"
        "!LOCALAPPDATA!\Programs\Python\Python314"
        "C:\Program Files\Python310"
        "C:\Program Files\Python314"
        "C:\Python310" "C:\Python314"
    ) do (
        if exist "%%~d\python.exe" (
            set "FALLBACK_VER="
            for /f "delims=" %%v in ('"%%~d\python.exe" -c "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))" 2^>nul') do set "FALLBACK_VER=%%v"
            if "!FALLBACK_VER!"=="310" if exist "offline_packages\numpy*cp310*.whl" if exist "offline_packages\torch*cp310*.whl" (
                set "PYTHON_EXE=%%~d\python.exe"
                echo [OK] Found Python 3.10 (offline-compatible^): %%~d\python.exe
                goto :CreateVenv
            )
            if "!FALLBACK_VER!"=="314" if exist "offline_packages\numpy*cp314*.whl" if exist "offline_packages\torch*cp314*.whl" (
                set "PYTHON_EXE=%%~d\python.exe"
                echo [OK] Found Python 3.14 (offline-compatible^): %%~d\python.exe
                goto :CreateVenv
            )
            echo [Info] Python at %%~d\python.exe is CP!FALLBACK_VER! ^(offline packages not available^)
        )
    )
    echo [FATAL] No compatible Python found for offline mode.
    echo         Install Python 3.10 from offline_packages\python-3.10.11-amd64.exe
    if not defined SILENT pause
    exit /b 1
) else (
    :: Online: uv can download Python if needed
    for /f "delims=" %%p in ('uv python find 3.10 2^>nul') do (
        if exist "%%p" (
            set "PYTHON_EXE=%%p"
            goto :OnlinePythonFound
        )
    )
    for /f "delims=" %%p in ('uv python find 2^>nul') do (
        if exist "%%p" (
            set "PYTHON_EXE=%%p"
            goto :OnlinePythonFound
        )
    )
    echo [Info] No Python found. uv will download Python 3.10...
    uv python install 3.10 >nul 2>&1
    for /f "delims=" %%p in ('uv python find 3.10 2^>nul') do (
        if exist "%%p" (
            set "PYTHON_EXE=%%p"
            goto :OnlinePythonFound
        )
    )
)
:: Fall through to OnlinePythonFound below

:OnlinePythonFound
if defined PYTHON_EXE (
    echo [OK] Python: !PYTHON_EXE!
) else (
    echo [FATAL] Could not set up Python.
    if not defined SILENT pause
    exit /b 1
)

:CreateVenv
:: ---- Reuse or create venv ----
if exist "venv\Scripts\python.exe" (
    if exist "venv\Scripts\activate.bat" (
        echo [OK] Existing venv found, reusing
    ) else (
        echo [Info] Cleaning broken venv...
        rmdir /s /q "venv" >nul 2>&1
    )
) else (
    REM venv does not exist — will create below
)
:: Only create if venv doesn't exist (was cleaned or never existed)
if not exist "venv\Scripts\python.exe" goto :NewVenv
goto :VenvDone

:NewVenv
echo [Info] Creating virtual environment...
if defined HAS_UV (
    uv venv venv --python "!PYTHON_EXE!" --link-mode=copy 2>&1
) else (
    "!PYTHON_EXE!" -m venv venv
)
if not exist "venv\Scripts\python.exe" (
    :: Retry without pip
    if defined HAS_UV (
        uv venv venv --python "!PYTHON_EXE!" --seed --link-mode=copy 2>&1
    ) else (
        "!PYTHON_EXE!" -m venv --without-pip venv
        if exist "venv\Scripts\python.exe" (
            echo [Info] Bootstrapping pip into venv...
            "venv\Scripts\python.exe" -m ensurepip --default-pip >nul 2>&1
        )
    )
)
if not exist "venv\Scripts\python.exe" (
    echo [FATAL] Cannot create virtual environment.
    if not defined SILENT pause
    exit /b 1
)
echo [OK] Virtual environment created
:VenvDone

:: Activate
call venv\Scripts\activate.bat
if not defined VIRTUAL_ENV (
    echo [FATAL] Failed to activate venv.
    if not defined SILENT pause
    exit /b 1
)
echo [OK] Virtual environment active: !VIRTUAL_ENV!

:: ============================================================
:: 8. Install dependencies
:: ============================================================
echo.
echo [5/7] Installing dependencies (!MODE! mode^)...

if "!MODE!"=="ONLINE" (
    echo [Info] Installing from PyPI via uv...
    echo     (typically 10-60 seconds with uv^)
    echo.
    if defined HAS_UV (
        uv pip install --link-mode=copy -r requirements.txt
    ) else (
        pip install -r requirements.txt
    )
) else (
    echo [Info] Installing from offline_packages\...
    echo     (3-8 minutes^)
    echo.
    if defined HAS_UV (
        uv pip install --link-mode=copy --no-index --find-links=offline_packages -r requirements.txt
    ) else (
        pip install --no-index --find-links=offline_packages -r requirements.txt
    )
)

if !errorlevel! neq 0 (
    echo.
    echo ============================================================
    echo [FATAL] Dependency installation failed.
    echo.
    echo   Troubleshooting:
    echo   1. Check PATH LENGTH: if your project is inside OneDrive,
    echo      WeChat folders, or deeply nested directories,
    echo      Windows MAX_PATH ^(260 chars^) may cause failures.
    echo      ^>^> Move to a short path like D:\WWTP_AI_System
    echo   2. Online: check network / firewall
    echo   3. Offline: verify offline_packages\ has all .whl files
    echo ============================================================
    if not defined SILENT pause
    exit /b 1
)
echo [OK] Dependencies installed

:: ============================================================
:: 9. Verify installation
:: ============================================================
:RunVerification
echo.
echo [6/7] Verifying installation...

for %%d in (logs outputs artifacts "models\lgbm") do (
    if not exist "%%d" mkdir "%%d" >nul 2>&1
)

set "VERIFY_FAIL=0"
set "HAS_DIAGNOSE="
if exist "utils\diagnose.py" set "HAS_DIAGNOSE=1"

echo   [1/3] Core data packages...
python -c "import numpy, pandas, scipy, sklearn" >nul 2>&1
if !errorlevel! neq 0 (
    set "VERIFY_FAIL=1" & echo [WARN] Core data packages failed
) else (
    echo [OK]   numpy, pandas, scipy, scikit-learn
)

echo   [2/3] ML and ONNX packages...
python -c "import lightgbm, onnxruntime, onnx, onnxscript" >nul 2>&1
if !errorlevel! neq 0 (
    set "VERIFY_FAIL=1" & echo [WARN] ML/ONNX packages failed
) else (
    echo [OK]   lightgbm, onnxruntime, onnx, onnxscript
)

echo   [3/3] Application + optional packages...
python -c "import yaml, ruamel.yaml, pydantic, loguru, joblib, matplotlib, streamlit, gymnasium, torch, stable_baselines3, psutil, numba" >nul 2>&1
if !errorlevel! neq 0 (
    set "VERIFY_FAIL=1" & echo [WARN] Some application/optional packages failed
) else (
    echo [OK]   All packages verified
)

:: ============================================================
:: Done
:: ============================================================
echo.
echo ============================================================
if "!VERIFY_FAIL!"=="1" (
    echo   Installation completed with WARNINGS
    echo ============================================================
    echo.
    echo [!]  Some modules failed verification.
    echo.
    if defined HAS_DIAGNOSE echo     Run diagnostics: venv\Scripts\python.exe utils\diagnose.py
    echo     Re-run install.bat or manually fix missing packages.
    echo.
    echo   Mode: !MODE!
    echo   Python: !PYTHON_EXE!
    echo   Venv: !PROJECT_ROOT!venv
    echo ============================================================
    echo.
    (
        echo === WWTP AI System - Install Summary ===
        echo Date: !date! !time!
        echo Result: WARNINGS
        echo Mode: !MODE!
        echo Python: !PYTHON_EXE!
        echo Venv: !PROJECT_ROOT!venv
        echo.
        echo [!] Some modules failed verification.
    ) > install.log 2>&1
    echo [Info] Install log saved to install.log
    if not defined SILENT pause
    exit /b 2
) else (
    echo   Installation Complete!
    echo ============================================================
    echo.
    if defined HAS_UV (echo   Package manager: uv) else (echo   Package manager: pip)
    echo   Mode:   !MODE!
    echo   Python: !PYTHON_EXE!
    echo   Venv:   !PROJECT_ROOT!venv
    echo.
    echo   Tips:
    echo     - Force offline: install.bat /offline
    echo     - Force online:  install.bat /online
    echo     - Unattended:    install.bat /silent
    echo     - Skip path warn: install.bat /force
    echo     - Combine:       install.bat /offline /silent /force
    echo.
    echo   To launch:
    echo     Double-click: start.bat
    echo   To stop:
    echo     Double-click: stop.bat
    if defined HAS_DIAGNOSE (
        echo   To diagnose:
        echo     Command line: venv\Scripts\python.exe utils\diagnose.py
    )
    echo.
    echo   Browser: http://127.0.0.1:8501
    echo ============================================================
    echo.
    (
        echo === WWTP AI System - Install Summary ===
        echo Date: !date! !time!
        echo Result: SUCCESS
        echo Mode: !MODE!
        echo Python: !PYTHON_EXE!
        echo Venv: !PROJECT_ROOT!venv
        echo.
        echo All modules verified.
        echo Browser: http://127.0.0.1:8501
    ) > install.log 2>&1
    echo [Info] Install log saved to install.log
    if not defined SILENT pause
    exit /b 0
)

:: ============================================================
:: PIP FALLBACK (uv not available, offline mode)
:: ============================================================
:PipFallback
echo.
echo [Info] Running in pip-only mode...

:: [Fix] MODE may not be set yet (network detection was skipped).
:: Determine MODE from FORCE_MODE override or by probing connectivity.
if not defined MODE (
    if "!FORCE_MODE!"=="OFFLINE" (
        set "MODE=OFFLINE"
    ) else if "!FORCE_MODE!"=="ONLINE" (
        set "MODE=ONLINE"
    ) else (
        :: 3-tier connectivity probe (ping → curl → PowerShell)
        set "MODE=OFFLINE"
        ping -n 1 -w 2000 pypi.org >nul 2>&1 && set "MODE=ONLINE"
        if "!MODE!"=="OFFLINE" curl -s --connect-timeout 5 https://pypi.org >nul 2>&1 && set "MODE=ONLINE"
        if "!MODE!"=="OFFLINE" powershell -NoProfile -Command "try {(Invoke-WebRequest https://pypi.org -TimeoutSec 5).StatusCode; exit 0} catch {exit 1}" >nul 2>&1 && set "MODE=ONLINE"
    )
    echo [Info] Mode: !MODE!
)

:: [Fix] Validate offline packages if MODE ended up as OFFLINE
if "!MODE!"=="OFFLINE" (
    set "HAS_OFFLINE=0"
    if exist "offline_packages\" (
        dir /b "offline_packages\torch-*.whl" >nul 2>&1 && dir /b "offline_packages\numpy-*.whl" >nul 2>&1 && set "HAS_OFFLINE=1"
    )
    if "!HAS_OFFLINE!"=="0" (
        echo.
        echo ============================================================
        echo [FATAL] Offline mode requires offline_packages\ directory
        echo with torch and numpy .whl files.
        echo ============================================================
        if not defined SILENT pause
        exit /b 1
    )
    echo [OK] Offline packages verified
)

:: Quick Python scan — in offline mode, only accept CP310/CP314
:: (offline packages don't have wheels for 3.11/3.12/3.13)
set "PYTHON_EXE="
for %%d in (
    "!LOCALAPPDATA!\Programs\Python\Python310"
    "!LOCALAPPDATA!\Programs\Python\Python314"
    "!LOCALAPPDATA!\Programs\Python\Python313"
    "!LOCALAPPDATA!\Programs\Python\Python312"
    "!LOCALAPPDATA!\Programs\Python\Python311"
    "C:\Program Files\Python310"
    "C:\Program Files\Python314"
    "C:\Python310" "C:\Python314"
) do (
    if exist "%%~d\python.exe" (
        if "!MODE!"=="OFFLINE" (
            :: Verify actual CP version matches offline packages
            set "FB_VER="
            for /f "delims=" %%v in ('"%%~d\python.exe" -c "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))" 2^>nul') do set "FB_VER=%%v"
            if not "!FB_VER!"=="310" if not "!FB_VER!"=="314" (
                echo [Info] Skipping Python CP!FB_VER! at %%~d\python.exe ^(offline packages not available^)
            ) else (
                set "PYTHON_EXE=%%~d\python.exe"
                echo [OK] Found Python CP!FB_VER! (offline-compatible^): !PYTHON_EXE!
                goto :FallbackVenv
            )
        ) else (
            set "PYTHON_EXE=%%~d\python.exe"
            echo [OK] Found Python: !PYTHON_EXE!
            goto :FallbackVenv
        )
    )
)
for /f "delims=" %%p in ('where python.exe 2^>nul') do (
    if "!MODE!"=="OFFLINE" (
        set "FB_PATH_VER="
        for /f "delims=" %%v in ('"%%p" -c "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))" 2^>nul') do set "FB_PATH_VER=%%v"
        if "!FB_PATH_VER!"=="310" if exist "offline_packages\numpy*cp310*.whl" if exist "offline_packages\torch*cp310*.whl" (
            set "PYTHON_EXE=%%p" & goto :FallbackVenv
        )
        if "!FB_PATH_VER!"=="314" if exist "offline_packages\numpy*cp314*.whl" if exist "offline_packages\torch*cp314*.whl" (
            set "PYTHON_EXE=%%p" & goto :FallbackVenv
        )
        echo [Info] Skipping Python CP!FB_PATH_VER! in PATH ^(offline packages not available^)
    ) else (
        "%%p" --version >nul 2>&1 && set "PYTHON_EXE=%%p" && goto :FallbackVenv
    )
)
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
    goto :FallbackVenv
)
echo [FATAL] No Python found. Install Python 3.10+ and retry.
if not defined SILENT pause
exit /b 1

:FallbackVenv
if exist "venv\Scripts\python.exe" goto :FallbackActivate
echo [Info] Creating venv with pip...
"!PYTHON_EXE!" -m venv venv
if !errorlevel! neq 0 (
    echo [Warn] Standard venv failed. Retrying with --without-pip...
    "!PYTHON_EXE!" -m venv --without-pip venv
    if exist "venv\Scripts\python.exe" (
        :: Bootstrap pip into the --without-pip venv
        echo [Info] Bootstrapping pip into venv...
        "venv\Scripts\python.exe" -m ensurepip --default-pip >nul 2>&1
    )
)

:FallbackActivate
call venv\Scripts\activate.bat
if not defined VIRTUAL_ENV (
    echo [FATAL] Cannot activate venv.
    if not defined SILENT pause
    exit /b 1
)

:: Install
if "!MODE!"=="OFFLINE" (
    pip install --no-index --find-links=offline_packages -r requirements.txt
) else (
    pip install -r requirements.txt
)
if !errorlevel! neq 0 (
    echo [FATAL] pip install failed.
    if not defined SILENT pause
    exit /b 1
)
echo [OK] Dependencies installed (pip fallback^)
goto :RunVerification
