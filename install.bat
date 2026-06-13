@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI System - Smart Installer

echo.
echo ============================================================
echo      WWTP AI - Smart Installer (Online ^& Offline)
echo ============================================================
echo.
echo This script will set up the complete Python environment
echo and install all dependencies automatically.
echo.

:: ============================================================
:: 1. Detect working directory
:: ============================================================
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0"
echo [Info] Project root: !PROJECT_ROOT!

:: ============================================================
:: 2. Detect network connectivity
:: ============================================================
echo [1/7] Checking network...
set "HAS_NET=0"
ping -n 1 -w 3000 pypi.org >nul 2>&1 && set "HAS_NET=1"
ping -n 2 -w 2000 pypi.org >nul 2>&1 && set "HAS_NET=1"

if "!HAS_NET!"=="1" (
    echo [OK] Network available - Online install mode
    set "MODE=ONLINE"
) else (
    echo [!]  No network - Offline install mode
    set "MODE=OFFLINE"
)

:: Check offline packages
set "HAS_OFFLINE=0"
if exist "offline_packages\" (
    dir /b "offline_packages\torch-*.whl" >nul 2>&1 && set "HAS_OFFLINE=1"
    dir /b "offline_packages\numpy-*.whl" >nul 2>&1 || set "HAS_OFFLINE=0"
)

if "!MODE!"=="OFFLINE" (
    if "!HAS_OFFLINE!"=="0" (
        echo.
        echo ============================================================
        echo [FATAL] Offline mode requires offline_packages\ directory
        echo with torch and numpy .whl files.
        echo.
        echo Please either:
        echo   1. Connect to the internet and re-run this script
        echo   2. Copy the offline_packages\ folder from the original
        echo      distribution to this directory
        echo ============================================================
        pause
        exit /b 1
    )
    echo [OK] Offline packages verified
)

:: ============================================================
:: 3. Find or install Python 3.10+
:: ============================================================
echo.
echo [2/7] Locating Python 3.10+...

set "PYTHON_EXE="

:: 3a. Check common install locations first (most reliable, no execution)
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python314"
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "C:\Program Files\Python314"
    "C:\Program Files\Python313"
    "C:\Program Files\Python312"
    "C:\Program Files\Python311"
    "C:\Program Files\Python310"
    "C:\Python314" "C:\Python313" "C:\Python312" "C:\Python311" "C:\Python310"
) do (
    if exist "%%~d\python.exe" (
        set "PYTHON_EXE=%%~d\python.exe"
        echo [OK] Found Python at: !PYTHON_EXE!
        goto :FoundPython
    )
)

:: 3b. Check PATH (only if not found in known locations)
for %%c in (python.exe python3.exe) do (
    for /f "delims=" %%p in ('where %%c 2^>nul') do (
        if exist "%%p" (
            REM Verify with --version (filters out Windows Store stubs)
            "%%p" --version >nul 2>&1
            if !errorlevel! equ 0 (
                echo [OK] Found Python in PATH: %%p
                set "PYTHON_EXE=%%p"
                goto :FoundPython
            )
        )
    )
)

:: 3c. Check existing venv (only as last resort — must have activate.bat too)
if exist "venv\Scripts\python.exe" (
    if exist "venv\Scripts\activate.bat" (
        set "PYTHON_EXE=venv\Scripts\python.exe"
        echo [OK] Using existing venv Python
        goto :CheckVenv
    ) else (
        echo [Warn] Broken venv found (missing activate.bat). Will recreate.
        rmdir /s /q "venv" >nul 2>&1
    )
)

:: 3d. Try offline Python installer
if "!HAS_OFFLINE!"=="1" (
    if exist "offline_packages\python-3.10.11-amd64.exe" (
        echo [!]  No Python found. Installing Python 3.10.11 from offline package...
        echo     (This may take a few minutes)
        "offline_packages\python-3.10.11-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=C:\Program Files\Python310\python.exe"
            echo [OK] Python installed successfully
            goto :FoundPython
        )
    )
)

:: 3e. Nothing worked
echo.
echo ============================================================
echo [FATAL] Python 3.10+ not found!
echo.
echo Please either:
echo   1. Install Python 3.10+ from https://python.org
echo      (check "Add Python to PATH" during installation)
echo   2. Place python-3.10.11-amd64.exe in offline_packages\
echo      and re-run this script
echo ============================================================
pause
exit /b 1

:FoundPython
echo [OK] Python: !PYTHON_EXE!

:: Extract CP version tag (e.g. 310, 314) for offline package matching
set "PY_VER_FULL=310"
for /f "delims=" %%v in ('"!PYTHON_EXE!" -c "import sys; sys.stdout.write(str(sys.version_info.major)+str(sys.version_info.minor))" 2^>nul') do set "PY_VER_FULL=%%v"
if "!PY_VER_FULL!"=="" set "PY_VER_FULL=310"
echo   Python CP tag: !PY_VER_FULL!

:: ============================================================
:: 4. Create virtual environment
:: ============================================================
:CheckVenv
echo.
echo [3/7] Setting up virtual environment...

:: [Fix] Clean up any broken leftover venv from previous failed attempts
if exist "venv" (
    if not exist "venv\Scripts\activate.bat" (
        echo [Info] Incomplete venv found, cleaning up...
        rmdir /s /q "venv" >nul 2>&1
        if exist "venv" (
            set "STALE=venv_stale_!RANDOM!"
            ren "venv" "!STALE!" >nul 2>&1
            rmdir /s /q "!STALE!" >nul 2>&1
        )
    ) else (
        echo [OK] Existing venv found
        set /p "RECREATE=Recreate venv? (y/N, default=N): "
        if /i "!RECREATE!"=="Y" (
            echo [Info] Removing old venv...
            rmdir /s /q "venv" >nul 2>&1
            if exist "venv" (
                set "STALE=venv_stale_!RANDOM!"
                ren "venv" "!STALE!" >nul 2>&1
                rmdir /s /q "!STALE!" >nul 2>&1
            )
            goto :CreateVenv
        )
        echo [OK] Reusing existing venv
        goto :ActivateVenv
    )
)

:CreateVenv
echo [Info] Creating virtual environment (this may take 1-2 minutes)...

:: [Fix] Always use --copies on Windows: avoids symlink permission errors
:: that occur with Chinese usernames, OneDrive-synced folders, or restricted paths.
:: Copies python.exe and all DLLs instead of creating symlinks.
"!PYTHON_EXE!" -m venv --copies venv
if !errorlevel! equ 0 goto :ActivateVenv

:: [Fix] If --copies failed, clean up and retry with --without-pip
echo [Warn] --copies failed. Retrying with --without-pip...
rmdir /s /q "venv" >nul 2>&1
"!PYTHON_EXE!" -m venv --without-pip --copies venv
if !errorlevel! equ 0 goto :ActivateVenv

:: Both failed — give actionable diagnosis
echo.
echo ============================================================
echo [FATAL] Cannot create virtual environment (Permission denied)
echo.
echo Likely causes and fixes:
echo.
echo   1. Antivirus or Windows Defender is blocking Python.
echo      >> Add this folder as an exclusion in Windows Security
echo         (Virus ^& threat protection ^> Manage settings ^> Exclusions)
echo.
echo   2. Folder is on a network drive, USB stick, or OneDrive folder.
echo      >> Move WWTP_AI_System to a local drive, e.g. D:\WWTP_AI_System
echo.
echo   3. Insufficient disk space or disk is write-protected.
echo.
echo   4. The username contains characters that Python cannot handle
echo      in file paths. This is a known Python bug on Windows.
echo      >> Create a local user account with ASCII-only name
echo         and run install.bat from that account.
echo ============================================================
pause
exit /b 1

:ActivateVenv
call venv\Scripts\activate.bat
if not defined VIRTUAL_ENV (
    echo [FATAL] Failed to activate venv. Is Python installed correctly?
    pause
    exit /b 1
)
echo [OK] Virtual environment active: !VIRTUAL_ENV!

:: ============================================================
:: [Fix] Force UTF-8 mode for all Python subprocesses / pip
:: This prevents GBK decode errors on Chinese Windows when reading
:: requirements.txt or any other UTF-8 file.
:: ============================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: ============================================================
:: 5. Install pip + upgrade
:: ============================================================
echo.
echo [4/7] Preparing pip...

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [Info] Bootstrapping pip...
    python -m ensurepip --default-pip >nul 2>&1
)

:: [Fix] Flatten nested if/else to avoid CMD parser confusion
if "!MODE!"=="ONLINE" goto :PipOnline
goto :PipOffline

:PipOnline
echo [Info] Upgrading pip (online)...
python -m pip install --upgrade pip >nul 2>&1
goto :PipDone

:PipOffline
echo [Info] Installing pip from offline packages...
if exist "offline_packages\pip-*.whl" python -m pip install --no-index --find-links=offline_packages pip >nul 2>&1
if exist "offline_packages\setuptools-*.whl" python -m pip install --no-index --find-links=offline_packages setuptools >nul 2>&1
goto :PipDone

:PipDone
echo [OK] Pip ready

:: ============================================================
:: 6. Install dependencies
:: ============================================================
echo.
echo [5/7] Installing dependencies...

if not exist "requirements.txt" (
    echo [FATAL] requirements.txt not found!
    pause
    exit /b 1
)

:: --- Re-detect Python version from venv (may differ from system) ---
set "PY_VER_FULL=310"
for /f "delims=" %%v in ('venv\Scripts\python.exe -c "import sys; sys.stdout.write(str(sys.version_info.major)+str(sys.version_info.minor))" 2^>nul') do set "PY_VER_FULL=%%v"
if "!PY_VER_FULL!"=="" set "PY_VER_FULL=310"

echo   Python: CP!PY_VER_FULL!
echo   Mode:   !MODE!
echo.

:: --- Online mode: always works, just confirm ---
if "!MODE!"=="ONLINE" (
    if not "!PY_VER_FULL!"=="310" (
        echo   [Info] Python CP!PY_VER_FULL! detected. Offline packages only
        echo          cover CP310 and CP314. Online install will download
        echo          the correct versions from PyPI automatically.
        echo.
    )
    goto :DepOnline
)

:: --- Offline mode: check if wheels exist for this Python version ---
set "OFFLINE_OK=0"
dir /b "offline_packages\numpy*cp!PY_VER_FULL!*.whl" >nul 2>&1 && set "OFFLINE_OK=1"

if "!OFFLINE_OK!"=="0" (
    echo   ============================================================
    echo   [FATAL] This Python version is NOT supported for offline install.
    echo.
    echo   Your Python:    CP!PY_VER_FULL!
    echo   Offline covers: CP310 ^(Python 3.10^) and CP314 ^(Python 3.14^)
    echo.
    echo   Available numpy wheels in offline_packages\:
    dir /b offline_packages\numpy*.whl 2>nul
    echo.
    echo   Solutions:
    echo   1. Connect to internet and re-run install.bat
    echo      ^(online mode works with any Python 3.10-3.14^)
    echo   2. Install Python 3.10.11 from offline_packages\:
    echo      offline_packages\python-3.10.11-amd64.exe
    echo   3. Manually add matching cp!PY_VER_FULL! .whl files
    echo      to offline_packages\ and retry
    echo   ============================================================
    pause
    exit /b 1
)

:DepOffline
echo [Info] Installing from offline_packages\...
echo     (This may take 3-8 minutes)
echo.

pip install --no-index --find-links=offline_packages -r requirements.txt
if errorlevel 1 (
    echo.
    echo ============================================================
    echo [FATAL] Offline installation failed.
    echo.
    echo   Troubleshooting:
    echo   1. Check if Python is 64-bit (32-bit not supported^)
    echo   2. Run: venv\Scripts\python.exe scripts\diagnose.py
    echo   3. Try online mode: connect to internet, re-run install.bat
    echo ============================================================
    pause
    exit /b 1
)
goto :DepDone

:DepOnline
echo [Info] Installing from PyPI (online)...
echo     (This may take 5-15 minutes depending on network speed)
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [!]  Some packages failed from PyPI.
    echo     Attempting to fill missing from offline_packages...
    pip install --no-index --find-links=offline_packages -r requirements.txt 2>nul
)
goto :DepDone

:DepDone
echo.
echo [OK] Dependencies installed

:: ============================================================
:: 6. Install PyTorch (if not already covered by requirements)
:: ============================================================
echo.
echo [6/7] Verifying PyTorch installation...

python -c "import torch" 2>nul
if errorlevel 1 (
    echo [!]  PyTorch not found, installing separately...
    if "!MODE!"=="ONLINE" (
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    ) else (
        pip install --no-index --find-links=offline_packages torch
    )
    if errorlevel 1 (
        echo [Warning] PyTorch install failed. TFT model will not work.
    ) else (
        echo [OK] PyTorch installed
    )
) else (
    echo [OK] PyTorch already installed
)

:: ============================================================
:: 7. Create runtime directories + verify
:: ============================================================
echo.
echo [7/7] Preparing directories and verifying installation...

for %%d in (logs outputs artifacts "models\lgbm") do (
    if not exist "%%d" mkdir "%%d" >nul 2>&1
)

echo.
echo   Verifying core modules...
python -c "import numpy, pandas, scipy, sklearn, lightgbm, onnxruntime, onnx, onnxscript, yaml, ruamel.yaml, pydantic, loguru, joblib, matplotlib, streamlit, gymnasium" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Some core modules failed - run installer again
) else (
    echo [OK]   All core modules verified
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [WARN] PyTorch missing - TFT training disabled
) else (
    echo [OK]   PyTorch
)

python -c "import stable_baselines3, psutil, numba" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Some optional modules missing
) else (
    echo [OK]   Optional modules (SB3, psutil, numba)
)

:: ============================================================
:: Done
:: ============================================================
echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo   Mode: !MODE!
echo   Python: !PYTHON_EXE!
echo   Venv: !PROJECT_ROOT!venv
echo.
echo   To launch:
echo     Double-click: start.bat  (starts server silently, opens browser)
echo   To stop:
echo     Double-click: stop.bat   (stops background server)
echo   To diagnose issues:
echo     Command line: venv\Scripts\python.exe scripts\diagnose.py
echo.
echo   Browser: http://127.0.0.1:8501
echo.
echo   To train models (use the Web UI for convenience):
echo     - Start the app (start.bat), then go to page 2
echo     - Or command line: venv\Scripts\python.exe train_tft.py
echo ============================================================
echo.
pause
exit /b 0
