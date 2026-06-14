@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Build EXE

echo ========================================================
echo    WWTP AI Packager - Online ^& Offline Build
echo ========================================================
echo.

:: ==========================================
:: 0. Detect uv + silent mode
:: ==========================================
set "HAS_UV="
where uv >nul 2>&1 && set "HAS_UV=1"
if defined HAS_UV (for /f "tokens=2" %%v in ('uv --version 2^>nul') do echo [Info] uv %%v available) else echo [Info] uv not found, using pip

set "SILENT="
if /i "%~1"=="/silent" set "SILENT=1"

:: ==========================================
:: 1. Ensure we have a working Python venv
::    NEVER install to system Python (no admin needed)
:: ==========================================
pushd "%~dp0"
for %%i in ("%~dp0.") do set "PROJECT_ROOT=%%~sfi"
if "!PROJECT_ROOT:~-1!"=="\" set "PROJECT_ROOT=!PROJECT_ROOT:~0,-1%"

if not exist "launcher.py" (
    echo [FATAL] launcher.py not found! Run from project root.
    goto :ErrorExit
)

:: Prefer the project venv (created by install.bat)
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
    echo [OK] Using project venv Python
    goto :VenvReady
)

:: Fallback: any Python in PATH — create a temp build venv
set "PYTHON_EXE=python"
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo [FATAL] No Python found. Run install.bat first.
    goto :ErrorExit
)
echo [OK] Python: !PYTHON_EXE!
echo [Info] Creating build virtual environment...
"!PYTHON_EXE!" -m venv venv_build
if exist "venv_build\Scripts\python.exe" (
    set "PYTHON_EXE=venv_build\Scripts\python.exe"
    echo [OK] Build venv created
) else (
    echo [FATAL] Cannot create build venv.
    goto :ErrorExit
)

:VenvReady
:: Activate the venv so uv/pip install into it (not system Python)
call "!PYTHON_EXE!\..\activate.bat" 2>nul
set "PIP_CMD=!PYTHON_EXE! -m pip"

:: ==========================================
:: 2. Network detection
:: ==========================================
echo.
echo [Check] Testing network...
set "HAS_NETWORK=0"
ping -n 1 -w 2000 pypi.org >nul 2>&1 && set "HAS_NETWORK=1"
if "!HAS_NETWORK!"=="0" ping -n 1 -w 2000 github.com >nul 2>&1 && set "HAS_NETWORK=1"

if "!HAS_NETWORK!"=="1" (
    echo [OK] Network available - online mode
    set "INSTALL_MODE=online"
) else (
    echo [!]  No network - offline mode
    set "INSTALL_MODE=offline"
)

:: ==========================================
:: 3. Check offline packages
:: ==========================================
set "OFFLINE_DIR=offline_packages"
set "HAS_OFFLINE=0"
if exist "!OFFLINE_DIR!\" (
    dir /b "!OFFLINE_DIR!\pyinstaller-*.whl" >nul 2>&1 && set "HAS_OFFLINE=1"
)
if "!INSTALL_MODE!"=="offline" (
    if "!HAS_OFFLINE!"=="0" (
        echo [FATAL] No network AND no PyInstaller wheel in offline_packages\
        goto :ErrorExit
    )
    echo [OK] Found offline PyInstaller package
)

:: ==========================================
:: 4. Install PyInstaller into venv
:: ==========================================
echo.
echo [1/5] Installing PyInstaller...

if "!INSTALL_MODE!"=="online" (
    echo   Online install...
    if defined HAS_UV (
        uv pip install pyinstaller 2>&1
    ) else (
        !PIP_CMD! install pyinstaller 2>&1
    )
    if !errorlevel! neq 0 (
        echo [!]  Online install failed, trying offline fallback...
        if "!HAS_OFFLINE!"=="1" (goto :InstallOffline)
        echo [FATAL] PyInstaller install failed.
        goto :ErrorExit
    )
    goto :PyInstallerOK
)

:InstallOffline
echo   Installing from offline_packages\...
if defined HAS_UV (
    uv pip install --no-index --find-links=!OFFLINE_DIR! pyinstaller 2>&1
) else (
    !PIP_CMD! install --no-index --find-links=!OFFLINE_DIR! pyinstaller 2>&1
)
if !errorlevel! neq 0 (
    echo [FATAL] PyInstaller offline install failed.
    echo   Ensure offline_packages\ has pyinstaller-*.whl AND its dependencies.
    goto :ErrorExit
)

:PyInstallerOK
echo [OK] PyInstaller installed

:: Verify
"!PYTHON_EXE!" -c "import PyInstaller" >nul 2>&1
if !errorlevel! neq 0 (
    echo [FATAL] PyInstaller import test failed!
    goto :ErrorExit
)
echo [OK] PyInstaller import verified

:: ==========================================
:: 5. Build the EXE
:: ==========================================
echo.
echo [2/5] Building EXE (this may take 1-3 minutes)...
echo --------------------------------------------------------

if exist "build" rmdir /s /q "build" >nul 2>&1
if exist "dist"  rmdir /s /q "dist"  >nul 2>&1
if exist "*.spec" del /f /q "*.spec" >nul 2>&1

set "HIDDEN_IMPORTS=--hidden-import=streamlit --hidden-import=torch --hidden-import=onnxruntime --hidden-import=lightgbm --hidden-import=sklearn --hidden-import=scipy --hidden-import=pandas --hidden-import=numpy --hidden-import=yaml --hidden-import=ruamel.yaml --hidden-import=pydantic --hidden-import=loguru --hidden-import=matplotlib --hidden-import=joblib --hidden-import=config.manager --hidden-import=pipeline.data --hidden-import=pipeline.inference --hidden-import=pipeline.gateway --hidden-import=pipeline.adapter --hidden-import=models.lgbm_features --hidden-import=models.lgbm --hidden-import=models.asm1.ode --hidden-import=models.asm1.env --hidden-import=models.tft --hidden-import=models.asm1.calibration --hidden-import=stable_baselines3"

set "ADD_DATA="
if exist "config.yaml" set "ADD_DATA=!ADD_DATA! --add-data=config.yaml;."
if exist "ui"        set "ADD_DATA=!ADD_DATA! --add-data=ui;ui"
if exist "config"    set "ADD_DATA=!ADD_DATA! --add-data=config;config"
if exist "models"    set "ADD_DATA=!ADD_DATA! --add-data=models;models"
if exist "pipeline"  set "ADD_DATA=!ADD_DATA! --add-data=pipeline;pipeline"
if exist "training"  set "ADD_DATA=!ADD_DATA! --add-data=training;training"
if exist "utils"     set "ADD_DATA=!ADD_DATA! --add-data=utils;utils"
if exist "data"      set "ADD_DATA=!ADD_DATA! --add-data=data;data"
if exist "artifacts" set "ADD_DATA=!ADD_DATA! --add-data=artifacts;artifacts"

pyinstaller --onefile --name="WWTP_AI" --clean --noconfirm !HIDDEN_IMPORTS! !ADD_DATA! --exclude-module=tkinter launcher.py

if !errorlevel! neq 0 (
    echo --------------------------------------------------------
    echo [Error] PyInstaller build failed!
    goto :ErrorExit
)
echo --------------------------------------------------------
echo [OK] Build completed

:: ==========================================
:: 6. Verify and move
:: ==========================================
echo.
echo [3/5] Verifying output...
set "EXE_NAME=WWTP_AI.exe"

if exist "dist\!EXE_NAME!" (
    move /y "dist\!EXE_NAME!" "%~dp0!EXE_NAME!" >nul 2>&1
    echo [OK] EXE: %~dp0!EXE_NAME!
) else (
    echo [Error] EXE not found in dist\
    dir /b dist\ 2>nul
    goto :ErrorExit
)

:: ==========================================
:: 7. Update offline packages (online only)
:: ==========================================
echo.
echo [4/5] Updating offline packages...
if "!INSTALL_MODE!"=="online" (
    if not exist "!OFFLINE_DIR!" mkdir "!OFFLINE_DIR!"
    if defined HAS_UV (
        uv pip install pyinstaller >nul 2>&1 & REM already installed
        !PIP_CMD! download -d "!OFFLINE_DIR!" pyinstaller setuptools altair 2>&1
    ) else (
        !PIP_CMD! download -d "!OFFLINE_DIR!" pyinstaller setuptools altair 2>&1
    )
    if !errorlevel! neq 0 (
        echo [!]  Warning: Failed to download some packages (non-fatal)
    ) else (
        echo [OK] Offline packages updated
    )
) else (
    echo   Skipped (offline mode)
)

:: ==========================================
:: 8. Cleanup
:: ==========================================
echo.
echo [5/5] Cleanup...
if exist "build"     rmdir /s /q "build"     >nul 2>&1
if exist "dist"      rmdir /s /q "dist"      >nul 2>&1
if exist "*.spec"    del /f /q "*.spec"      >nul 2>&1
if exist "venv_build" rmdir /s /q "venv_build" >nul 2>&1
echo [OK] Cleanup complete

:: ==========================================
:: 9. Done
:: ==========================================
echo.
echo ========================================================
echo [SUCCESS] Build complete!
echo   EXE: %~dp0!EXE_NAME!
echo ========================================================
echo.
echo Run !EXE_NAME! to launch.
echo.
if not defined SILENT pause
exit /b 0

:ErrorExit
echo.
echo ========================================================
echo [BUILD FAILED] Check errors above.
echo ========================================================
if exist "venv_build" rmdir /s /q "venv_build" >nul 2>&1
if not defined SILENT pause
exit /b 1
