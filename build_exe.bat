@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title WWTP AI - Build EXE (Online/Offline)

echo ========================================================
echo    WWTP AI Packager - Online ^& Offline Build
echo ========================================================
echo.

:: ==========================================
:: 1. Pre-flight checks
:: ==========================================
if not exist "launcher.py" (
    echo [FATAL] launcher.py not found!
    echo Run this script from the project root directory.
    goto :ErrorExit
)
echo [OK] launcher.py found

:: Detect Python
set "PYTHON_EXE=python"
where python >nul 2>&1
if errorlevel 1 (
    :: Try venv
    if exist "venv\Scripts\python.exe" (
        set "PYTHON_EXE=venv\Scripts\python.exe"
        echo [OK] Using venv Python
    ) else (
        echo [FATAL] Python not found! Install Python or activate venv first.
        goto :ErrorExit
    )
) else (
    echo [OK] Python: !PYTHON_EXE!
)

:: ==========================================
:: 2. Detect network connectivity
:: ==========================================
echo.
echo [Check] Testing network connectivity...
set "HAS_NETWORK=0"
ping -n 1 -w 2000 pypi.org >nul 2>&1 && set "HAS_NETWORK=1"
ping -n 2 -w 2000 pypi.org >nul 2>&1 && set "HAS_NETWORK=1"

if "!HAS_NETWORK!"=="1" (
    echo [OK] Network available - online install mode
    set "INSTALL_MODE=online"
) else (
    echo [!] No network - offline install mode
    set "INSTALL_MODE=offline"
)

:: ==========================================
:: 3. Check/find offline packages (for offline mode or fallback)
:: ==========================================
set "OFFLINE_DIR=offline_packages"
set "HAS_OFFLINE=0"
if exist "!OFFLINE_DIR!\" (
    dir /b "!OFFLINE_DIR!\pyinstaller-*.whl" >nul 2>&1 && set "HAS_OFFLINE=1"
)

if "!INSTALL_MODE!"=="offline" (
    if "!HAS_OFFLINE!"=="0" (
        echo [FATAL] No network AND no offline PyInstaller package found!
        echo   Place pyinstaller-*.whl in !OFFLINE_DIR!\ or connect to network.
        goto :ErrorExit
    )
    echo [OK] Found offline PyInstaller package
)

:: ==========================================
:: 4. Install PyInstaller (online or offline)
:: ==========================================
echo.
echo [1/5] Installing PyInstaller...

set "PIP_CMD=!PYTHON_EXE! -m pip"

if "!INSTALL_MODE!"=="online" (
    echo   Trying online install...
    !PIP_CMD! install pyinstaller 2>&1
    if errorlevel 1 (
        echo [!] Online install failed, trying offline fallback...
        if "!HAS_OFFLINE!"=="1" (
            goto :InstallOffline
        ) else (
            echo [FATAL] PyInstaller install failed and no offline package available.
            goto :ErrorExit
        )
    )
    goto :PyInstallerOK
)

:InstallOffline
echo   Installing PyInstaller from offline packages...
!PIP_CMD! install --no-index --find-links=!OFFLINE_DIR! pyinstaller 2>&1
if errorlevel 1 (
    echo [FATAL] PyInstaller offline install failed!
    echo   Check if offline_packages\ contains pyinstaller-*.whl and its dependencies.
    goto :ErrorExit
)

:PyInstallerOK
echo [OK] PyInstaller installed

:: Verify
!PYTHON_EXE! -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
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

:: Clean previous builds
if exist "build" rmdir /s /q "build" >nul 2>&1
if exist "dist" rmdir /s /q "dist" >nul 2>&1
if exist "*.spec" del /f /q "*.spec" >nul 2>&1

:: [Fix] Include hidden imports for all critical packages
set "HIDDEN_IMPORTS="
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=streamlit"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=torch"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=onnxruntime"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=lightgbm"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=sklearn"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=scipy"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=pandas"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=numpy"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=yaml"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=ruamel.yaml"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=pydantic"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=loguru"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=matplotlib"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=joblib"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=config_manager"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=data_pipeline"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=inference"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=predictor_gateway"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=predictor_adapter"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=lgbm_feature_builder"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=lgbm_baseline"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=asm1_ode_solver"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=asm1_ppo_env"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=tft_pure_pytorch"
set "HIDDEN_IMPORTS=!HIDDEN_IMPORTS! --hidden-import=stable_baselines3"

:: [Fix] Include data files
set "ADD_DATA="
if exist "config.yaml" set "ADD_DATA=!ADD_DATA! --add-data=config.yaml;."
if exist "ui"        set "ADD_DATA=!ADD_DATA! --add-data=ui;ui"
if exist "models"    set "ADD_DATA=!ADD_DATA! --add-data=models;models"
if exist "data"      set "ADD_DATA=!ADD_DATA! --add-data=data;data"
if exist "artifacts" set "ADD_DATA=!ADD_DATA! --add-data=artifacts;artifacts"

:: Run PyInstaller (show full output for debugging)
pyinstaller ^
    --onefile ^
    --name="WWTP_AI" ^
    --clean ^
    --noconfirm ^
    !HIDDEN_IMPORTS! ^
    !ADD_DATA! ^
    --exclude-module=tkinter ^
    launcher.py

if errorlevel 1 (
    echo --------------------------------------------------------
    echo [Error] PyInstaller build failed! Scroll up for details.
    goto :ErrorExit
)
echo --------------------------------------------------------
echo [OK] PyInstaller build completed

:: ==========================================
:: 6. Verify and move output
:: ==========================================
echo.
echo [3/5] Verifying output...
set "EXE_NAME=WWTP_AI.exe"

:: PyInstaller output dir
if exist "dist\!EXE_NAME!" (
    echo [OK] EXE found in dist\
    move /y "dist\!EXE_NAME!" "%~dp0!EXE_NAME!" >nul 2>&1
    echo [OK] Moved to: %~dp0!EXE_NAME!
) else (
    :: Try Chinese filename (for backward compat)
    if exist "dist\WWTP_AI.exe" (
        move /y "dist\WWTP_AI.exe" "%~dp0WWTP_AI.exe" >nul 2>&1
        set "EXE_NAME=WWTP_AI.exe"
        echo [OK] Moved to: %~dp0!EXE_NAME!
    ) else (
        echo [Error] EXE not found in dist\ directory!
        echo Contents of dist\:
        dir /b dist\ 2>nul
        goto :ErrorExit
    )
)

:: ==========================================
:: 7. Build offline package if pip is available
:: ==========================================
echo.
echo [4/5] Updating offline packages (if pip available)...
if "!INSTALL_MODE!"=="online" (
    :: Download fresh PyInstaller and its pure-Python deps to offline dir
    if not exist "!OFFLINE_DIR!" mkdir "!OFFLINE_DIR!"
    echo   Downloading PyInstaller + deps for future offline use...
    !PIP_CMD! download -d "!OFFLINE_DIR!" pyinstaller setuptools altair 2>&1
    if errorlevel 1 (
        echo [!] Warning: Failed to download some offline packages (non-fatal)
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
echo [5/5] Cleaning up build artifacts...
if exist "build"  rmdir /s /q "build"  >nul 2>&1
if exist "dist"   rmdir /s /q "dist"   >nul 2>&1
if exist "*.spec" del /f /q "*.spec"   >nul 2>&1
echo [OK] Cleanup complete

:: ==========================================
:: 9. Optional: Create desktop shortcut
:: ==========================================
echo.
set /p "CREATE_SHORTCUT=Create desktop shortcut? (Y/N, default=N): "
if /i "!CREATE_SHORTCUT!"=="Y" (
    powershell -NoProfile -Command ^
        "$WshShell = New-Object -comObject WScript.Shell; ^
        $Shortcut = $WshShell.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'WWTP_AI.lnk')); ^
        $Shortcut.TargetPath = '%~dp0!EXE_NAME!'; ^
        $Shortcut.WorkingDirectory = '%~dp0'; ^
        $Shortcut.Save()"
    echo [OK] Desktop shortcut created
)

:: ==========================================
:: Done
:: ==========================================
echo.
echo ========================================================
echo [SUCCESS] Build complete!
echo   EXE: %~dp0!EXE_NAME!
echo   Mode: !INSTALL_MODE!
echo ========================================================
echo.
echo Run !EXE_NAME! to launch the application.
echo.
pause
exit /b 0

:: ==========================================
:ErrorExit
echo.
echo ========================================================
echo [BUILD FAILED] Check errors above.
echo ========================================================
pause
exit /b 1
