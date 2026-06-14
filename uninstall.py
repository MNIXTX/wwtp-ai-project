#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WWTP AI System - 一键卸载/重置工具
功能：停止进程 → 删除 venv → 清理运行数据 → (可选)删除整个项目

支持：
  - CLI 交互式运行:  python uninstall.py
  - 非交互模式:      python uninstall.py --yes --soft
  - 作为模块导入:    from uninstall import clean_runtime_data, delete_venv
"""

import os
import sys
import shutil
import stat
import subprocess
import argparse
from pathlib import Path

# ==================== 配置区域 ====================
PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"

# 软重置清理目录（仅临时/缓存类，不伤代码和数据）
SOFT_CLEAN_DIRS = [
    "logs",
    "artifacts",
    "outputs",
    "__pycache__",
    "build",         # PyInstaller build artifacts
    "dist",          # PyInstaller output
]

# 软重置清理的根目录文件（运行时生成的临时文件）
SOFT_CLEAN_FILES = [
    "install.log",           # install.bat 安装日志
    ".server.pid",           # start.bat 进程 PID 文件
    "streamlit_crash.log",   # start.bat 崩溃日志（在 logs/ 下）
    "install_debug.log",     # 调试日志
]

# 硬卸载额外清理目录（包含模型、数据、配置）
HARD_CLEAN_DIRS = SOFT_CLEAN_DIRS + [
    "models",
    "data",
    "offline_packages",
]

# 硬卸载清理的根目录文件
HARD_CLEAN_FILES = SOFT_CLEAN_FILES + [
    "*.spec",                # PyInstaller spec files
]

# 核心代码文件（用于判断是否在项目根目录）
CORE_FILES = ["launcher.py", "config.yaml", "requirements.txt"]


# ==================== 工具函数 ====================
def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(text):
    print(f"\n[→] {text}")


def run_command(cmd, shell=True, capture_output=False):
    """执行系统命令"""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=capture_output,
            text=True, timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def confirm(prompt):
    """获取用户确认"""
    while True:
        choice = input(f"{prompt} (y/N): ").strip().lower()
        if choice in ('y', 'yes'):
            return True
        elif choice in ('n', 'no', ''):
            return False


# ==================== 内置安全删除（无第三方依赖） ====================

def _safe_delete_dir(path: Path) -> tuple:
    """Standalone safe directory deletion using only stdlib.

    Does NOT depend on config_manager, yaml, or any venv-installed packages.
    Works even when the venv is broken or already deleted.

    Returns (deleted_count, locked_count).
    """
    if not path.exists():
        return 0, 0

    deleted, locked = 0, 0

    def _on_rm_error(func, failed_path, exc_info):
        """Handle read-only files by chmod + retry."""
        try:
            os.chmod(failed_path, stat.S_IWRITE)
            func(failed_path)
        except Exception:
            raise

    # Special handling for logs directory: delete files individually
    # to avoid "file in use" errors on Windows.
    if path.name == "logs" and path.is_dir():
        for item in sorted(path.iterdir(), key=lambda x: x.name, reverse=True):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, onerror=_on_rm_error)
                deleted += 1
            except (PermissionError, OSError):
                locked += 1
        try:
            path.rmdir()
            deleted += 1
        except (PermissionError, OSError):
            locked += 1
        return deleted, locked

    # General directory deletion
    try:
        shutil.rmtree(path, onerror=_on_rm_error)
        return 1, 0
    except PermissionError:
        # Fallback: delete file by file
        for item in sorted(path.iterdir(), key=lambda x: x.name, reverse=True):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, onerror=_on_rm_error)
                deleted += 1
            except (PermissionError, OSError):
                locked += 1
        try:
            path.rmdir()
            deleted += 1
        except (PermissionError, OSError):
            locked += 1
        return deleted, locked


def _safe_delete_files(file_patterns: list, quiet=False):
    """Delete individual files matching glob patterns in the project root.

    Unlike _safe_delete_dir, this handles single files (e.g. install.log,
    .server.pid, *.spec) that are generated at the project root level.
    """
    import glob as _glob
    deleted_count = 0
    for pattern in file_patterns:
        for filepath in _glob.glob(str(PROJECT_ROOT / pattern)):
            try:
                os.unlink(filepath)
                deleted_count += 1
                if not quiet:
                    print(f"  [OK] Deleted: {Path(filepath).name}")
            except (PermissionError, OSError) as e:
                if not quiet:
                    print(f"  [WARN] Could not delete {Path(filepath).name}: {e}")
    return deleted_count


# ==================== 卸载步骤 ====================

def safety_check():
    """安全检查：确保在正确的项目目录下"""
    missing_files = [f for f in CORE_FILES if not (PROJECT_ROOT / f).exists()]
    if len(missing_files) >= 1:  # [Fix] 少1个就报错，之前是 >1 太松
        print(f"\n[ERROR] Current directory ({PROJECT_ROOT}) is not the project root!")
        print(f"   Missing files: {missing_files}")
        print("   Place uninstall.py in the WWTP_AI_System root directory.")
        sys.exit(1)


def stop_running_processes():
    """停止所有与本项目相关的 Python 进程（排除自身）"""
    print_step("Scanning and stopping related processes...")
    project_path_str = str(PROJECT_ROOT).lower()
    self_pid = str(os.getpid())

    if sys.platform == "win32":
        # [Fix] 使用 PowerShell 获取含命令行参数的进程列表
        ps_cmd = (
            'powershell -NoProfile -Command '
            '"Get-CimInstance Win32_Process -Filter \\"Name=\'python.exe\'\\" | '
            'Select-Object ProcessId, CommandLine | '
            'ConvertTo-Csv -NoTypeInformation"'
        )
        success, stdout, _ = run_command(ps_cmd, capture_output=True)
        if success and stdout.strip():
            for line in stdout.splitlines():
                if project_path_str in line.lower():
                    # CSV 格式: "12345","python ..."
                    parts = line.replace('"', '').split(',')
                    if len(parts) >= 1 and parts[0].strip().isdigit():
                        pid = parts[0].strip()
                        if pid == self_pid:
                            continue
                        print(f"  [STOP] Terminating PID={pid}...")
                        run_command(f'taskkill /F /PID {pid}')
    else:
        # Linux/Mac
        import signal
        success, stdout, _ = run_command(
            f"pgrep -f 'python.*{PROJECT_ROOT}'",
            capture_output=True
        )
        if success and stdout.strip():
            for pid in stdout.strip().splitlines():
                pid = pid.strip()
                if pid.isdigit() and pid != self_pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"  [STOP] Process PID={pid} terminated")
                    except ProcessLookupError:
                        pass

    print("  Process cleanup done")


def delete_venv(quiet=False):
    """删除本地 venv 虚拟环境（使用 stdlib，无需 venv 依赖）

    On Windows, if the venv is currently activated or has locked files,
    we try multiple strategies: direct deletion → file-by-file → rename-then-delete.
    """
    if not quiet:
        print_step(f"Deleting virtual environment: {VENV_DIR}")
    if not VENV_DIR.exists():
        if not quiet:
            print(f"  [INFO] {VENV_DIR} not found, skip")
        return

    # Strategy 1: direct deletion
    try:
        _safe_delete_dir(VENV_DIR)
        if not VENV_DIR.exists():
            if not quiet:
                print(f"  [OK] Virtual environment deleted")
            return
    except Exception:
        pass

    # Strategy 2: rename-then-delete (handles locked .dll files on Windows)
    if not quiet:
        print(f"  [INFO] venv is locked, trying rename-then-delete...")
    import tempfile, time
    try:
        stale = Path(tempfile.gettempdir()) / f"venv_stale_{int(time.time())}"
        shutil.move(str(VENV_DIR), str(stale))
        _safe_delete_dir(stale)
        if not quiet:
            print(f"  [OK] Virtual environment deleted (rename-then-delete)")
    except Exception as e:
        if not quiet:
            print(f"  [WARN] Cannot fully delete venv: {e}")
            print(f"  [INFO] You may need to manually delete: {VENV_DIR}")
            if sys.platform == "win32":
                print(f"  [INFO] Try: rmdir /s /q \"{VENV_DIR}\" in a new cmd window")


def clean_runtime_data(dirs_to_clean, quiet=False):
    """清理运行时数据（使用 stdlib，处理 Windows 文件锁）"""
    if not quiet:
        print_step("Cleaning runtime data...")
    for dir_name in dirs_to_clean:
        path = PROJECT_ROOT / dir_name
        if not path.exists():
            if not quiet:
                print(f"  [INFO] Not found: {dir_name}/")
            continue
        try:
            deleted, locked = _safe_delete_dir(path)
            if not quiet:
                status = f"  [OK] Deleted: {dir_name}/"
                if locked:
                    status += f" ({locked} files locked, skipped)"
                print(status)
        except Exception as e:
            if not quiet:
                print(f"  [WARN] Delete failed {dir_name}/: {e}")


def full_uninstall_project():
    """删除除 uninstall.py 以外的所有文件"""
    print_step(f"WARNING: Purging project root: {PROJECT_ROOT}")
    print("   (Keeping uninstall.py for final self-destruct)")

    for item in PROJECT_ROOT.iterdir():
        if item.name == "uninstall.py":
            continue
        try:
            if item.is_dir():
                _safe_delete_dir(item)
            else:
                item.unlink()
            print(f"    [DEL] {item.name}")
        except Exception as e:
            print(f"    [WARN] Delete failed {item.name}: {e}")


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(
        description="WWTP AI System - Uninstall / Reset Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python uninstall.py              # Interactive mode
  python uninstall.py --yes --soft # Non-interactive soft reset
  python uninstall.py --yes --hard # Non-interactive full reset
        """,
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--soft", action="store_true", help="Soft reset: logs + artifacts + outputs + venv (keep code/models/data)")
    parser.add_argument("--hard", action="store_true", help="Full reset: soft + models + data + offline_packages")
    parser.add_argument("--purge", action="store_true", help="[DANGER] Delete EVERYTHING including source code")
    parser.add_argument("--choice", "-c", type=str, choices=["1", "2", "3"], help="Operation choice (1=soft, 2=hard, 3=purge)")
    args = parser.parse_args()

    safety_check()

    # Resolve mode
    if args.choice:
        choice = args.choice
    elif args.purge:
        choice = "3"
    elif args.hard:
        choice = "2"
    elif args.soft:
        choice = "1"
    else:
        # Interactive mode
        print_header("WWTP AI System - Uninstall / Reset Wizard")
        print(f"\nProject path: {PROJECT_ROOT}")
        print(f"Venv path:    {VENV_DIR}")
        print("\nSelect operation mode:")
        print("  1. Soft reset: clean logs/temp/artifacts/build + delete venv")
        print("     (preserves: code, models, data, offline packages)")
        print("  2. Full reset: soft reset + delete models, data, offline packages")
        print("     (preserves: source code only, must retrain models)")
        print("  3. [DANGER] Hard purge: delete EVERYTHING including source code")
        print("\n  Note: pip and uv package caches (in %APPDATA%) are outside")
        print("  the project folder and NOT cleaned by this tool.")
        choice = input("\nEnter option (1/2/3): ").strip()
        if choice not in ('1', '2', '3'):
            print("Invalid input, exiting.")
            return
        if not confirm("\nThis operation is irreversible! Confirm?"):
            print("Cancelled.")
            return

    # --yes implies non-interactive + skip confirmations
    verbose = not args.yes
    errors = []

    # Stop processes (always verbose — user should know what's being killed)
    try:
        stop_running_processes()
    except Exception as e:
        errors.append(f"Stop processes: {e}")

    # Delete venv
    try:
        delete_venv(quiet=not verbose)
    except Exception as e:
        errors.append(f"Delete venv: {e}")

    # Clean data
    try:
        if choice == '1':
            clean_runtime_data(SOFT_CLEAN_DIRS, quiet=not verbose)
            _safe_delete_files(SOFT_CLEAN_FILES, quiet=not verbose)
            if verbose:
                print("\nSoft reset complete!")
                print("Code, models, data, and offline packages are preserved.")
                print("Run install.bat to rebuild the environment.")

        elif choice == '2':
            clean_runtime_data(HARD_CLEAN_DIRS, quiet=not verbose)
            _safe_delete_files(HARD_CLEAN_FILES, quiet=not verbose)
            if verbose:
                print("\nFull reset complete!")
                print("Code is preserved. Models, data, and offline packages are deleted.")
                print("Run install.bat to rebuild the environment, then retrain models.")

        elif choice == '3':
            if args.yes or confirm("\nFINAL WARNING: This will delete ALL source code. Are you SURE?"):
                clean_runtime_data(HARD_CLEAN_DIRS, quiet=not verbose)
                _safe_delete_files(HARD_CLEAN_FILES, quiet=not verbose)
                full_uninstall_project()
                if verbose:
                    print("\nProject purged.")
                    print("TIP: Manually delete this uninstall.py script and the empty folder.")
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
    except Exception as e:
        errors.append(f"Cleanup: {e}")

    if errors:
        print("\n[WARN] Some steps had errors:")
        for err in errors:
            print(f"  - {err}")

    if verbose:
        print()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
