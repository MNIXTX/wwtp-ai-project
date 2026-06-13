#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WWTP AI System - 一键卸载/重置工具
功能：停止进程 → 删除 venv → 清理运行数据 → (可选)删除整个项目
"""

import os
import sys
import shutil
import subprocess
import signal
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
]

# 硬卸载额外清理目录（包含模型、数据、配置）
HARD_CLEAN_DIRS = SOFT_CLEAN_DIRS + [
    "models",
    "data",
    "offline_packages",
]

# 核心代码文件（用于判断是否在项目根目录）
CORE_FILES = ["launcher.py", "config.yaml", "predictor_gateway.py"]


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
    """停止所有与本项目相关的 Python 进程"""
    print_step("Scanning and stopping related processes...")
    project_path_str = str(PROJECT_ROOT).lower()

    if sys.platform == "win32":
        # [Fix] 使用 PowerShell 获取含命令行参数的进程列表
        # WMIC 已废弃且输出格式与代码解析逻辑不兼容
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
                    if len(parts) >= 1 and parts[0].isdigit():
                        pid = parts[0]
                        run_command(f'taskkill /F /PID {pid}')
                        print(f"  [STOP] Process PID={pid} terminated")
    else:
        # Linux/Mac
        success, stdout, _ = run_command(
            f"ps aux | grep python | grep '{PROJECT_ROOT}' | grep -v grep",
            capture_output=True
        )
        if success and stdout.strip():
            for line in stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"  [STOP] Process PID={pid} terminated")
                    except ProcessLookupError:
                        pass

    print("  Process cleanup done")


def delete_venv():
    """删除本地 venv 虚拟环境"""
    print_step(f"Deleting virtual environment: {VENV_DIR}")
    if VENV_DIR.exists():
        try:
            shutil.rmtree(VENV_DIR, onerror=_on_rm_error)
            print(f"  [OK] Virtual environment deleted")
        except Exception as e:
            print(f"  [WARN] Cannot delete {VENV_DIR}: {e}")
    else:
        print(f"  [INFO] {VENV_DIR} not found, skip")


def _on_rm_error(func, path, exc_info):
    """
    shutil.rmtree 错误回调：处理 Windows 文件锁定问题。
    对只读文件先改权限再重试；其他错误静默跳过。
    """
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass  # 跳过无法删除的文件，不中断整体流程


def clean_runtime_data(dirs_to_clean):
    """清理运行时数据"""
    print_step("Cleaning runtime data...")
    for dir_name in dirs_to_clean:
        path = PROJECT_ROOT / dir_name
        if path.exists():
            try:
                shutil.rmtree(path, onerror=_on_rm_error)
                print(f"  [OK] Deleted: {dir_name}/")
            except Exception as e:
                print(f"  [WARN] Delete failed {dir_name}/: {e}")
        else:
            print(f"  [INFO] Not found: {dir_name}/")


def full_uninstall_project():
    """删除除 uninstall.py 以外的所有文件"""
    print_step(f"WARNING: Purging project root: {PROJECT_ROOT}")
    print("   (Keeping uninstall.py for final self-destruct)")

    for item in PROJECT_ROOT.iterdir():
        if item.name == "uninstall.py":
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, onerror=_on_rm_error)
            else:
                item.unlink()
            print(f"    [DEL] {item.name}")
        except Exception as e:
            print(f"    [WARN] Delete failed {item.name}: {e}")


def remove_env_variables():
    """移除环境变量"""
    print_step("Checking environment variables...")
    vars_to_remove = ["SMART_DOSING_HOME", "PYTHONPATH"]
    if sys.platform == "win32":
        for var in vars_to_remove:
            run_command(f'setx {var} ""', shell=True)
            print(f"  [OK] Cleared {var} (set to empty)")
    else:
        print("  [INFO] Linux/Mac: manually check .bashrc or .zshrc for export variables")


# ==================== 主程序 ====================
def main():
    safety_check()
    print_header("WWTP AI System - Uninstall / Reset Wizard")

    print(f"\nProject path: {PROJECT_ROOT}")
    print(f"Venv path:    {VENV_DIR}")

    print("\nSelect operation mode:")
    print("  1. Soft reset: clean logs/temp/artifacts + delete venv (keep code, models, data)")
    print("  2. Full reset: soft reset + delete models, data, offline packages (keep only code)")
    print("  3. [DANGER] Hard purge: delete EVERYTHING including source code")

    choice = input("\nEnter option (1/2/3): ").strip()

    if choice not in ('1', '2', '3'):
        print("Invalid input, exiting.")
        return

    if not confirm("\nThis operation is irreversible! Confirm?"):
        print("Cancelled.")
        return

    # [Fix] 每步独立 try，一步失败不中断后续删除
    errors = []

    try:
        stop_running_processes()
    except Exception as e:
        errors.append(f"Stop processes: {e}")

    try:
        delete_venv()
    except Exception as e:
        errors.append(f"Delete venv: {e}")

    try:
        if choice == '1':
            clean_runtime_data(SOFT_CLEAN_DIRS)
            print("\nSoft reset complete!")
            print("Code, models, data, and offline packages are preserved.")
            print("Run install.bat to rebuild the environment.")

        elif choice == '2':
            clean_runtime_data(HARD_CLEAN_DIRS)
            print("\nFull reset complete!")
            print("Code is preserved. Models, data, and offline packages are deleted.")
            print("Run install.bat to rebuild the environment, then retrain models.")

        elif choice == '3':
            if confirm("\nFINAL WARNING: This will delete ALL source code. Are you SURE?"):
                clean_runtime_data(HARD_CLEAN_DIRS)
                full_uninstall_project()
                remove_env_variables()
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

    # [Fix] 暂停等待用户查看结果，防止窗口瞬间关闭
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
