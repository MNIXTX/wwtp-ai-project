#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WWTP AI 系统诊断工具 — 新电脑/部署环境一键排查
================================================
用法: python utils/diagnose.py
     或在项目根目录: python utils/diagnose.py
"""

import sys
import os
from pathlib import Path

# --- 修复 Windows GBK 编码问题 ---
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 设置根目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI 颜色
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

results = {"pass": 0, "fail": 0, "warn": 0}


def ok(msg):
    results["pass"] += 1
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def fail(msg):
    results["fail"] += 1
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg):
    results["warn"] += 1
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def hdr(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


# ============================================================
def check_python():
    hdr("1. Python 环境")
    v = sys.version_info
    venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    print(f"  Python {v.major}.{v.minor}.{v.micro} @ {sys.executable}")
    if not is_venv:
        print(f"\n  {RED}{BOLD}>>>  " + "=" * 60 + f"{RESET}")
        print(f"  {RED}{BOLD}>>>  警告：你正在使用系统 Python 运行诊断！{RESET}")
        print(f"  {RED}{BOLD}>>>  所有依赖安装在 venv 中，系统 Python 无法检测到。{RESET}")
        print(f"  {RED}{BOLD}>>>  请使用以下命令重新运行：{RESET}")
        if venv_python.exists():
            print(f"  {GREEN}>>>    ..\\venv\\Scripts\\python.exe diagnose.py{RESET}")
        else:
            print(f"  {RED}>>>    venv 不存在！请先运行 install.bat{RESET}")
        print(f"  {RED}{BOLD}>>>  " + "=" * 60 + f"{RESET}\n")

    if v.major == 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor} 版本兼容")
    elif v.major == 3 and v.minor >= 8:
        warn(f"Python {v.major}.{v.minor} 可用但建议 >= 3.10")
    else:
        fail(f"Python 版本 {v.major}.{v.minor} 过低，需要 3.10+")
        print("  请安装 Python 3.10+ https://www.python.org/downloads/")
        return False
    return True


def check_dependencies():
    hdr("2. 核心依赖")
    deps = {
        "streamlit": "Streamlit Web 框架",
        "pandas": "数据处理",
        "numpy": "数值计算",
        "lightgbm": "LightGBM 模型",
        "onnxruntime": "ONNX 推理引擎",
        "yaml": "YAML 配置解析",
        "pydantic": "配置校验",
        "loguru": "日志系统",
        "sklearn": "Scikit-learn（归一化）",
        "joblib": "模型持久化",
    }
    for mod, desc in deps.items():
        try:
            __import__(mod)
            ok(f"{mod} — {desc}")
        except ImportError:
            fail(f"{mod} 未安装 — {desc}")
            print(f"  修复: pip install {mod}")
    try:
        __import__("ruamel.yaml")
        ok("ruamel.yaml — 保留注释的YAML读写")
    except ImportError:
        warn("ruamel.yaml 未安装（非必需）")


def check_config():
    hdr("3. 配置文件")
    cfg = PROJECT_ROOT / "config.yaml"
    if cfg.exists():
        ok(f"config.yaml 存在 ({cfg.stat().st_size / 1024:.0f} KB)")
        try:
            import yaml
            with open(cfg, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if 'paths' in data and 'scada_data_csv' in data.get('paths', {}):
                csv_path = data['paths']['scada_data_csv']
                ok(f"paths.scada_data_csv = '{csv_path}'")
            else:
                fail("config.yaml 缺少 paths.scada_data_csv 字段")
        except Exception as e:
            fail(f"config.yaml 解析失败: {e}")
    else:
        fail("config.yaml 不存在！")
        print(f"  预期位置: {cfg}")
        print("  修复: 从备份恢复或运行 python config/manager.py 生成默认配置")


def check_data():
    hdr("4. 数据文件")
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        csvs = list(data_dir.glob("*.csv"))
        if csvs:
            ok(f"data/ 目录存在，{len(csvs)} 个 CSV 文件")
            for f in sorted(csvs, key=lambda x: x.stat().st_size, reverse=True)[:5]:
                size_kb = f.stat().st_size / 1024
                if size_kb < 1:
                    warn(f"  {f.name} — {size_kb:.1f} KB (可能为空)")
                else:
                    ok(f"  {f.name} — {size_kb:.0f} KB")
        else:
            fail("data/ 目录存在但无 CSV 文件")
            print("  请将 SCADA 历史数据 CSV 放入 data/ 目录")
    else:
        fail("data/ 目录不存在")
        data_dir.mkdir(parents=True, exist_ok=True)
        ok("已自动创建 data/ 目录（请放入 CSV 文件）")


def check_models():
    hdr("5. 模型文件（完整性检查）")
    tft = PROJECT_ROOT / "models" / "industrial_tft.onnx"
    lgbm = PROJECT_ROOT / "models" / "lgbm" / "lgbm_wwtp.txt"

    # TFT
    if tft.exists():
        size = tft.stat().st_size
        if size > 1000:
            with open(tft, 'rb') as f:
                magic = f.read(1)
            if magic == b'\x08':
                ok(f"TFT (ONNX): {size / 1024:.0f} KB — 格式合法")
            else:
                fail(f"TFT (ONNX): 文件已损坏 (非法头部: {magic!r})")
                print(f"  修复: 重新训练 TFT 模型 python training/train_tft.py")
        elif size > 0:
            fail(f"TFT (ONNX): 文件异常小 ({size} bytes)，可能损坏")
        else:
            fail(f"TFT (ONNX): 文件为空")
    else:
        warn("TFT (ONNX) 模型文件不存在（将仅使用 LGBM）")

    # LGBM
    if lgbm.exists():
        size = lgbm.stat().st_size
        if size > 100:
            with open(lgbm, 'rb') as f:
                header = f.read(10)
            if header.startswith(b'tree'):
                ok(f"LGBM: {size / 1024:.0f} KB — 格式合法")
            else:
                fail(f"LGBM: 文件已损坏 (非法头部: {header!r})")
                print(f"  修复: 重新训练 LGBM 模型 python training/train_lgbm.py")
        elif size > 0:
            fail(f"LGBM: 文件异常小 ({size} bytes)，可能损坏")
        else:
            fail(f"LGBM: 文件为空")
    else:
        warn("LGBM 模型文件不存在（将仅使用 TFT）")

    # 检查损坏的备份
    corrupted = lgbm.parent / "lgbm_wwtp.txt.corrupted"
    if corrupted.exists():
        warn(f"发现损坏的 LGBM 模型备份: {corrupted}")
        print("  已自动跳过，如需恢复请先验证文件完整性后重命名")


def check_scalers():
    hdr("6. 归一化器文件")
    artifacts = PROJECT_ROOT / "artifacts"
    scaler = artifacts / "scaler.pkl"
    target = artifacts / "target_scaler.pkl"

    for name, path in [("Feature Scaler", scaler), ("Target Scaler", target)]:
        if path.exists():
            size = path.stat().st_size
            if size > 100:
                with open(path, 'rb') as f:
                    magic = f.read(2)
                if magic in (b'\x80\x03', b'\x80\x04', b'\x80\x05'):
                    ok(f"{name}: {size / 1024:.1f} KB — 格式合法")
                else:
                    fail(f"{name}: pickle 格式异常 (魔数: {magic!r})")
            elif size > 0:
                fail(f"{name}: 文件异常小 ({size} bytes)")
            else:
                fail(f"{name}: 文件为空")
        else:
            warn(f"{name} 不存在（将不进行特征归一化）")


def check_disk():
    hdr("7. 磁盘空间")
    try:
        import shutil
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        if free_gb > 10:
            ok(f"可用空间: {free_gb:.1f} GB / {total_gb:.0f} GB")
        elif free_gb > 2:
            warn(f"可用空间较少: {free_gb:.1f} GB（建议 > 10 GB）")
        else:
            fail(f"可用空间严重不足: {free_gb:.1f} GB（需要 > 2 GB）")
    except Exception:
        warn("无法检测磁盘空间（非 Windows 系统？）")


def check_network():
    hdr("8. 网络端口")
    port = 8501
    host = "127.0.0.1"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex((host, port))
        s.close()
        if result == 0:
            warn(f"端口 {port} 已被占用（可能有其他 Streamlit 实例在运行）")
            print(f"  修复: 运行 stop.bat 或手动终止 python.exe 进程")
        else:
            ok(f"端口 {port} 空闲")
    except Exception as e:
        warn(f"端口检测失败: {e}")


def check_quickstart():
    hdr("9. 快速启动测试")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  Streamlit 版本: ", end="")
    try:
        import streamlit
        print(streamlit.__version__)
        ok("Streamlit 导入正常")
    except Exception as e:
        fail(f"Streamlit 导入失败: {e}")
        return

    print(f"\n  启动命令:")
    print(f"  {CYAN}cd {PROJECT_ROOT}{RESET}")
    print(f"  {CYAN}venv\\Scripts\\python.exe -m streamlit run ui/app.py --server.port=8501{RESET}")
    print(f"\n  或在 PowerShell 中:")
    print(f"  {CYAN}.\\venv\\Scripts\\python.exe -m streamlit run ui/app.py --server.headless=true --server.port=8501{RESET}")


# ============================================================
def main():
    print(f"{BOLD}{CYAN}")
    print("+==============================================+")
    print("|    WWTP AI System Diagnostic v1.0            |")
    print("|    Water-treatment AI Platform Health Check  |")
    print("+==============================================+")
    print(RESET)

    checks = [
        check_python,
        check_dependencies,
        check_config,
        check_data,
        check_models,
        check_scalers,
        check_disk,
        check_network,
        check_quickstart,
    ]

    for check in checks:
        try:
            check()
        except Exception as e:
            fail(f"检查过程异常: {e}")

    # 总结
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  诊断完成: {GREEN}{results['pass']} 通过{RESET}  "
          f"{YELLOW}{results['warn']} 警告{RESET}  "
          f"{RED}{results['fail']} 失败{RESET}  "
          f"(共 {total} 项)")
    print(f"{BOLD}{'='*60}{RESET}")

    if results["fail"] > 0:
        print(f"\n{RED}发现 {results['fail']} 个问题需要修复。{RESET}")
        print("常见修复步骤:")
        print("  0. 确保使用 venv Python 运行本诊断:")
        print("       ..\\venv\\Scripts\\python.exe diagnose.py")
        print("  1. 安装缺失依赖: ..\\venv\\Scripts\\python.exe -m pip install -r ..\\requirements.txt")
        print("  2. 重建虚拟环境: 删除 venv 文件夹，重新运行 install.bat")
        print("  3. 恢复配置文件: 从备份复制 config.yaml")
        print("  4. 重新训练模型: ..\\venv\\Scripts\\python.exe ..\\training\\run_pipeline.py")
        print("  5. 清理进程:     运行 ..\\stop.bat")
        return 1
    else:
        print(f"\n{GREEN}所有关键检查通过！可以启动系统。{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
