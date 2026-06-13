#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管道主入口脚本 (商业交付级 v3.3)
支持 UI Worker 线程调用 (带进度回调) 与 CLI 独立运行
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import os
import shutil
import warnings
import gc
from pathlib import Path
from typing import Callable, Optional

# 屏蔽 Pandas 常见的 Downcasting 警告，保持日志干净
warnings.simplefilter(action='ignore', category=FutureWarning)

# ================= 1. 核心环境与路径防护 (引导阶段) =================
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
    IS_PACKAGED = True
else:
    PROJECT_ROOT = Path(__file__).parent.resolve()
    IS_PACKAGED = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 🚀 【核心修复】工业级虚拟环境检测函数
def check_virtual_environment():
    """
    全面检测当前是否运行在虚拟环境中。
    兼容：标准 venv, 旧版 virtualenv, Conda, Poetry, Pipenv 等。
    """
    # 1. 检测旧版 virtualenv (Python 2 或早期 Python 3)
    if hasattr(sys, 'real_prefix'):
        return True
        
    # 2. 检测标准 venv (PEP 405, Python 3.3+) / Poetry / Pipenv
    # 在标准 venv 中，sys.base_prefix 指向基础 Python，sys.prefix 指向虚拟环境
    if hasattr(sys, 'base_prefix') and (sys.prefix != sys.base_prefix):
        return True
        
    # 3. 检测 Conda 环境
    # Conda 激活时通常会设置 CONDA_PREFIX 环境变量
    if os.environ.get('CONDA_PREFIX'):
        return True
    
    # 4. 兜底检测：路径特征 (针对某些未正确设置 prefix 的魔改 Conda 或特殊 IDE)
    exec_path = sys.executable.lower()
    prefix_path = sys.prefix.lower()
    conda_keywords = ['conda', 'anaconda', 'miniconda']
    
    if any(kw in exec_path for kw in conda_keywords) or any(kw in prefix_path for kw in conda_keywords):
        return True
        
    return False

if not IS_PACKAGED:
    if not check_virtual_environment():
        sys.stderr.write("\033[91m" + "="*60 + "\033[0m\n")
        sys.stderr.write("\033[91m❌ 严重警告: 检测到当前未使用虚拟环境运行!\033[0m\n")
        sys.stderr.write(f"当前解释器: {sys.executable}\n")
        sys.stderr.write("\033[93m💡 请使用以下任一方式运行:\033[0m\n")
        sys.stderr.write("   • PowerShell: .\\venv\\Scripts\\python.exe run_pipeline.py\n")
        sys.stderr.write("   • Conda: conda activate <your_env_name> && python run_pipeline.py\n")
        sys.stderr.write("\033[91m" + "="*60 + "\033[0m\n")
# ============================================================

# ================= 2. 业务模块导入 (带防御) =================
from loguru import logger

logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=False,
    level="INFO"
)

from config_manager import CFG
from data_pipeline import WWTPDataPipeline

# 🚀 psutil 延迟导入与优雅降级
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("⚠️ 未检测到 psutil 库，磁盘空间预检功能将被禁用。(可通过 pip install psutil 开启)")

# ===================================================

def check_disk_space(target_path: Path, min_free_gb: float = 2.0):
    """提前检测磁盘空间"""
    if not HAS_PSUTIL:
        return

    try:
        usage = psutil.disk_usage(str(target_path))
        free_gb = usage.free / (1024**3)
        if free_gb < min_free_gb:
            raise RuntimeError(f"磁盘空间不足！{target_path} 所在磁盘仅剩 {free_gb:.1f}GB，需要至少 {min_free_gb}GB。")
    except RuntimeError:
        raise  # 磁盘空间不足是致命错误，直接上抛
    except Exception as e:
        # psutil 自身异常（权限不足、路径不存在等）降级为警告
        logger.warning(f"⚠️ 磁盘空间检测失败: {e}")

def check_write_permission(target_path: Path):
    """提前检测目录写入权限"""
    test_file = target_path / ".write_test_tmp"
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise PermissionError(f"无写入权限！程序无法在 {target_path} 中创建文件，请检查文件夹权限或以管理员身份运行。")


def run_pipeline(
    progress_callback: Optional[Callable[[float, str], None]] = None,
    csv_path_override: Optional[str] = None,
) -> dict:
    """
    数据管道主入口（供 UI Worker 和 CLI 共同调用）

    Args:
        progress_callback: 进度回调函数 (percent, text)
        csv_path_override: 可选，直接指定 CSV 文件路径（绕过 config.yaml）
    """
    def report_progress(percent: float, text: str):
        logger.info(f"[{percent:.0f}%] {text}")
        if progress_callback:
            try:
                progress_callback(percent, text)
            except Exception:
                pass  # 忽略 UI 回调中的异常，保护核心管道

    pipeline_cfg = CFG.pipeline

    if csv_path_override:
        csv_path = Path(csv_path_override)
        # 相对路径基于 PROJECT_ROOT 解析（与 config.yaml 行为一致）
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        logger.info(f"📂 使用命令行/UI 指定的数据源覆盖: {csv_path}")
    else:
        csv_path_config = CFG.paths.scada_data_csv
        csv_path = Path(csv_path_config) if Path(csv_path_config).is_absolute() else (PROJECT_ROOT / csv_path_config)
    
    save_dir_config = CFG.paths.artifacts_dir
    save_dir = Path(save_dir_config) if Path(save_dir_config).is_absolute() else (PROJECT_ROOT / save_dir_config)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report_progress(5, "正在进行环境预检 (磁盘与权限)...")
    check_disk_space(save_dir, min_free_gb=2.0)
    check_write_permission(save_dir)

    if not csv_path.exists():
        logger.error(f"❌ 数据文件不存在: {csv_path}")
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    report_progress(10, "正在初始化数据管道...")
    pipeline = WWTPDataPipeline(
        freq=pipeline_cfg.freq,
        lookback=pipeline_cfg.lookback,
        horizon=pipeline_cfg.horizon,
        max_interp_gap=pipeline_cfg.max_interp_gap,
        test_ratio=pipeline_cfg.test_ratio
    )

    logger.info(f"🚀 启动数据管道 | 数据源: {csv_path} | 输出目录: {save_dir}")
    
    df_raw = df_clean = tft_data = lgbm_data = result = None
    
    try:
        report_progress(15, "正在加载与校验 CSV 数据...")
        df_raw = pipeline.load_and_validate(str(csv_path))
        
        report_progress(30, "正在清洗、对齐与插值...")
        df_clean = pipeline.clean_and_resample(df_raw)
        
        report_progress(60, "正在构建 TFT 时序滑窗序列...")
        tft_data = pipeline.build_tft_sequences(df_clean)
        
        report_progress(80, "正在构建 LGBM 表格特征...")
        lgbm_data = pipeline.build_lgbm_features(df_clean)
        
        report_progress(95, "正在持久化保存管道工件 (Scaler/Config)...")
        pipeline.save_artifacts(str(save_dir))
        
        result = {'tft': tft_data, 'lgbm': lgbm_data, 'clean_df': df_clean}
        
    except MemoryError:
        logger.error("🚨 致命错误：系统内存溢出 (MemoryError)！")
        raise
    except PermissionError as e:
        logger.error(f"🚨 权限错误：{e}")
        raise

    logger.success("\n" + "=" * 60)
    logger.success(f"TFT 训练集形状: {result['tft']['X_train'].shape}")
    logger.success(f"LGBM 特征数量:  {len(result['lgbm']['feature_names'])}")
    logger.success(f"清洗后有效样本:  {len(result['clean_df'])}")
    logger.success("=" * 60)

    output_meta = {
        "save_dir": str(save_dir.resolve()),
        "tft_train_shape": result['tft']['X_train'].shape,
        "lgbm_feature_count": len(result['lgbm']['feature_names']),
        "valid_samples": len(result['clean_df'])
    }
    
    del result, tft_data, lgbm_data, df_clean, df_raw
    gc.collect()
    
    report_progress(100, f"✅ 数据管道执行完毕! 产物已保存至: {output_meta['save_dir']}")

    # 🚀 【跨文件手术 3】增加训练调度钩子，防止数据管道跑完后模型未训练
    logger.info("\n" + "="*60)
    logger.info("📋 【下一步操作指引】数据特征已就绪，请启动模型训练：")
    logger.info("   1. 训练 LGBM/TFT 预测模型: python train.py --model lgbm_tft")
    logger.info("   2. 训练 PPO 控制策略:      python train.py --model ppo")
    logger.info("   3. 启动实时推理引擎:       python inference.py")
    logger.info("="*60 + "\n")

    return output_meta


# ================= CLI 专属测试区 =================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WWTP 数据管道 — 数据清洗、特征工程与训练数据生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                          # 使用 config.yaml 中的默认路径
  python run_pipeline.py --csv data/my_plant.csv  # 指定 CSV 文件（绕过 config.yaml）
  python run_pipeline.py --csv "D:/工厂数据/2024年/1月.csv"  # 支持绝对路径和中文路径
        """.strip(),
    )
    parser.add_argument(
        "--csv", dest="csv_path",
        help="直接指定 SCADA 数据 CSV 文件路径（相对或绝对路径），无需修改 config.yaml",
    )
    args = parser.parse_args()

    if not IS_PACKAGED:
        pycache_dir = PROJECT_ROOT / "__pycache__"
        if pycache_dir.exists():
            logger.info("🧹 清理 Python 缓存，确保使用最新代码...")
            try:
                for pyc in pycache_dir.glob("*.pyc"):
                    pyc.unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"清理缓存时遇到轻微阻碍 (不影响运行): {e}")

    try:
        def cli_progress(percent: float, text: str):
            logger.info(f"进度 [{percent:5.1f}%] | {text}")

        meta = run_pipeline(
            progress_callback=cli_progress,
            csv_path_override=args.csv_path,
        )

        logger.info("\n🔍 [CLI模式] 开始验证持久化恢复...")
        restored = WWTPDataPipeline.load_artifacts(meta["save_dir"])
        logger.success(f"✅ 管道恢复验证 | 特征数: {len(restored.feature_names)} | 已拟合: {restored.is_fitted}")
    except Exception as e:
        logger.error(f"❌ 数据管道执行失败: {e}")
        sys.exit(1)