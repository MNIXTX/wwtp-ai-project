#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training/train_tft.py
TFT (Temporal Fusion Transformer) 模型专用训练脚本
配合 config_manager.py 与 tft_pure_pytorch.py 使用
"""

import sys
import os

if sys.platform == 'win32':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try: s.reconfigure(encoding='utf-8')
            except Exception: pass
import platform  # [新增] 用于检测操作系统
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

# ==================== 1. 环境与路径引导 ====================
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==================== 2. 业务模块导入 ====================
from config.manager import CFG
from pipeline.data import WWTPDataPipeline
from models.tft import TFTEngine, TrainConfig
from utils.dataset import TFTDataset

def train_tft():
    """执行 TFT 模型训练全流程"""
    logger.info("=" * 60)
    logger.info("🚀 开始执行 TFT 模型训练任务")
    logger.info("=" * 60)

    # --- 1. 配置参数准备 ---
    model_cfg = CFG.model
    train_cfg_params = CFG.training
    paths_cfg = CFG.paths

    # 构建训练配置对象
    train_config = TrainConfig(
        epochs=train_cfg_params.tft_epochs,
        batch_size=model_cfg.tft_batch_size,
        learning_rate=train_cfg_params.tft_lr,
        seq_len=model_cfg.tft_seq_len,
        num_features=model_cfg.tft_num_features,
        hidden_size=model_cfg.tft_hidden_size,
        lstm_layers=model_cfg.tft_lstm_layers,
        dropout=model_cfg.tft_dropout,
        num_attn_heads=model_cfg.tft_num_attn_heads,
        device=model_cfg.device,
        target_clip_min=None,
        target_clip_max=None
    )

    logger.info(f"⚙️  训练配置加载完成 | 轮次: {train_config.epochs} | 批次: {train_config.batch_size}")

    # --- 2. 数据管道执行 ---
    logger.info("🏭 初始化数据管道...")
    artifacts_dir = paths_cfg.artifacts_dir
    if not artifacts_dir.exists():
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline = WWTPDataPipeline(
        freq=CFG.pipeline.freq,
        lookback=CFG.pipeline.lookback,
        horizon=CFG.pipeline.horizon,
        max_interp_gap=CFG.pipeline.max_interp_gap,
        test_ratio=CFG.pipeline.test_ratio
    )

    try:
        df_raw = pipeline.load_and_validate()
        df_clean = pipeline.clean_and_resample(df_raw)
        tft_data = pipeline.build_tft_sequences(df_clean)
        pipeline.save_artifacts(artifacts_dir)
    except Exception as e:
        logger.error(f"❌ 数据管道执行失败: {e}")
        sys.exit(1)

    # --- 3. 模型训练 (核心修改区域) ---
    logger.info("🧠 初始化 TFT 算法引擎...")

    # A. 准备 PyTorch 张量
    X_train_np = tft_data['X_train']
    y_train_np = tft_data['y_train']

    # [重要] 统一 y 形状为 (N, 1): y 可能是 (N, horizon) 或 (N, horizon, 1)
    # 使用 horizon 窗口的均值作为目标（比仅取最后一步更稳定，减少单点噪声）
    if y_train_np.ndim == 3:
        y_train_np = y_train_np.mean(axis=1)   # (N, horizon, 1) -> (N, 1)
    elif y_train_np.ndim == 2 and y_train_np.shape[1] > 1:
        y_train_np = y_train_np.mean(axis=1, keepdims=True)  # (N, horizon) -> (N, 1)

    # [验证] X 特征维度必须与模型配置一致
    expected_features = train_config.num_features
    actual_columns = X_train_np.shape[2]
    if actual_columns != expected_features:
        raise ValueError(
            f"Feature dimension mismatch: X has {actual_columns} columns, "
            f"model expects {expected_features}. "
            f"Check tft_feature_names vs tft_num_features in config.yaml."
        )

    # [重要] 强制转换为 float32，避免 LSTM 计算类型错误
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)

    logger.info(f"📊 数据就绪 | X:{X_train_tensor.shape} | y:{y_train_tensor.shape}")

    # B. 创建 Dataset 和 DataLoader
    dataset = TFTDataset(X_train_tensor, y_train_tensor)

    # [优化] 针对 Windows 系统的多进程兼容性处理
    num_workers = 0 if platform.system() == "Windows" else 4

    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(train_config.device == "cuda"),
        drop_last=True  # 丢弃最后不足一个 batch 的数据，防止 BatchNorm 报错
    )

    logger.success(f"✅ DataLoader 就绪 | 批次数: {len(dataloader)} | Workers: {num_workers}")

    # [新增] 安全检查：如果数据集太小导致 drop_last=True 后无 batch，则降级处理
    if len(dataloader) == 0:
        logger.warning("⚠️ 数据集样本数不足一个 batch，回退到 drop_last=False")
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(train_config.device == "cuda"),
            drop_last=False
        )
        logger.info(f"🔄 DataLoader 已重建 | 批次数: {len(dataloader)}")

    # C. 启动训练
    engine = TFTEngine(config=train_config)

    try:
        logger.info("🔥 开始训练循环...")
        # [修正] 确保传入的是 dataloader 对象
        # 如果你的 TFTEngine.train 还没改好，这里会报错。
        # 必须确保 tft_pure_pytorch.py 中的 train 方法签名是 def train(self, dataloader):
        result = engine.train(dataloader)

        final_loss = result.get('final_loss', 0)
        logger.success(f"✅ 训练完成！最终 Loss: {final_loss:.4f}")

    except KeyboardInterrupt:
        logger.warning("🛑 训练被用户手动中断。")
        sys.exit(0)
    except TypeError as te:
        # [捕获特定错误] 如果这里报 missing argument，说明引擎没改对
        logger.error(f"❌ 接口调用错误: {te}")
        logger.error("请检查 tft_pure_pytorch.py 中的 train 方法是否已改为接收 dataloader")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 训练异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 4. 模型导出 ---
    logger.info("📦 正在导出 ONNX 模型...")
    try:
        model_dir = paths_cfg.model_dir
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = engine.export_onnx(save_dir=str(model_dir), filename="industrial_tft.onnx")

        verification = TFTEngine.verify_onnx(
            onnx_path=onnx_path,
            seq_len=model_cfg.tft_seq_len,
            num_features=model_cfg.tft_num_features
        )
        logger.success(f"🎉 ONNX 验证通过 | 示例预测: {verification['prediction']:.4f}")

    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"❌ ONNX 导出缺少依赖: {e}")
        logger.warning("请运行: pip install onnxscript 后重试")
    except FileNotFoundError as e:
        logger.error(f"❌ 导出路径不可访问: {e}")
    except Exception as e:
        logger.error(f"❌ ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        logger.warning("模型权重已训练完成，但 ONNX 导出未完成。")

    logger.info("=" * 60)
    logger.info("💾 流程结束")
    logger.info("=" * 60)

if __name__ == "__main__":
    train_tft()