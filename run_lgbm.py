#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WWTP 智能优化系统 - LGBM 真实数据一键训练入口
负责"买菜"（数据管道）+ 启动"发动机"（LGBMTrainEngine）
"""
import sys
import os

if sys.platform == 'win32':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try: s.reconfigure(encoding='utf-8')
            except Exception: pass

import joblib
import shutil
import numpy as np
from pathlib import Path
from loguru import logger

# 确保能导入同级目录的模块
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config_manager import CFG
from data_pipeline import WWTPDataPipeline
from lgbm_feature_builder import FeatureConfig, LGBMFeatureBuilder
from lgbm_baseline import LGBMTrainEngine, LGBMTrainConfig

def main():
    logger.info("🚀 启动 WWTP LGBM 真实数据训练流水线...")
    
    # ================= 1. 买菜：运行数据管道 =================
    artifacts_dir = BASE_DIR / CFG.paths.artifacts_dir
    cache_file = artifacts_dir / "lgbm_feature_cache.joblib"
    
    # 🚀 【核心修复 1】防坑机制：校验缓存有效性，防止读取到上次报错留下的"毒缓存"
    use_cache = False
    if cache_file.exists():
        try:
            logger.info(f"⚡ 发现特征缓存，正在校验维度: {cache_file}")
            cached_data = joblib.load(cache_file)
            # 校验 X_train 的列数是否等于 feature_names 的长度
            if hasattr(cached_data['X_train'], 'shape') and len(cached_data['X_train'].shape) == 2:
                actual_dim = cached_data['X_train'].shape[1]
                claimed_dim = len(cached_data.get('feature_names', []))
                if actual_dim == claimed_dim and actual_dim > 0:
                    logger.success(f"✅ 缓存维度校验通过 ({actual_dim}维)，秒级加载！")
                    lgbm_data = cached_data
                    use_cache = True
                else:
                    raise ValueError(f"缓存维度不匹配: 矩阵{actual_dim}维 vs 名字{claimed_dim}个")
            else:
                raise ValueError("缓存数据形状异常")
        except Exception as e:
            logger.warning(f"⚠️ 缓存校验失败 ({e})，将自动销毁旧缓存并重新清洗数据...")
            if cache_file.exists():
                os.remove(cache_file)

    # 如果缓存无效或不存在，跑完整管道
    if not use_cache:
        logger.info("🍳 正在运行数据管道清洗真实 SCADA 数据...")
        pipeline = WWTPDataPipeline()
        result = pipeline.run_full_pipeline(save_dir=str(artifacts_dir))
        lgbm_data = result['lgbm']
        
        # 存个干净的新缓存，下次跑就快了
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(lgbm_data, cache_file)
        logger.success(f"💾 纯净特征矩阵已缓存至: {cache_file}")

    # 🚀 【诊断探针】在这里打印真实维度，彻底终结维度悬案
    X_train_shape = lgbm_data['X_train'].shape if hasattr(lgbm_data['X_train'], 'shape') else 'N/A'
    logger.info(f"🔍 数据交接诊断 | X_train矩阵: {X_train_shape} | feature_names数量: {len(lgbm_data['feature_names'])}")

    # ================= 2. 组装发动机配置 =================
    # 从 config.yaml 动态读取训练参数
    raw_params = CFG.training.lgbm_params
    params_dict = raw_params.model_dump() if hasattr(raw_params, 'model_dump') else dict(raw_params)
    
    train_cfg = LGBMTrainConfig(
        n_estimators=CFG.training.lgbm_n_estimators,
        early_stop_rounds=CFG.training.lgbm_early_stop_rounds,
        log_every=CFG.training.lgbm_log_every,
        divergence_threshold=CFG.training.divergence_threshold,
        lgbm_params=params_dict
    )
    
    # 🚀 【终极修复 2】配置"一统天下"：在实例化时直接注入 YAML 配置，彻底消灭默认值！
    # 提取 config.yaml 中的 lgbm.features 配置块
    yaml_feat_cfg = CFG.lgbm.features
    
    # 将 Pydantic 配置精准映射给 FeatureConfig (Dataclass)
    # 注意：这里假设您的 FeatureConfig 接受这些参数。如果您的重构版支持自动推断，这里传空列表也可。
    feat_cfg = FeatureConfig(
        feature_columns=list(yaml_feat_cfg.feature_columns),  # 强制使用 YAML 中的 3 个基础列
        lag_hours=list(yaml_feat_cfg.lag_hours),              # 强制使用 YAML 中的 5 个滞后
        rolling_windows=list(yaml_feat_cfg.rolling_windows),  # 强制使用 YAML 中的 3 个滚动
        target_col=yaml_feat_cfg.target_column                # 目标列
    )
    
    # 计算并打印真实特征维度；当 YAML 为空列表时，走动态推断模式，维度应来自数据管道输出
    feature_builder = LGBMFeatureBuilder(feat_cfg)
    expected_dim = len(lgbm_data.get('feature_names', []))
    if not feat_cfg.feature_columns:
        expected_dim = len(lgbm_data.get('feature_names', []))
    else:
        expected_dim = len(feature_builder._compute_expected_columns(feat_cfg.feature_columns))
    logger.success(
        f"✅ 特征配置已从 YAML 同步 | 基础列: {feat_cfg.feature_columns or '动态推断'} | 真实特征维度: {expected_dim}"
    )

    # ================= 3. 点火：启动训练引擎 =================
    # 定义模型最终保存目录 (严格对齐 config.yaml)
    save_dir = str(BASE_DIR / CFG.paths.lgbm_model_dir)
    
    engine = LGBMTrainEngine(
        train_config=train_cfg,
        feature_config=feat_cfg,
        # 简单的控制台进度条打印
        progress_cb=lambda p, t: print(f"\r[{p:5.1f}%] {t}", end=""), 
        log_cb=lambda lvl, msg: logger.log(lvl, msg)
    )
    
    print("\n") # 换个行
    
    # 🚀 【防御性编程】捕获训练异常，防止控制台直接卡死或崩溃闪退
    try:
        result = engine.train_and_save(lgbm_data, save_dir=save_dir)
    except Exception as e:
        logger.exception(f"❌ 训练引擎发生未捕获异常: {e}")
        result = {'success': False, 'error': str(e)}
    
    # ================= 4. 交付成果 =================
    if result.get('success'):
        logger.success("\n" + "="*50)
        logger.success("🎉 真实数据 LGBM 模型训练圆满成功！")
        logger.success(f"📁 您的模型文件在这里: {result.get('model_path')}")
        logger.success(f"📊 模型 MAE: {result.get('mae', 0):.3f} | RMSE: {result.get('rmse', 0):.3f}")
        logger.success("="*50 + "\n")
    else:
        logger.error(f"❌ 训练失败: {result.get('error')}")

if __name__ == "__main__":
    main()