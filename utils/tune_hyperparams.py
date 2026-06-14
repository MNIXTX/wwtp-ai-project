#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超参数自动优化 — 基于 Optuna (来自 pytorch-forecasting 最佳实践)

支持:
  - LGBM 超参搜索 (learning_rate, max_depth, num_leaves, reg_alpha, reg_lambda, ...)
  - TFT 超参搜索 (hidden_size, dropout, learning_rate, lstm_layers, attn_heads)
  - 时间序列交叉验证评分

用法:
  python utils/tune_hyperparams.py --model lgbm --trials 50    # LGBM 调优
  python utils/tune_hyperparams.py --model tft --trials 30     # TFT 调优
  python utils/tune_hyperparams.py --model all --trials 50     # 全部调优
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings("ignore")

from loguru import logger

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  optuna 未安装。运行: pip install optuna")


def _load_data():
    """Load cleaned data from pipeline."""
    from pipeline.data import WWTPDataPipeline
    pipeline = WWTPDataPipeline()
    df_raw = pipeline.load_and_validate()
    df_clean = pipeline.clean_and_resample(df_raw)
    tft_data = pipeline.build_tft_sequences(df_clean)
    lgbm_data = pipeline.build_lgbm_features(df_clean)
    return tft_data, lgbm_data, pipeline


# ═══════════════════════════════════════════════════════════
# LGBM Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════

def _lgbm_objective(trial, X, y, feature_names, n_cv_folds=5):
    """Optuna objective for LightGBM."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
    }
    n_estimators = trial.suggest_int('n_estimators', 200, 3000)
    early_stop = trial.suggest_int('early_stop_rounds', 20, 150)

    # Time-series CV
    n = len(X)
    fold_size = n // (n_cv_folds + 1)
    scores = []

    for fold in range(n_cv_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end - test_start < 10:
            continue

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_te, label=y_te, reference=train_data)

        model = lgb.train(
            params, train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(period=0)]
        )
        if model is not None and model.current_iteration() > 0:
            n_trees = model.best_iteration + 1 if model.best_iteration > 0 else model.current_iteration()
            preds = model.predict(X_te, num_iteration=n_trees)
            scores.append(mean_absolute_error(y_te, preds))

    return np.mean(scores) if scores else float('inf')


def tune_lgbm(lgbm_data: dict, n_trials: int = 50, n_cv_folds: int = 5):
    """Run Optuna optimization for LightGBM."""
    if not HAS_OPTUNA:
        return None

    X = np.vstack([lgbm_data['X_train'], lgbm_data['X_test']])
    y = np.concatenate([lgbm_data['y_train'], lgbm_data['y_test']])
    feature_names = lgbm_data.get('feature_names', [])

    logger.info(f"LGBM tuning: {n_trials} trials, {n_cv_folds}-fold CV, "
                f"X={X.shape}, features={len(feature_names)}")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    def objective(trial):
        return _lgbm_objective(trial, X, y, feature_names, n_cv_folds)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    logger.success(f"LGBM tuning done ({elapsed:.0f}s) | Best MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    return {
        'best_mae': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'elapsed_sec': elapsed,
    }


# ═══════════════════════════════════════════════════════════
# TFT Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════

def _tft_objective(trial, X_train, y_train, X_test, y_test, num_features, seq_len):
    """Optuna objective for TFT."""
    import torch
    from torch.utils.data import DataLoader
    from models.tft import IndustrialTFT, TrainConfig, TFTEngine
    from utils.dataset import TFTDataset

    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    attn_heads = trial.suggest_categorical('attn_heads', [2, 4, 8])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Ensure attn_heads divides hidden_size
    if hidden_size % attn_heads != 0:
        return float('inf')

    config = TrainConfig(
        epochs=30, batch_size=batch_size, learning_rate=learning_rate,
        seq_len=seq_len, num_features=num_features,
        hidden_size=hidden_size, lstm_layers=lstm_layers,
        dropout=dropout, num_attn_heads=attn_heads,
        device='cpu',
    )

    try:
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        X_v = torch.tensor(X_test, dtype=torch.float32)
        y_v = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

        dataset = TFTDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

        engine = TFTEngine(config=config)
        result = engine.train(dataloader)

        # Validation loss
        engine.model.eval()
        with torch.no_grad():
            preds, _ = engine.model(X_v)
            val_loss = torch.nn.functional.mse_loss(preds, y_v).item()
        return val_loss
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return float('inf')


def tune_tft(tft_data: dict, n_trials: int = 30):
    """Run Optuna optimization for TFT."""
    if not HAS_OPTUNA:
        return None

    X_train = tft_data['X_train']
    y_train = tft_data['y_train'].mean(axis=1, keepdims=True)  # (N, horizon) → (N, 1)
    X_test = tft_data['X_test']
    y_test = tft_data['y_test'].mean(axis=1, keepdims=True)
    num_features = X_train.shape[2]
    seq_len = X_train.shape[1]

    logger.info(f"TFT tuning: {n_trials} trials, X={X_train.shape}, features={num_features}")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    def objective(trial):
        return _tft_objective(trial, X_train, y_train, X_test, y_test, num_features, seq_len)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    logger.success(f"TFT tuning done ({elapsed:.0f}s) | Best loss: {study.best_value:.6f}")
    logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    return {
        'best_loss': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'elapsed_sec': elapsed,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--model", choices=["lgbm", "tft", "all"], default="lgbm")
    parser.add_argument("--trials", "-n", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for LGBM")
    parser.add_argument("--output", "-o", default=None, help="Save best params to JSON")
    args = parser.parse_args()

    if not HAS_OPTUNA:
        print("Please install optuna: pip install optuna")
        sys.exit(1)

    print("Loading data via pipeline...")
    tft_data, lgbm_data, pipeline = _load_data()

    results = {}

    if args.model in ("lgbm", "all"):
        print("\n" + "=" * 60)
        print("  Optimizing LightGBM Hyperparameters")
        print("=" * 60)
        results['lgbm'] = tune_lgbm(lgbm_data, n_trials=args.trials, n_cv_folds=args.cv_folds)

    if args.model in ("tft", "all"):
        print("\n" + "=" * 60)
        print("  Optimizing TFT Hyperparameters")
        print("=" * 60)
        results['tft'] = tune_tft(tft_data, n_trials=max(args.trials // 2, 15))

    if args.output and results:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nBest params saved to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
