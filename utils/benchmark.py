#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型基准对比测试 — 评估 LGBM, TFT, 融合 Ensemble 的预测精度

用法:
  python utils/benchmark.py                          # 全基准测试
  python utils/benchmark.py --model lgbm             # 仅 LGBM
  python utils/benchmark.py --model tft --epochs 30  # TFT (30 epochs)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import time
import json
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader

from loguru import logger
from pipeline.data import WWTPDataPipeline
from config.manager import CFG


@dataclass
class BenchmarkResult:
    model_name: str
    mae: float
    rmse: float
    r2: float
    train_time_sec: float
    inference_time_ms: float
    params: dict


def _load_data():
    pipeline = WWTPDataPipeline()
    df_raw = pipeline.load_and_validate()
    df_clean = pipeline.clean_and_resample(df_raw)
    tft_data = pipeline.build_tft_sequences(df_clean)
    lgbm_data = pipeline.build_lgbm_features(df_clean)
    return tft_data, lgbm_data, pipeline


def benchmark_lgbm(lgbm_data: dict) -> BenchmarkResult:
    """Train and evaluate LightGBM."""
    import lightgbm as lgb

    X_train = lgbm_data['X_train']
    y_train = lgbm_data['y_train']
    X_test = lgbm_data['X_test']
    y_test = lgbm_data['y_test']

    params = CFG.training.lgbm_params
    n_estimators = CFG.training.lgbm_n_estimators
    early_stop = CFG.training.lgbm_early_stop_rounds

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    t0 = time.time()
    model = lgb.train(
        params, train_data,
        num_boost_round=n_estimators,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(period=0)]
    )
    train_time = time.time() - t0

    n_trees = model.best_iteration + 1 if model.best_iteration > 0 else model.current_iteration()

    t0 = time.time()
    preds = model.predict(X_test, num_iteration=n_trees)
    inf_time = (time.time() - t0) / len(X_test) * 1000

    return BenchmarkResult(
        model_name='LightGBM',
        mae=mean_absolute_error(y_test, preds),
        rmse=np.sqrt(mean_squared_error(y_test, preds)),
        r2=r2_score(y_test, preds),
        train_time_sec=train_time,
        inference_time_ms=inf_time,
        params={'n_estimators': n_trees, **{k: v for k, v in params.items() if k in ['learning_rate', 'num_leaves', 'max_depth']}},
    )


def benchmark_tft(tft_data: dict, epochs: int = 30) -> BenchmarkResult:
    """Train and evaluate TFT."""
    from models.tft import TrainConfig, TFTEngine
    from utils.dataset import TFTDataset

    X_train = tft_data['X_train']
    y_train_2d = tft_data['y_train'].mean(axis=1, keepdims=True)
    X_test = tft_data['X_test']
    y_test_2d = tft_data['y_test'].mean(axis=1, keepdims=True)

    config = TrainConfig(
        epochs=epochs,
        batch_size=CFG.model.tft_batch_size,
        learning_rate=CFG.training.tft_lr,
        seq_len=CFG.model.tft_seq_len,
        num_features=CFG.model.tft_num_features,
        hidden_size=CFG.model.tft_hidden_size,
        lstm_layers=CFG.model.tft_lstm_layers,
        dropout=CFG.model.tft_dropout,
        num_attn_heads=CFG.model.tft_num_attn_heads,
        device='cpu',
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_2d, dtype=torch.float32)
    dataset = TFTDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    engine = TFTEngine(config=config)
    t0 = time.time()
    engine.train(dataloader)
    train_time = time.time() - t0

    engine.model.eval()
    X_v = torch.tensor(X_test, dtype=torch.float32)
    y_v = torch.tensor(y_test_2d, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        preds, _ = engine.model(X_v)
    inf_time = (time.time() - t0) / len(X_test) * 1000

    preds_np = preds.numpy().flatten()
    y_np = y_v.numpy().flatten()

    return BenchmarkResult(
        model_name='TFT',
        mae=mean_absolute_error(y_np, preds_np),
        rmse=np.sqrt(mean_squared_error(y_np, preds_np)),
        r2=r2_score(y_np, preds_np),
        train_time_sec=train_time,
        inference_time_ms=inf_time,
        params={'hidden_size': config.hidden_size, 'attn_heads': config.num_attn_heads, 'epochs': epochs},
    )


def benchmark_ensemble(tft_result, lgbm_result,
                        lgbm_preds, tft_preds, y_test,
                        weight_lgbm=0.5, weight_tft=0.5) -> BenchmarkResult:
    """Evaluate weighted ensemble of LGBM + TFT."""
    ensemble_preds = weight_lgbm * lgbm_preds + weight_tft * tft_preds

    return BenchmarkResult(
        model_name=f'Ensemble (LGBM×{weight_lgbm} + TFT×{weight_tft})',
        mae=mean_absolute_error(y_test, ensemble_preds),
        rmse=np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        r2=r2_score(y_test, ensemble_preds),
        train_time_sec=tft_result.train_time_sec + lgbm_result.train_time_sec,
        inference_time_ms=tft_result.inference_time_ms + lgbm_result.inference_time_ms,
        params={'weight_lgbm': weight_lgbm, 'weight_tft': weight_tft},
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model benchmark")
    parser.add_argument("--model", choices=["lgbm", "tft", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30, help="TFT epochs")
    parser.add_argument("--output", "-o", default=None, help="Save results to JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("  WWTP Model Benchmark")
    print("=" * 60)

    print("\n[1/3] Loading data via pipeline...")
    tft_data, lgbm_data, _ = _load_data()
    print(f"  TFT: X_train={tft_data['X_train'].shape}, y_train={tft_data['y_train'].shape}")
    print(f"  LGBM: X_train={lgbm_data['X_train'].shape}, features={len(lgbm_data.get('feature_names', []))}")

    results = []

    if args.model in ("lgbm", "all"):
        print("\n[2/3] Benchmarking LightGBM...")
        r_lgbm = benchmark_lgbm(lgbm_data)
        results.append(r_lgbm)
        print(f"  MAE={r_lgbm.mae:.3f}  RMSE={r_lgbm.rmse:.3f}  R²={r_lgbm.r2:.4f}  "
              f"Train={r_lgbm.train_time_sec:.0f}s  Inference={r_lgbm.inference_time_ms:.3f}ms")

    if args.model in ("tft", "all"):
        print(f"\n[3/3] Benchmarking TFT ({args.epochs} epochs)...")
        r_tft = benchmark_tft(tft_data, epochs=args.epochs)
        results.append(r_tft)
        print(f"  MAE={r_tft.mae:.3f}  RMSE={r_tft.rmse:.3f}  R²={r_tft.r2:.4f}  "
              f"Train={r_tft.train_time_sec:.0f}s  Inference={r_tft.inference_time_ms:.3f}ms")

    if len(results) >= 2:
        lgbm_preds = np.array([])
        tft_preds = np.array([])
        y_test = lgbm_data['y_test']
        # Recompute for ensemble
        import lightgbm as lgb
        train_data = lgb.Dataset(lgbm_data['X_train'], label=lgbm_data['y_train'])
        model_lgbm = lgb.train(CFG.training.lgbm_params, train_data,
                               num_boost_round=CFG.training.lgbm_n_estimators,
                               callbacks=[lgb.early_stopping(CFG.training.lgbm_early_stop_rounds),
                                         lgb.log_evaluation(period=0)])
        n_t = model_lgbm.best_iteration + 1
        lgbm_preds = model_lgbm.predict(lgbm_data['X_test'], num_iteration=n_t)

        from models.tft import TrainConfig, TFTEngine
        from utils.dataset import TFTDataset
        config = TrainConfig(epochs=15, batch_size=128,
                            seq_len=CFG.model.tft_seq_len,
                            num_features=CFG.model.tft_num_features,
                            hidden_size=CFG.model.tft_hidden_size,
                            num_attn_heads=CFG.model.tft_num_attn_heads,
                            device='cpu')
        engine = TFTEngine(config=config)
        X_t = torch.tensor(tft_data['X_train'], dtype=torch.float32)
        y_t = torch.tensor(tft_data['y_train'].mean(axis=1, keepdims=True), dtype=torch.float32)
        ds = TFTDataset(X_t, y_t)
        engine.train(DataLoader(ds, batch_size=128, shuffle=True, drop_last=True))
        engine.model.eval()
        with torch.no_grad():
            tft_preds, _ = engine.model(torch.tensor(tft_data['X_test'], dtype=torch.float32))
        tft_preds = tft_preds.numpy().flatten()

        if len(lgbm_preds) == len(tft_preds):
            r_ens = benchmark_ensemble(r_tft, r_lgbm, lgbm_preds, tft_preds, y_test[:len(lgbm_preds)])
            results.append(r_ens)
            print(f"\n  Ensemble: MAE={r_ens.mae:.3f}  RMSE={r_ens.rmse:.3f}  R²={r_ens.r2:.4f}")

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'Model':25s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'Train':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r.model_name:25s} {r.mae:8.3f} {r.rmse:8.3f} {r.r2:8.4f} {r.train_time_sec:7.0f}s")

    if args.output:
        out = {r.model_name: {'mae': r.mae, 'rmse': r.rmse, 'r2': r.r2,
                'train_time': r.train_time_sec, 'params': r.params} for r in results}
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
