#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI 推理服务 — 将模型推理与 Streamlit UI 解耦

架构:
  Streamlit (port 8501) ──HTTP──▶ FastAPI (port 8502) ──▶ TFT/LGBM/ASM1

优势:
  - 模型只加载一次，常驻内存
  - Streamlit 页面刷新不触发模型重载
  - 可独立扩展/重启推理服务
  - 支持 POST /predict 和 GET /health

启动:
  python api_server.py                        # 默认端口 8502
  python api_server.py --port 8503 --reload  # 开发模式

依赖: pip install fastapi uvicorn
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# Global model cache — loaded once at startup
# ═══════════════════════════════════════════════════════════

_model_cache: dict = {}


def _load_models():
    """Load all models into memory (called once at startup)."""
    logger.info("Loading prediction models...")
    t0 = time.time()

    try:
        from pipeline.gateway import WaterQualityPredictor
        predictor = WaterQualityPredictor()
        # Warm-up: run a dummy prediction to initialize ONNX session
        dummy_history = {
            "inf_cod": [300.0] * 24, "inf_nh3": [30.0] * 24,
            "flow": [10000.0] * 24, "DO_reactor": [2.0] * 24,
            "MLSS_reactor": [3000.0] * 24, "temp": [15.0] * 24, "pH": [7.3] * 24,
        }
        predictor.predict(dummy_history)
        _model_cache["predictor"] = predictor
        _model_cache["status"] = {
            "tft_ok": getattr(predictor, 'tft_session', None) is not None,
            "lgbm_ok": getattr(predictor, 'lgbm_model', None) is not None,
            "loaded": True,
        }
        logger.success(f"Models loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        _model_cache["status"] = {"tft_ok": False, "lgbm_ok": False, "loaded": False, "error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load models on startup, clean up on shutdown."""
    _load_models()
    yield
    _model_cache.clear()


app = FastAPI(
    title="WWTP AI Inference API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_methods=["*"], allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    history: dict = Field(..., description="Dict of feature_name -> list of values (lookback window)")
    return_importance: bool = Field(default=False, description="Return TFT feature importance weights")


class PredictResponse(BaseModel):
    predictions: dict = Field(..., description="tft_cod, lgbm_cod, final_cod, divergence")
    status: str = "ok"
    warnings: list = []
    inference_time_ms: float = 0.0
    feature_importance: Optional[dict] = None
    model_status: dict = {}


class HealthResponse(BaseModel):
    status: str
    models: dict
    uptime_sec: float


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════

_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check + model status."""
    return HealthResponse(
        status="ok" if _model_cache.get("status", {}).get("loaded") else "degraded",
        models=_model_cache.get("status", {}),
        uptime_sec=time.time() - _start_time,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run dual-model prediction."""
    predictor = _model_cache.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0 = time.time()
    try:
        result = predictor.predict(req.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    inference_ms = (time.time() - t0) * 1000

    predictions = result.get("predictions", {})
    resp = PredictResponse(
        predictions=predictions,
        status=result.get("status", "ok"),
        warnings=result.get("warnings", []),
        inference_time_ms=inference_ms,
        model_status=_model_cache.get("status", {}),
    )

    if req.return_importance:
        resp.feature_importance = result.get("feature_importance")

    return resp


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="WWTP AI Inference API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logger.info(f"Starting inference API at http://{args.host}:{args.port}")
    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=args.reload)
