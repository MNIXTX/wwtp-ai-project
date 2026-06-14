"""UI 适配层的预测网关兼容入口。

该模块提供一个稳定的 WaterQualityPredictor 导出接口，避免 UI 层
直接依赖内部实现细节或缺失的推理依赖。
"""

import logging
from typing import Any, Dict

from config.manager import CFG

logger = logging.getLogger(__name__)


class FallbackWaterQualityPredictor:
    """当真实网关模块不可用时的安全降级实现。"""

    def __init__(self) -> None:
        self.tft_path = getattr(getattr(CFG, "paths", None), "model_dir", "models")
        self.lgbm_path = getattr(getattr(CFG, "paths", None), "lgbm_model_dir", "models/lgbm")
        self.tft_session = None
        self.lgbm_model = None

    def is_healthy(self) -> bool:
        return False

    def get_health_details(self) -> Dict[str, str]:
        return {
            "tft_status": "offline",
            "lgbm_status": "offline",
            "tft_msg": "真实预测网关未就绪，已启用降级模式",
            "lgbm_msg": "真实预测网关未就绪，已启用降级模式",
        }

    def predict(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "degraded",
            "predictions": {
                "final_cod": None,
                "tft_cod": None,
                "lgbm_cod": None,
            },
            "feature_importance": {},
            "warnings": ["预测网关尚未就绪，当前为降级模式。"],
            "inference_time_ms": 0.0,
        }

    def get_feature_names(self) -> list:
        """返回 TFT 模型需要的特征名列表（降级模式下返回默认值）"""
        return ["inf_cod", "inf_nh3", "DO_reactor", "MLSS_reactor"]


try:
    from pipeline.gateway import WaterQualityPredictor as _RealWaterQualityPredictor
except Exception as exc:  # pragma: no cover - runtime compatibility path
    logger.warning("预测网关模块加载失败，已启用降级适配器: %s", exc)
    WaterQualityPredictor = FallbackWaterQualityPredictor
else:
    # 仅在真实网关可用时使用它；否则保持降级实现。
    WaterQualityPredictor = _RealWaterQualityPredictor
