import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import onnxruntime as ort
import time
import os
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

from config_manager import CFG
from lgbm_baseline import validate_dual_model_consensus
from lgbm_feature_builder import LGBMFeatureBuilder, FeatureConfig


# ==========================================
# 文件完整性预检工具（防止损坏文件导致 Native 层死循环）
# ==========================================
def _validate_lgbm_model_file(filepath: Path) -> Optional[str]:
    """检查 LGBM 模型文件头部是否合法。合法返回 None，否则返回错误描述。"""
    try:
        if filepath.stat().st_size < 10:
            return f"文件过小 ({filepath.stat().st_size} bytes)，可能为空或截断"
        with open(filepath, 'rb') as f:
            header = f.read(100)
        # LGBM v4 模型文件应以 b'tree' 开头（文本格式）
        if not header.startswith(b'tree'):
            return f"文件头非法 (期望 'tree'，实际: {header[:20]!r})"
        return None  # 合法
    except Exception as e:
        return f"预检失败: {e}"


def _validate_onnx_model_file(filepath: Path) -> Optional[str]:
    """检查 ONNX 文件是否以合法的 protobuf 魔数开头。合法返回 None，否则返回错误描述。"""
    try:
        if filepath.stat().st_size < 8:
            return f"文件过小 ({filepath.stat().st_size} bytes)"
        with open(filepath, 'rb') as f:
            magic = f.read(8)
        # ONNX protobuf 格式: 0x08 开头 (varint field 1, wire type 0)
        if magic[0:1] != b'\x08':
            return f"非 ONNX protobuf 格式 (首字节: {magic[0]:#04x})"
        return None  # 合法
    except Exception as e:
        return f"预检失败: {e}"


def _validate_joblib_file(filepath: Path) -> Optional[str]:
    """检查 joblib/pickle 文件是否基本合法（大小 + 魔数）。合法返回 None，否则返回错误描述。"""
    try:
        size = filepath.stat().st_size
        if size < 4:
            return f"文件过小 ({size} bytes)"
        if size > 100 * 1024 * 1024:  # 100 MB 上限
            return f"文件过大 ({size / 1024 / 1024:.1f} MB)，拒绝加载"
        with open(filepath, 'rb') as f:
            magic = f.read(4)
        # joblib 文件以 b'\x80\x03' (pickle protocol 3+) 或 b'\x00\x00' (numpy) 开头
        if magic[:2] not in (b'\x80\x03', b'\x80\x04', b'\x80\x05', b'\x00\x00'):
            return f"非 pickle/joblib 格式 (前2字节: {magic[:2]!r})"
        return None  # 合法
    except Exception as e:
        return f"预检失败: {e}"


def _safe_load_with_timeout(load_func, filepath: str, timeout: float = 15.0, label: str = "") -> Tuple[bool, any]:
    """
    在线程中执行加载函数，超时则放弃。
    返回 (success, result_or_error_string)。
    用于防止损坏文件的 C 层死循环导致整个进程卡死。
    """
    result_holder = {"ok": False, "value": None, "error": None}

    def _worker():
        try:
            result_holder["value"] = load_func()
            result_holder["ok"] = True
        except Exception as e:
            result_holder["error"] = str(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        logger.error(f"[TIMEOUT] {label} 加载超时 ({timeout}s) — 文件可能已损坏: {filepath}")
        return False, f"加载超时 ({timeout}s)，文件可能已损坏"

    if result_holder["ok"]:
        return True, result_holder["value"]
    else:
        return False, result_holder["error"]


def _infer_lgbm_base_columns_from_model(lgbm_model) -> Optional[List[str]]:
    """
    从已加载的 LGBM 模型中反推基础特征列名。
    用于 artifacts 缺失或过期时自动恢复，避免 4 列 → 38 特征 vs 101 模型特征的维度错配。

    LGBM feature names look like:
        flow, inf_cod, inf_cod_lag_1, inf_cod_roll_mean_6, hour, dayofweek, ...
    通过去重后缀反推原始基础列。
    """
    import re
    try:
        all_names = lgbm_model.feature_name()
        if not all_names:
            return None
        base = []
        seen = set()
        for name in all_names:
            # 去掉 lag/rolling 后缀
            root = re.sub(r'_(lag|roll_mean|roll_std)_\d+$', '', name)
            if root not in seen and root not in ('hour', 'dayofweek'):
                seen.add(root)
                base.append(root)
        if len(base) >= 4:
            logger.info(f"从 LGBM 模型反推基础列: {len(base)} 列 → {base}")
            return base
        return None
    except Exception as e:
        logger.warning(f"无法从 LGBM 模型反推基础列: {e}")
        return None

class WaterQualityPredictor:
    """
    工业级水质预测推理网关 (ONNX + LightGBM 双引擎)
    支持 TFT 与 LightGBM 双模型交叉验证，具备 Fail-Safe 安全拦截机制
    """
    def __init__(
        self, 
        num_features: Optional[int] = None, 
        seq_len: Optional[int] = None,
        tft_model_filename: Optional[str] = None,
        lgbm_model_filename: Optional[str] = None
    ):
        # 从配置中读取参数，支持构造函数覆盖
        self.num_features = num_features or CFG.model.tft_num_features
        self.seq_len = seq_len or CFG.model.tft_seq_len
        
        # 严格对齐 config.yaml 中的路径配置
        tft_filename = tft_model_filename or "industrial_tft.onnx"
        self.tft_path = os.path.join(CFG.paths.model_dir, tft_filename)
        
        lgbm_filename = lgbm_model_filename or CFG.paths.lgbm_model_file
        self.lgbm_path = os.path.join(CFG.paths.lgbm_model_dir, lgbm_filename)
        
        # 🚀 【诊断增强】打印绝对路径，彻底排查“找不到模型”的幽灵问题
        abs_tft_path = os.path.abspath(self.tft_path)
        abs_lgbm_path = os.path.abspath(self.lgbm_path)
        logger.info(f"[DIAG] 模型搜索路径 (绝对路径):")
        logger.info(f"  -> TFT  : {abs_tft_path}")
        logger.info(f"  -> LGBM : {abs_lgbm_path}")

        # 1. 初始化 TFT 模型 (使用 ONNX Runtime 加速推理)
        self.tft_session = None
        self.tft_error = None
        tft_p = Path(self.tft_path)
        if tft_p.exists() and tft_p.stat().st_size > 0:
            # 预检 ONNX 文件头，防止损坏文件导致 ONNX Runtime 卡死
            tft_err = _validate_onnx_model_file(tft_p)
            if tft_err:
                self.tft_error = f"文件损坏: {tft_err}"
                logger.error(f"[ERROR] TFT 模型文件损坏: {self.tft_path} — {tft_err}")
            else:
                ok, result = _safe_load_with_timeout(
                    lambda: ort.InferenceSession(self.tft_path, providers=['CPUExecutionProvider']),
                    self.tft_path, timeout=20.0, label="TFT (ONNX)"
                )
                if ok:
                    self.tft_session = result
                    logger.info(f"[OK] TFT (ONNX) 模型加载成功: {self.tft_path}")
                else:
                    self.tft_error = f"加载失败: {result}"
                    logger.error(f"[ERROR] TFT (ONNX) 模型加载失败: {result}")
        else:
            self.tft_error = "文件不存在或为空"
            logger.warning(f"[WARN] TFT 模型文件未找到: {self.tft_path}，将仅使用 LGBM 推理")
            
        # 2. 初始化 LightGBM 模型 (传统机器学习基线)
        self.lgbm_model = None
        self.lgbm_error = None
        lgbm_p = Path(self.lgbm_path)
        if lgbm_p.exists() and lgbm_p.stat().st_size > 0:
            # 预检 LGBM 文件头，防止损坏文件导致 C++ 层死循环（核心修复！）
            lgbm_err = _validate_lgbm_model_file(lgbm_p)
            if lgbm_err:
                self.lgbm_error = f"文件损坏: {lgbm_err}"
                logger.error(f"[ERROR] LGBM 模型文件损坏: {self.lgbm_path} — {lgbm_err}")
            else:
                ok, result = _safe_load_with_timeout(
                    lambda: lgb.Booster(model_file=self.lgbm_path),
                    self.lgbm_path, timeout=15.0, label="LGBM Booster"
                )
                if ok:
                    self.lgbm_model = result
                    logger.info(f"[OK] LGBM 模型加载成功: {self.lgbm_path}")
                else:
                    self.lgbm_error = f"加载失败: {result}"
                    logger.error(f"[ERROR] LGBM 模型加载失败: {result}")
        else:
            self.lgbm_error = "文件不存在或为空"
            logger.warning(f"[WARN] LGBM 模型文件未找到: {self.lgbm_path}，将仅使用 TFT 推理")
            
        # 3. 从配置中读取交叉验证阈值
        self.divergence_threshold = CFG.training.divergence_threshold

        # 4. [Fix] Load scalers from artifacts for TFT input/output normalization
        self.feature_scaler = None
        self.target_scaler = None
        self._lgbm_base_cols = list(CFG.lgbm.features.feature_columns)
        self._load_scalers()

        # 5. [Fix] Init LGBMFeatureBuilder - load base columns from training artifacts
        lgbm_cols = list(CFG.lgbm.features.feature_columns)
        if not lgbm_cols:
            # Try to load the actual base columns used during training
            artifacts_dir = Path(CFG.paths.artifacts_dir)
            pipeline_config_path = artifacts_dir / 'pipeline_config.json'
            if pipeline_config_path.exists():
                try:
                    with open(pipeline_config_path, 'r') as f:
                        saved = json.load(f)
                    saved_cols = saved.get('lgbm_base_columns_for_inference')
                    if saved_cols:
                        lgbm_cols = list(saved_cols)
                        logger.info(f"LGBM columns loaded from training artifacts: {len(lgbm_cols)} columns")
                except Exception:
                    pass
            # Fallback: try to infer from trained LGBM model (most reliable)
            if not lgbm_cols and self.lgbm_model is not None:
                inferred = _infer_lgbm_base_columns_from_model(self.lgbm_model)
                if inferred:
                    lgbm_cols = inferred
                    logger.info(f"LGBM columns inferred from model: {len(lgbm_cols)} columns")
            # Last resort: TFT features (4 cols) — will likely mismatch the model
            if not lgbm_cols:
                lgbm_cols = list(CFG.model.tft_feature_names)
                logger.warning(f"Using TFT features as LGBM columns (fallback): {lgbm_cols}")
        self._lgbm_base_cols = lgbm_cols

        self._lgbm_builder = LGBMFeatureBuilder(FeatureConfig(
            feature_columns=lgbm_cols,
            rolling_feature_columns=list(CFG.lgbm.features.rolling_feature_columns) if list(CFG.lgbm.features.rolling_feature_columns) else lgbm_cols,
            lag_hours=list(CFG.lgbm.features.lag_hours),
            rolling_windows=list(CFG.lgbm.features.rolling_windows),
            target_col=CFG.lgbm.features.target_column,
        ))

        logger.info(f"[OK] Predictor ready | TFT features: {self.num_features} | "
                    f"Seq: {self.seq_len}h | Divergence threshold: {self.divergence_threshold}")

    def _load_scalers(self):
        """[Fix] 加载特征/目标归一化器（带文件完整性预检，防止损坏 pickle 导致死循环）"""
        artifacts_dir = Path(CFG.paths.artifacts_dir)
        scaler_path = artifacts_dir / 'scaler.pkl'
        target_scaler_path = artifacts_dir / 'target_scaler.pkl'

        import joblib
        if scaler_path.exists() and scaler_path.stat().st_size > 0:
            sc_err = _validate_joblib_file(scaler_path)
            if sc_err:
                logger.error(f"[ERROR] Feature scaler 文件损坏: {scaler_path} — {sc_err}")
            else:
                ok, result = _safe_load_with_timeout(
                    lambda: joblib.load(scaler_path),
                    str(scaler_path), timeout=10.0, label="Feature Scaler"
                )
                if ok:
                    self.feature_scaler = result
                    logger.info(f"[OK] Feature scaler loaded ({self.feature_scaler.n_features_in_} features)")
                else:
                    logger.warning(f"[WARN] Failed to load feature scaler: {result}")
        else:
            logger.warning(f"[WARN] Feature scaler not found at {scaler_path} — TFT input will NOT be normalized")

        if target_scaler_path.exists() and target_scaler_path.stat().st_size > 0:
            ts_err = _validate_joblib_file(target_scaler_path)
            if ts_err:
                logger.error(f"[ERROR] Target scaler 文件损坏: {target_scaler_path} — {ts_err}")
            else:
                ok, result = _safe_load_with_timeout(
                    lambda: joblib.load(target_scaler_path),
                    str(target_scaler_path), timeout=10.0, label="Target Scaler"
                )
                if ok:
                    self.target_scaler = result
                    logger.info(f"[OK] Target scaler loaded (mean={self.target_scaler.mean_[0]:.2f}, std={self.target_scaler.scale_[0]:.2f})")
                else:
                    logger.warning(f"[WARN] Failed to load target scaler: {result}")
        else:
            logger.warning(f"[WARN] Target scaler not found at {target_scaler_path} — TFT output will NOT be inverse-transformed")

    def get_feature_names(self) -> List[str]:
        """返回 TFT 模型需要的特征名列表 (供 UI 动态生成输入框)"""
        return list(CFG.model.tft_feature_names)

    def is_healthy(self) -> bool:
        """
        检查预测网关健康状态。
        只要 TFT (ONNX) 或 LightGBM 中至少有一个模型成功加载，即视为健康。
        """
        return self.tft_session is not None or self.lgbm_model is not None

    def get_health_details(self) -> Dict[str, str]:
        """
        获取详细的健康诊断信息（供 UI 侧边栏精准显示）
        """
        details = {
            "tft_status": "offline",
            "lgbm_status": "offline",
            "tft_msg": "",
            "lgbm_msg": ""
        }
        
        if self.tft_session is not None:
            details["tft_status"] = "online"
            details["tft_msg"] = "ONNX 运行正常"
        else:
            details["tft_msg"] = self.tft_error or "未知错误"
            
        if self.lgbm_model is not None:
            details["lgbm_status"] = "online"
            details["lgbm_msg"] = "Booster 运行正常"
        else:
            details["lgbm_msg"] = self.lgbm_error or "未知错误"
            
        return details

    def _prepare_tft_input(self, history_data: Dict[str, List[float]]) -> np.ndarray:
        """
        将字典格式的历史数据转换为 TFT 输入数组 (1, seq_len, num_features).
        [Fix] 应用 feature_scaler 做 z-score 归一化，与训练时保持一致.
        """
        feature_keys = CFG.model.tft_feature_names

        data_matrix = []
        for key in feature_keys[:self.num_features]:
            if key in history_data:
                series = list(history_data[key])[-self.seq_len:]
                if len(series) < self.seq_len:
                    series = [series[0]] * (self.seq_len - len(series)) + series
                data_matrix.append(series)
            else:
                logger.warning(f"[WARN] Missing feature in history: {key}, filling with 0")
                data_matrix.append([0.0] * self.seq_len)

        X = np.array(data_matrix, dtype=np.float32).T  # (seq_len, num_features)

        # [Fix] Apply feature scaler transform (z-score normalization)
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X).astype(np.float32)

        # [Fix] 防止全零输入导致 LayerNorm 除零崩溃
        # 当所有时间步输入相同时，z-score 归一化后全为 0，LayerNorm 的方差为 0 触发 ONNX 错误：
        # "Size of X.shape[axis:] must be larger than 1, got 1"
        # 加入 1e-6 量级的随机噪声，不影响预测结果但防止除零
        if np.allclose(X, 0, atol=1e-8) or np.std(X) < 1e-8:
            X += np.random.randn(*X.shape).astype(np.float32) * 1e-6

        return np.expand_dims(X, axis=0)  # (1, seq_len, num_features)

    def _prepare_lgbm_input(self, history_data: Dict[str, List[float]]) -> Optional[np.ndarray]:
        """
        Prepare LGBM input features.
        [Fix] Use explicit base columns (self._lgbm_base_cols) instead of relying on
        empty CFG.lgbm.features.feature_columns which causes empty DataFrame at inference.
        """
        try:
            lgbm_base_cols = self._lgbm_base_cols
            if not lgbm_base_cols:
                logger.error("[ERROR] No LGBM base columns configured for inference")
                return None

            max_window = max(max(CFG.lgbm.features.lag_hours), max(CFG.lgbm.features.rolling_windows))
            keep_len = max(max_window * 2, 50)

            df_dict = {}
            for col in lgbm_base_cols:
                if col in history_data:
                    series = list(history_data[col])
                    if len(series) < keep_len:
                        # Pad short series to keep_len
                        series = [series[0]] * (keep_len - len(series)) + series
                    df_dict[col] = series[-keep_len:]
                else:
                    df_dict[col] = [0.0] * keep_len

            df = pd.DataFrame(df_dict)
            if df.empty or len(df.columns) == 0:
                logger.error("[ERROR] LGBM input DataFrame is empty")
                return None

            df_feat = self._lgbm_builder.build(df, is_inference=True)
            df_aligned = self._lgbm_builder.align_columns(df_feat)
            X_lgbm = df_aligned.iloc[[-1]].values.astype(np.float32)

            if self.lgbm_model and X_lgbm.shape[1] != self.lgbm_model.num_feature():
                logger.error(f"[ERROR] LGBM feature count mismatch: got {X_lgbm.shape[1]}, expected {self.lgbm_model.num_feature()}")
                # 自动修复：从模型反推正确的 base columns，重建 builder
                inferred = _infer_lgbm_base_columns_from_model(self.lgbm_model)
                if inferred and inferred != self._lgbm_base_cols:
                    logger.info(f"自动修复：从模型反推 {len(inferred)} 个基础列，重建特征构建器")
                    self._lgbm_base_cols = inferred
                    self._lgbm_builder = LGBMFeatureBuilder(FeatureConfig(
                        feature_columns=inferred,
                        rolling_feature_columns=list(CFG.lgbm.features.rolling_feature_columns) if list(CFG.lgbm.features.rolling_feature_columns) else inferred,
                        lag_hours=list(CFG.lgbm.features.lag_hours),
                        rolling_windows=list(CFG.lgbm.features.rolling_windows),
                        target_col=CFG.lgbm.features.target_column,
                    ))
                    # 用修正后的 builder 重试
                    df_feat = self._lgbm_builder.build(df, is_inference=True)
                    df_aligned = self._lgbm_builder.align_columns(df_feat)
                    X_lgbm = df_aligned.iloc[[-1]].values.astype(np.float32)
                    if X_lgbm.shape[1] == self.lgbm_model.num_feature():
                        logger.info(f"自动修复成功：特征维度已对齐 ({X_lgbm.shape[1]})")
                    else:
                        logger.error(f"自动修复失败：仍不匹配 ({X_lgbm.shape[1]} != {self.lgbm_model.num_feature()})")
                        return None
                else:
                    return None

            return X_lgbm

        except Exception as e:
            logger.error(f"[ERROR] LGBM feature construction failed: {e}")
            return None

    def predict(self, history_data: Dict[str, List[float]]) -> Dict:
        """核心推理接口"""
        start_time = time.time()
        result = {
            "status": "success",
            "predictions": {},
            "feature_importance": {},
            "warnings": [],
            "inference_time_ms": 0.0
        }

        # ================= 1. TFT (ONNX) 模型推理 =================
        tft_pred = None
        try:
            if self.tft_session:
                X_tft = self._prepare_tft_input(history_data)
                ort_outs = self.tft_session.run(None, {'input_data': X_tft})
                
                tft_pred_raw = float(ort_outs[0][0][0])  # z-score normalized value
                tft_weights = ort_outs[1].mean(axis=(0, 1))

                # [Fix] Inverse transform to convert from z-score back to real COD (mg/L)
                if self.target_scaler is not None:
                    tft_pred = float(self.target_scaler.inverse_transform(
                        np.array([[tft_pred_raw]], dtype=np.float32)
                    )[0, 0])
                else:
                    tft_pred = tft_pred_raw  # fallback (no scaler available)

                result["predictions"]["tft_cod"] = round(tft_pred, 2)
                
                feature_names = CFG.model.tft_feature_names 
                for name, weight in zip(feature_names[:self.num_features], tft_weights):
                    result["feature_importance"][name] = round(float(weight), 4)
            else:
                logger.warning("[WARN] TFT 模型未加载，跳过推理")
        except Exception as e:
            err_msg = str(e)
            # 裁剪 ONNX 超长错误信息，只保留关键部分
            if len(err_msg) > 300:
                err_msg = err_msg[:300] + "..."
            logger.error(f"[ERROR] TFT 推理失败: {err_msg}")
            result["warnings"].append(
                f"TFT模型推理异常（可能是模型与当前 ONNX Runtime 版本不兼容，"
                f"建议重新训练: python train_tft.py）: {err_msg}"
            )

        # ================= 2. LightGBM 模型推理 =================
        lgbm_pred = None
        if self.lgbm_model:
            try:
                X_lgbm = self._prepare_lgbm_input(history_data)
                if X_lgbm is not None:
                    lgbm_pred = float(self.lgbm_model.predict(X_lgbm)[0])
                    result["predictions"]["lgbm_cod"] = round(lgbm_pred, 2)
                else:
                    logger.error("[ERROR] LGBM 输入特征构建失败")
            except Exception as e:
                logger.error(f"[ERROR] LightGBM 推理失败: {e}")
                result["warnings"].append(f"LGBM模型推理异常: {str(e)}")

        # ================= 3. 双模型交叉验证 (核心安全机制) =================
        if tft_pred is not None and lgbm_pred is not None:
            consensus_result = validate_dual_model_consensus(
                lgbm_prediction=lgbm_pred, 
                tft_prediction=tft_pred, 
                threshold=self.divergence_threshold
            )
            
            result["predictions"]["divergence"] = round(consensus_result["divergence"], 2)
            
            if consensus_result["is_consensus"]:
                result["predictions"]["final_cod"] = round(consensus_result["final_prediction"], 2)
            else:
                warning_msg = f"[WARN] {consensus_result['message']}，已拦截自动控制指令！"
                logger.warning(warning_msg)
                result["warnings"].append(warning_msg)
                result["status"] = "warning_divergence"
                result["predictions"]["final_cod"] = round(max(tft_pred, lgbm_pred), 2)
                
        elif tft_pred is not None:
            result["predictions"]["final_cod"] = round(tft_pred, 2)
        elif lgbm_pred is not None:
            result["predictions"]["final_cod"] = round(lgbm_pred, 2)
        else:
            result["status"] = "error_no_models"
            result["warnings"].append("[ERROR] 致命错误：TFT 与 LGBM 模型均未成功加载！")

        result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return result

if __name__ == "__main__":
    import json
    
    mock_scada_data = {
        'inf_cod': list(np.random.normal(350, 30, 48)),
        'inf_nh3': list(np.random.normal(30, 5, 48)),       
        'DO_reactor': list(np.random.normal(2.0, 0.2, 48)),
        'MLSS_reactor': list(np.random.normal(3000, 100, 48)),
        'flow': list(np.random.normal(10000, 500, 48)),     
        'do_meas': list(np.random.normal(2.0, 0.2, 48))     
    }
    
    predictor = WaterQualityPredictor()
    
    logger.info("\n--- 开始执行水质预测与交叉验证 ---")
    output = predictor.predict(mock_scada_data)
    
    print("\n" + "="*40)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("="*40)