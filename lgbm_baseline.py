# lgbm_baseline.py
import numpy as np
import pandas as pd  
import lightgbm as lgb
import joblib
import os
import time
import shutil
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional, Callable, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from loguru import logger

# 假设 LGBMFeatureBuilder 和 FeatureConfig 在同级目录
from lgbm_feature_builder import LGBMFeatureBuilder, FeatureConfig
# 🚀 【手术 1】引入全局配置管理器，替代原生 yaml 解析
from config_manager import CFG 

# ==========================================
# 1. 解耦核心：配置数据类与回调接口
# ==========================================
@dataclass
class LGBMTrainConfig:
    """LightGBM 训练配置"""
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression', 'metric': 'mae', 'verbose': -1,
        'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1,
        'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8
    })
    n_estimators: int = 500
    early_stop_rounds: int = 50
    log_every: int = 50
    
    divergence_threshold: float = 10.0
    min_free_gb: float = 2.0

@dataclass
class LGBMTrainState:
    """训练状态载体"""
    iteration: int
    total_iterations: int
    train_loss: float
    valid_loss: float
    elapsed_time: float

ProgressCallback = Callable[[float, str], None]
StateCallback = Callable[[LGBMTrainState], None]
LogCallback = Callable[[str, str], None]

# ==========================================
# 2. 工业级 LGBM 训练引擎
# ==========================================
class LGBMTrainEngine:
    """LightGBM 训练引擎 (支持中断、环境预检、细粒度状态抛出)"""
    
    def __init__(self, 
                 train_config: LGBMTrainConfig,
                 feature_config: FeatureConfig,
                 progress_cb: Optional[ProgressCallback] = None,
                 state_cb: Optional[StateCallback] = None,
                 log_cb: Optional[LogCallback] = None):
        
        self.train_cfg = train_config
        self.feat_builder = LGBMFeatureBuilder(feature_config)
        
        self.progress_cb = progress_cb or (lambda p, t: None)
        self.state_cb = state_cb or (lambda s: None)
        self.log_cb = log_cb or (lambda lvl, msg: logger.log(lvl, msg))
        
        self._stop_requested = False
        self._start_time = 0.0

    def _log(self, level: str, msg: str):
        try:
            self.log_cb(level, msg)
        except Exception:
            logger.log(level, msg)

    def request_stop(self):
        self._stop_requested = True
        self._log("WARNING", " 收到停止指令，LightGBM 将在当前迭代后终止...")

    def _check_environment(self, save_dir: Path):
        """环境预检下沉：彻底解除对 run_pipeline 的依赖"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        total, used, free = shutil.disk_usage(save_dir)
        free_gb = free / (1024 ** 3)
        if free_gb < self.train_cfg.min_free_gb:
            raise IOError(f"磁盘空间不足: 剩余 {free_gb:.2f} GB，需要 {self.train_cfg.min_free_gb} GB")
            
        test_file = save_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise PermissionError(f"无写入权限: {save_dir}")

    def _lgbm_callback(self, env: lgb.callback.CallbackEnv):
        """LightGBM 原生回调拦截器：注入 UI 状态与中断信号"""
        if self._stop_requested:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
            
        train_loss = 0.0
        valid_loss = 0.0
        try:
            if env.evaluation_result_list and len(env.evaluation_result_list) > 0:
                train_loss = env.evaluation_result_list[0][2]
            if env.evaluation_result_list and len(env.evaluation_result_list) > 1:
                valid_loss = env.evaluation_result_list[1][2]
        except (IndexError, TypeError):
            pass 
        
        state = LGBMTrainState(
            iteration=env.iteration + 1,
            total_iterations=self.train_cfg.n_estimators,
            train_loss=train_loss,
            valid_loss=valid_loss,
            elapsed_time=time.time() - self._start_time
        )
        self.state_cb(state)
        
        progress = (env.iteration + 1) / self.train_cfg.n_estimators * 100
        self.progress_cb(progress, f"Iter {env.iteration+1}/{self.train_cfg.n_estimators} | Valid MAE: {valid_loss:.3f}")

    def _safe_to_dataframe(self, X: Any, feature_cols: list) -> pd.DataFrame:
        """安全转换：防止已有 DataFrame 被强行覆盖列名导致特征错位"""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=feature_cols)

    def train_and_save(self, lgbm_data: dict, save_dir: str) -> Dict[str, Any]:
        """执行训练、评估与模型持久化"""
        self._stop_requested = False
        self._start_time = time.time()
        save_path = Path(save_dir)
        
        self._log("INFO", "⏳ 正在进行环境预检 (磁盘与权限)...")
        try:
            self._check_environment(save_path)
        except Exception as e:
            self._log("ERROR", f"❌ 环境预检失败: {e}")
            return {"success": False, "error": str(e)}

        self._log("INFO", f"--- 开始训练 LightGBM (Max {self.train_cfg.n_estimators} Rounds) ---")
        self.progress_cb(0.0, "开始训练...")
        
        X_train, X_test = lgbm_data['X_train'], lgbm_data['X_test']
        y_train, y_test = lgbm_data['y_train'], lgbm_data['y_test']
        
        # 优先使用数据管道传来的特征名，兜底使用 feat_builder 的期望列
        feature_cols = lgbm_data.get('feature_names', self.feat_builder.expected_columns)
        
        df_train = self._safe_to_dataframe(X_train, feature_cols)
        df_test = self._safe_to_dataframe(X_test, feature_cols)
        
        # 修复 NaN 黑洞：强制 fillna 并转换为 float32
        X_train_aligned = self.feat_builder.align_columns(df_train).fillna(0.0).astype(np.float32).values
        X_test_aligned = self.feat_builder.align_columns(df_test).fillna(0.0).astype(np.float32).values
        
        train_data = lgb.Dataset(X_train_aligned, label=y_train, feature_name=feature_cols)
        test_data = lgb.Dataset(X_test_aligned, label=y_test, feature_name=feature_cols, reference=train_data)
        
        # 参数清洗：剥离不属于 LightGBM 原生 params 字典的控制参数
        clean_params = {k: v for k, v in self.train_cfg.lgbm_params.items() 
                        if k not in ['n_estimators', 'early_stop_rounds', 'log_every']}
        
        model = None
        try:
            model = lgb.train(
                clean_params, 
                train_data, 
                num_boost_round=self.train_cfg.n_estimators, 
                valid_sets=[train_data, test_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.train_cfg.early_stop_rounds),
                    lgb.log_evaluation(self.train_cfg.log_every),
                    self._lgbm_callback  
                ]
            )
        except Exception as e:
            self._log("ERROR", f"❌ 训练过程发生未知异常: {e}")
            return {"success": False, "error": str(e)}

        # 【Bug 1 修复】LightGBM 内部会吞掉 EarlyStopException 并正常返回 model，
        # 所以 except EarlyStopException 是死代码。改为在返回后检查 _stop_requested 标志。
        if self._stop_requested:
            self._log("WARNING", " 训练已被用户手动终止。")

        # 【Bug 2 修复】原条件 model.best_iteration <= 0 会误杀有效模型：
        #   - best_iteration 是 0-based，首轮即为最佳时值为 0
        #   - 未触发早停时某些版本 best_iteration 也可能为 0
        # 改用 current_iteration() 判断模型是否至少训练了 1 轮
        if model is None or model.current_iteration() <= 0:
            self._log("ERROR", " 训练未产生有效模型 (可能首轮即触发早停或数据异常)")
            return {"success": False, "error": "未产生有效模型"}

        # 【Bug 3 修复】best_iteration 是 0-based 索引，而 predict/save_model 的
        # num_iteration 参数表示"使用前 N 棵树"。best_iteration=99 意味着最佳是第100棵树，
        # 应传 100 而非 99，否则会漏掉最后一棵树。
        num_trees = model.best_iteration + 1 if model.best_iteration > 0 else model.current_iteration()

        preds = model.predict(X_test_aligned, num_iteration=num_trees)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        self._log("SUCCESS", f"✅ 测试集评估: MAE = {mae:.3f}, RMSE = {rmse:.3f}")
        
        #  严格对齐 config.yaml 中的专属 LGBM 路径
        # save_dir 已经由入口传入为 models/lgbm，因此这里不能再额外拼一个 /lgbm
        lgbm_model_dir = save_path
        lgbm_model_dir.mkdir(parents=True, exist_ok=True)
        model_path = lgbm_model_dir / CFG.paths.lgbm_model_file
        cols_path = lgbm_model_dir / "lgbm_feature_cols.joblib"
        
        model.save_model(str(model_path), num_iteration=num_trees)
        # 🚀 持久化特征列名，确保推理时能正确对齐特征维度
        joblib.dump(feature_cols, cols_path)
        logger.info(f"特征列名已保存至: {cols_path}")
        
        self.progress_cb(100.0, "训练完成！")
        return {
            "success": True, "model": model, "mae": mae, "rmse": rmse, 
            "model_path": str(model_path), "feature_cols": feature_cols
        }

# ==========================================
# 3. 纯算法层：双模型共识验证器
# ==========================================
def validate_dual_model_consensus(
    lgbm_prediction: Union[float, np.ndarray, dict], 
    tft_prediction: Union[float, np.ndarray, dict], 
    threshold: float = None,
    use_relative: bool = False 
) -> dict:
    """
    防御性共识验证器：自动解析 dict/array/float，
    新增：读取 config.yaml 中的融合权重进行加权平均，而非无脑 /2
    """
    # 动态获取阈值，若未传则从 CFG 读取
    if threshold is None:
        threshold = getattr(CFG.training, 'divergence_threshold', 10.0)

    def _extract_scalar(val: Any, name: str) -> float:
        if isinstance(val, dict):
            if 'prediction' in val:
                return float(np.array(val['prediction']).flatten()[0])
            raise KeyError(f"{name} 字典中缺少 'prediction' 键")
        elif isinstance(val, (list, tuple, np.ndarray)):
            return float(np.array(val).flatten()[0])
        else:
            return float(val)

    try:
        lgbm_val = _extract_scalar(lgbm_prediction, "LGBM")
        tft_val = _extract_scalar(tft_prediction, "TFT")
    except Exception as e:
        return {
            "is_consensus": False, "final_prediction": None, "divergence": -1.0,
            "message": f"❌ 预测值解析失败: {e}"
        }

    diff = abs(lgbm_val - tft_val)
    
    if use_relative:
        denominator = max(abs(lgbm_val), abs(tft_val), 1e-6)
        divergence_metric = (diff / denominator) * 100.0 
        metric_name = "相对误差%"
    else:
        divergence_metric = diff
        metric_name = "绝对差值"

    if divergence_metric > threshold:
        return {
            "is_consensus": False, "final_prediction": None, "divergence": round(divergence_metric, 4),
            "message": f"模型分歧过大 ({metric_name}:{divergence_metric:.2f})，建议切换人工复核模式"
        }
    else:
        # 🚀 【手术 3】读取 config.yaml 中的融合权重
        w_lgbm = getattr(CFG.model, 'model_fusion_weight_lgbm', 0.5)
        w_tft = getattr(CFG.model, 'model_fusion_weight_tft', 0.5)
        
        # 归一化权重 (防止用户配置加起来不等于 1)
        total_w = w_lgbm + w_tft
        w_lgbm_norm = w_lgbm / total_w if total_w > 0 else 0.5
        w_tft_norm = w_tft / total_w if total_w > 0 else 0.5
        
        fused = (lgbm_val * w_lgbm_norm) + (tft_val * w_tft_norm)
        
        return {
            "is_consensus": True, 
            "final_prediction": round(fused, 4), 
            "divergence": round(divergence_metric, 4),
            "message": f"双模型共识度高 ({metric_name}:{divergence_metric:.2f})，加权融合预测可信"
        }

# ==========================================
# 4. CLI 专属适配器 (配置驱动示范)
# ==========================================
if __name__ == "__main__":
    # 抛弃 yaml.safe_load，直接使用全局 CFG 对象
    try:
        train_cfg = LGBMTrainConfig(
            n_estimators=CFG.training.lgbm_n_estimators,
            early_stop_rounds=CFG.training.lgbm_early_stop_rounds,
            log_every=CFG.training.lgbm_log_every,
            divergence_threshold=CFG.training.divergence_threshold,
            # 兼容 Pydantic / Dict / Box
            lgbm_params=CFG.training.lgbm_params.model_dump() if hasattr(CFG.training.lgbm_params, 'model_dump') else dict(CFG.training.lgbm_params)
        )
        logger.info(f"✅ 成功从 CFG 加载 LGBM 配置 (Estimators: {train_cfg.n_estimators})")
    except Exception as e:
        logger.warning(f"⚠️ 读取 CFG 失败 ({e})，使用默认硬编码配置")
        train_cfg = LGBMTrainConfig(n_estimators=100, early_stop_rounds=20)

    # Mock 数据与特征配置
    mock_feature_cols = ['flow', 'inf_cod', 'inf_nh3', 'do_meas', 'kla_meas', 
                         'flow_lag_1', 'inf_cod_lag_1', 'inf_nh3_lag_1', 'do_meas_lag_1', 'kla_meas_lag_1',
                         'hour', 'dayofweek']
    
    X_train_mock = np.random.randn(1000, len(mock_feature_cols))
    X_train_mock[0, 5:10] = np.nan 
    
    mock_lgbm_data = {
        'X_train': X_train_mock,
        'X_test': np.random.randn(200, len(mock_feature_cols)),
        'y_train': np.random.randn(1000),
        'y_test': np.random.randn(200),
        'feature_names': mock_feature_cols
    }
    
    feat_cfg = FeatureConfig() 
    
    # 使用 CFG 中的专属 LGBM 目录
    save_dir = str(Path(CFG.paths.model_dir) / "lgbm")
    
    engine = LGBMTrainEngine(
        train_config=train_cfg,
        feature_config=feat_cfg,
        progress_cb=lambda p, t: logger.info(f"[{p:.1f}%] {t}"),
        state_cb=lambda s: logger.info(f"Iter {s.iteration} | Valid MAE: {s.valid_loss:.4f}"),
        log_cb=lambda lvl, msg: logger.log(lvl, msg)
    )
    
    result = engine.train_and_save(mock_lgbm_data, save_dir=save_dir)
    
    if result['success']:
        logger.success(f"🎉 训练成功 | MAE: {result['mae']:.3f}")
        
        # 测试共识验证器 (手术 3)
        mock_tft_dict = {"prediction": 38.0, "weights_mean": [0.1, 0.2, 0.3, 0.4]}
        
        consensus_abs = validate_dual_model_consensus(35.0, mock_tft_dict) # 自动读取 CFG.training.divergence_threshold (3.0)
        logger.info(f"绝对误差共识: {consensus_abs['message']}")
        
        consensus_rel = validate_dual_model_consensus(2.1, {"prediction": 2.5}, threshold=10.0, use_relative=True)
        logger.info(f"相对误差共识: {consensus_rel['message']}")