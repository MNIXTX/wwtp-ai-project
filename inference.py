# inference.py
import sys

if sys.platform == 'win32':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try: s.reconfigure(encoding='utf-8')
            except Exception: pass

import numpy as np
import pandas as pd
import json
import os
import glob
import time
import threading
from typing import Optional, Dict, Any, Generator, List
from collections import deque
from loguru import logger

from config_manager import CFG
from asm1_ode_solver import ASM1Solver, ASM1Parameters, ReactorConfig
from asm1_ppo_env import WWTPControlEnv
# 🚀 【核心修改 1】引入统一的特征构建器，彻底消灭重复代码和特征 Bug
from lgbm_feature_builder import LGBMFeatureBuilder

# ==================== 0. 全局兼容性与路径基准 ====================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable) 
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ModelManager:
    """自动扫描目录，加载最新模型（按修改时间排序）"""
    def __init__(self, model_dir: Optional[str] = None):
        base_model_dir = model_dir or CFG.paths.model_dir
        self.model_dir = base_model_dir if os.path.isabs(base_model_dir) else os.path.join(BASE_DIR, base_model_dir)
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            logger.warning(f"⚠️ 模型目录不存在，已自动创建: {self.model_dir}")

    def get_latest_model(self, extensions: list) -> Optional[str]:
        candidates = []
        for ext in extensions:
            pattern = os.path.join(self.model_dir, f"*{ext}")
            candidates.extend(glob.glob(pattern))
        
        if not candidates:
            return None
            
        latest = max(candidates, key=os.path.getmtime)
        logger.info(f"🔍 找到最新模型: {os.path.basename(latest)}")
        return latest

class DataDrivenModels:
    """智能加载 LGBM/TFT 模型（自动检测最新版本并构建真实推理特征）"""
    def __init__(self, model_manager: ModelManager):
        self.lgbm_loaded = False
        self.tft_loaded = False
        
        # 🚀 【核心修改 2】使用 deque 替代频繁的 pd.concat，大幅提升高频推理性能
        self.max_history_needed = max(max(CFG.lgbm.features.lag_hours), max(CFG.lgbm.features.rolling_windows)) + 10
        self.history_buffer = deque(maxlen=self.max_history_needed)
        
        self.w_lgbm = getattr(CFG.model, 'model_fusion_weight_lgbm', 0.5)
        self.w_tft = getattr(CFG.model, 'model_fusion_weight_tft', 0.5)
        
        # 1. 尝试加载 LGBM 模型
        lgbm_path = model_manager.get_latest_model(['.txt']) 
        if lgbm_path:
            try:
                import lightgbm as lgb
                self.lgbm_model = lgb.Booster(model_file=lgbm_path)
                self.lgbm_loaded = True
                logger.success(f"✅ LGBM 真实模型加载成功 (期望特征数: {self.lgbm_model.num_feature()})")
            except Exception as e:
                logger.error(f"❌ LGBM 加载失败: {e}，将降级为 Mock")

        # 2. 尝试加载 TFT 模型 (ONNX 格式)
        tft_path = model_manager.get_latest_model(['.onnx'])
        if tft_path:
            try:
                import onnxruntime as ort
                self.tft_session = ort.InferenceSession(tft_path, providers=['CPUExecutionProvider'])
                self.tft_loaded = True
                logger.success("✅ TFT ONNX 模型加载成功 (强制使用 CPU 推理)")
            except Exception as e:
                logger.error(f"❌ TFT 加载失败: {e}，将降级为 Mock")

        if not self.lgbm_loaded and not self.tft_loaded:
            logger.info("💡 未找到 LGBM/TFT 真实模型，已激活【智能 Mock 残差补偿】模式")

    def update_history(self, sensor_data: Dict[str, float]):
        """更新历史缓冲区（轻量级字典追加，避免频繁创建 DataFrame）"""
        lgbm_cols = list(CFG.lgbm.features.feature_columns)
        tft_cols = list(CFG.model.tft_feature_names)
        all_needed_cols = list(set(lgbm_cols + tft_cols))
        
        new_row = {col: float(sensor_data.get(col, 0.0)) for col in all_needed_cols}
        self.history_buffer.append(new_row)

    def _build_lgbm_features(self) -> Optional[np.ndarray]:
        """
        🚀 【核心修改 3】彻底废弃手工 for 循环，全面复用 LGBMFeatureBuilder
        确保 Lag/Rolling/Time 特征与训练时 100% 咬合，且修复了时间坍塌 Bug。
        """
        if len(self.history_buffer) < max(CFG.lgbm.features.lag_hours):
            return None 
            
        # 将 deque 转换为 DataFrame
        df = pd.DataFrame(list(self.history_buffer))
        
        # 调用统一的特征构建器 (is_inference=True 会触发修复后的时间推断逻辑)
        builder = LGBMFeatureBuilder(CFG.lgbm.features)
        df_feat = builder.build(df, is_inference=True)

        # 强制对齐列顺序
        df_aligned = builder.align_columns(df_feat)
        
        # 取最后一行作为当前推理输入
        X = df_aligned.iloc[[-1]].values.astype(np.float32)
        
        # 校验特征数量
        if self.lgbm_loaded and X.shape[1] != self.lgbm_model.num_feature():
            logger.error(f"❌ LGBM 特征数不匹配: 构建 {X.shape[1]}, 期望 {self.lgbm_model.num_feature()}")
            return None
            
        return X

    def predict_residual(self) -> np.ndarray:
        """
        Predict effluent COD using ML models.
        [Fix] Returns 1-value (eff_cod prediction) instead of forcing 3-value output.
        The 3-value pattern was a design error — ML models predict COD, not state residuals.
        """
        lgbm_result = None
        if self.lgbm_loaded:
            try:
                X_lgbm = self._build_lgbm_features()
                if X_lgbm is not None:
                    pred = self.lgbm_model.predict(X_lgbm).flatten()
                    lgbm_result = float(pred[0])
            except Exception as e:
                logger.warning(f"LGBM inference error: {e}")

        tft_result = None
        if self.tft_loaded:
            try:
                seq_len = CFG.model.tft_seq_len
                num_features = CFG.model.tft_num_features
                tft_cols = list(CFG.model.tft_feature_names)

                df_tft = pd.DataFrame(list(self.history_buffer)).tail(seq_len)
                for col in tft_cols:
                    if col not in df_tft.columns:
                        df_tft[col] = 0.0
                df_tft = df_tft[tft_cols]

                if len(df_tft) == 0:
                    # Cold start: buffer is empty, use zero-filled placeholder
                    df_tft = pd.DataFrame(np.zeros((seq_len, len(tft_cols)), dtype=np.float32), columns=tft_cols)
                elif len(df_tft) < seq_len:
                    pad_values = df_tft.iloc[0].values
                    pad_df = pd.DataFrame([pad_values] * (seq_len - len(df_tft)), columns=df_tft.columns)
                    df_tft = pd.concat([pad_df, df_tft], ignore_index=True)

                X_tft = df_tft.values.astype(np.float32).reshape(1, seq_len, num_features)
                input_name = self.tft_session.get_inputs()[0].name
                ort_outputs = self.tft_session.run(None, {input_name: X_tft})
                tft_result = float(ort_outputs[0].flatten()[0])
            except Exception as e:
                logger.warning(f"TFT ONNX inference error: {e}")

        # [Fix] Return the ML-predicted COD value (scalar in numpy array for API compat)
        if lgbm_result is not None and tft_result is not None:
            w_l = self.w_lgbm / (self.w_lgbm + self.w_tft)
            return np.array([w_l * lgbm_result + (1 - w_l) * tft_result])
        elif lgbm_result is not None:
            return np.array([lgbm_result])
        elif tft_result is not None:
            return np.array([tft_result])
        else:
            # Mock fallback
            return np.array([0.0])

class FusionInferenceEngine:
    """核心引擎：ASM1 (机理) + 数据驱动 (残差补偿) + PPO (决策)"""
    def __init__(self, model_dir: Optional[str] = None):
        self._lock = threading.Lock()
        
        self.model_manager = ModelManager(model_dir)
        # 🚀 从配置文件注入反应器体积，确保 ASM1 与 config.yaml 中的 asm1.volume 一致
        reactor_cfg = ReactorConfig(
            volume=getattr(CFG.asm1, 'volume', 5000.0),
            S_O_sat=getattr(CFG.asm1, 'saturation_do', 9.0),
        )
        self.asm1_solver = ASM1Solver(ASM1Parameters(), reactor=reactor_cfg)
        self.data_models = DataDrivenModels(self.model_manager)
        
        self.ppo_model = None
        self.use_ppo = False
        ppo_path = self.model_manager.get_latest_model(['.zip'])
        
        if ppo_path:
            try:
                logger.info(f"🤖 加载最新 PPO 策略: {os.path.basename(ppo_path)}")
                from stable_baselines3 import PPO
                self.ppo_model = PPO.load(ppo_path, device="cpu")
                self.use_ppo = True
                logger.success("✅ PPO 策略加载成功 (已优化为 CPU 推理)")
            except ImportError:
                logger.warning("⚠️ 未安装 stable-baselines3，无法加载 PPO 模型")
            except Exception as e:
                logger.error(f"❌ PPO 加载失败: {e}，将使用启发式策略")
        else:
            logger.warning("⚠️ 未找到 PPO 模型 (.zip)，将使用启发式策略")
        
        try:
            self.env = WWTPControlEnv()
        except Exception as e:
            logger.error(f"❌ WWTPControlEnv 初始化失败: {e}，推理引擎可能无法正常工作！")
            self.env = None

    def _heuristic_policy(self, obs: np.ndarray) -> np.ndarray:
        """启发式基准策略 (Rule-based)，作为 PPO 失效时的最后防线"""
        rules = getattr(CFG, 'rl', None)
        if rules and hasattr(rules, 'heuristic_rules'):
            h_rules = rules.heuristic_rules
            cod_in = float(obs[6]) if len(obs) > 6 else 0
            if cod_in > getattr(h_rules, 'high_cod_threshold', 400):
                return np.array(getattr(h_rules, 'action_high', [120.0, 1.0]))
        return np.array([80.0, 0.8]) 

    def run_control_cycle(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """执行一个完整的控制周期 (具备工业级 Fail-Safe 故障安全机制与线程锁)"""
        with self._lock:
            try:
                self.data_models.update_history(sensor_data)
                
                state_map = getattr(CFG.sensors, 'state_mapping', None)
                def get_sensor_val(mapped_key: str, default_val: float) -> float:
                    actual_key = getattr(state_map, mapped_key, mapped_key) if state_map else mapped_key
                    return float(sensor_data.get(actual_key, default_val))

                # 🚀 【跨文件手术 1A】修复物理量纲穿越：SCADA 的 eff_cod 是总 COD，必须乘以 fraction 转为溶解性 COD (S_S)
                raw_eff_cod = get_sensor_val('S_S', 45.0) 
                s_s_fraction = getattr(CFG.sensors, 'S_S_fraction_of_eff_cod', 0.15)
                current_S_S = raw_eff_cod * s_s_fraction 

                current_state = np.array([
                    current_S_S,                      # S_S (溶解性 COD, 约 5~15)
                    get_sensor_val('S_NH', 4.5),      # S_NH (氨氮)
                    get_sensor_val('S_O', 2.0),       # S_O (溶解氧)
                    get_sensor_val('X_H', 3000.0),    # X_H (异养菌/MLSS)
                    150.0                             # X_A (自养菌)
                ], dtype=np.float32)
                
                Q_in = float(sensor_data.get('Q_in', getattr(CFG.asm1, 'default_flow', 10000.0)))
                # 🚀 进水总 COD 转换为溶解性 COD (S_S)，与出水端 S_S_fraction 保持一致
                raw_cod_in = float(sensor_data.get('COD_in', getattr(CFG.asm1, 'default_cod_in', 350.0)))
                influent_soluble_fraction = 0.50  # 进水溶解性比例通常 40%~60%
                S_S_in = raw_cod_in * influent_soluble_fraction
                S_NH_in = float(sensor_data.get('NH3_in', getattr(CFG.asm1, 'default_nh3_in', 30.0)))
                
                if self.env is None:
                    raise RuntimeError("WWTPControlEnv 未成功初始化")

                self.env.state = current_state
                self.env.Q_in = Q_in
                self.env.S_S_in = S_S_in
                self.env.S_NH_in = S_NH_in
                
                current_Kla, current_R = 80.0, 0.8
                # 注意：反应器体积由 solver.reactor.volume 提供 (来自 config.yaml asm1.volume)
                result = self.asm1_solver.solve(
                    t_span_days=(0, 1/24), y0=current_state,
                    Q_in=Q_in * (1 + current_R),
                    S_S_in=S_S_in, S_NH_in=S_NH_in, S_O_in=1.5,
                    KLa=current_Kla, R=current_R, dt_hours=1.0
                )
                if result.get('success', False):
                    asm1_pred_state = result['y'][-1]
                    # 🚀 检查 NaN 传播：若求解器返回 NaN，回退到当前状态
                    if np.any(np.isnan(asm1_pred_state)):
                        logger.warning("ASM1 求解返回 NaN 状态，回退到当前观测状态")
                        asm1_pred_state = current_state.copy()
                else:
                    logger.warning(f"ASM1 ODE 求解失败: {result.get('message', 'Unknown')}，回退到当前观测状态")
                    asm1_pred_state = current_state.copy()
                
                # [Fix] Get ML-predicted effluent COD (single scalar value)
                data_pred = self.data_models.predict_residual()
                ml_cod_pred = float(data_pred[0]) if len(data_pred) > 0 else None

                # ASM1 provides the full 5-state vector. ML models only predict effluent COD.
                # Use ASM1 for state estimation (physics-based), ML COD prediction as monitoring reference.
                fused_pred_state = asm1_pred_state.copy()

                self.env.state = fused_pred_state
                obs_for_ppo = self.env._get_obs(current_Kla, current_R)

                if self.use_ppo and self.ppo_model is not None:
                    action, _ = self.ppo_model.predict(obs_for_ppo, deterministic=True)
                else:
                    action = self._heuristic_policy(obs_for_ppo)

                target_KLa, target_R = self.env._map_action(action)

                return {
                    "timestamp": sensor_data.get("timestamp", "manual_input"),
                    "control_actions": {
                        "target_KLa": round(float(target_KLa), 2),
                        "target_R_ratio": round(float(target_R), 2),
                        "blower_frequency": round(float(target_KLa) / 3.5, 1),
                        "pump_flow_rate": round(float(Q_in * target_R), 1)
                    },
                    "predictions": {
                        "pred_COD_out": round(float(fused_pred_state[0]), 2),
                        "pred_NH3_out": round(float(fused_pred_state[1]), 2),
                        "pred_DO_out": round(float(fused_pred_state[2]), 2),
                        "ml_cod_prediction": round(ml_cod_pred, 2) if ml_cod_pred is not None else None,
                    },
                    "system_status": "PPO_ACTIVE" if self.use_ppo else "HEURISTIC_FALLBACK",
                    "models_active": {
                        "PPO": self.use_ppo,
                        "LGBM": self.data_models.lgbm_loaded,
                        "TFT": self.data_models.tft_loaded
                    },
                    "status_code": "OK"
                }
                
            except Exception as e:
                logger.error(f"🚨 推理周期发生严重异常: {e}。已触发 Fail-Safe 机制，下发安全默认指令！")
                return {
                    "timestamp": sensor_data.get("timestamp", "manual_input"),
                    "control_actions": {
                        "target_KLa": 80.0,      
                        "target_R_ratio": 0.8,   
                        "blower_frequency": 22.9,
                        "pump_flow_rate": 8000.0
                    },
                    "predictions": {"error": str(e)},
                    "system_status": "FAIL_SAFE_MODE",
                    "status_code": "ERROR"
                }

# ==================== 3. 交互式手动输入模块 (仅限 CLI 测试) ====================
def get_float_input(prompt: str, default_val: float) -> float:
    while True:
        try:
            user_input = input(f"{prompt} [默认: {default_val}]: ").strip()
            if user_input == '':
                return default_val
            return float(user_input)
        except ValueError:
            print("⚠️ 输入无效，请输入数字！")
        except EOFError:
            return default_val 

def interactive_manual_stream() -> Generator[Dict[str, float], None, None]:
    print("\n" + "="*50)
    print("📥 进入【手动数据输入模式】")
    print("💡 提示：直接按回车键 (Enter) 将使用括号内的默认值。")
    print("🛑 提示：在任何输入框中输入 'q' 或 'quit' 可退出程序。")
    print("="*50 + "\n")
    
    cycle_count = 1
    while True:
        print(f"\n--- 🔄 第 {cycle_count} 次推理控制周期 ---")
        
        try:
            q_in_str = input(f"🌊 进水流量 Q_in (m³/d) [默认: 10000]: ").strip()
        except EOFError:
            break
            
        if q_in_str.lower() in ['q', 'quit', 'exit']:
            print("\n👋 收到退出指令，正在安全关闭推理引擎...")
            break
            
        try:
            Q_in = float(q_in_str) if q_in_str else 10000.0
        except ValueError:
            Q_in = 10000.0

        COD_in = get_float_input("🧪 进水 COD (mg/L)", 350.0)
        NH3_in = get_float_input("🧪 进水氨氮 NH3 (mg/L)", 30.0)
        
        print("\n(以下为反应器内部状态，直接回车使用默认值)")
        COD_reactor = get_float_input("  - 反应器 COD (mg/L)", 45.0)
        NH3_reactor = get_float_input("  - 反应器氨氮 (mg/L)", 4.5)
        DO_reactor = get_float_input("  - 反应器 DO (mg/L)", 2.0)
        MLSS_reactor = get_float_input("  - 反应器 MLSS (mg/L)", 3000.0)

        sensor_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime()),
            "Q_in": Q_in, "COD_in": COD_in, "NH3_in": NH3_in,
            "COD_reactor": COD_reactor, "NH3_reactor": NH3_reactor,
            "DO_reactor": DO_reactor, "MLSS_reactor": MLSS_reactor, "X_A_reactor": 150.0
        }
        
        yield sensor_data
        cycle_count += 1

# ==================== 4. 主程序 ====================
def main():
    logger.info("="*70)
    logger.info("🏭 WWTP AI 融合推理引擎 (ASM1 + Data-Driven + PPO) 启动")
    logger.info("="*70)
    
    engine = FusionInferenceEngine()
    logger.info("\n✅ 引擎初始化完毕，等待用户输入传感器数据...\n")
    
    for sensor_data in interactive_manual_stream():
        logger.info(f"📡 接收到数据 | 进水COD: {sensor_data['COD_in']} | 进水氨氮: {sensor_data['NH3_in']}")
        start_time = time.time()
        command = engine.run_control_cycle(sensor_data)
        elapsed_time = (time.time() - start_time) * 1000 
        
        print("\n" + "-"*50)
        if command.get("status_code") == "OK":
            logger.success(f"📤 AI 推理完成 (耗时: {elapsed_time:.1f} ms) | 下发控制指令:")
        else:
            logger.error(f"📤 触发安全降级 (耗时: {elapsed_time:.1f} ms) | 下发保守指令:")
            
        print(json.dumps(command, indent=4, ensure_ascii=False))
        print("-"*50 + "\n")

if __name__ == "__main__":
    main()