# ui/core/service_adapter.py
"""
SystemAdapter — Streamlit UI 与底层算法之间的安全桥接层
- 单例管理 (预测网关 / 数据管道)
- 系统健康检查
- 看板数据聚合
- 配置文件读写 (原子操作 + 热重载)
- 后台训练任务管理
"""
import sys
import os
import subprocess
import time
import tempfile
import shutil
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
from ruamel.yaml import YAML

# ---- 路径引导 (一次性) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- ruamel.yaml 实例 (用于保留注释的 YAML 读写) ----
yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.indent(mapping=2, sequence=4, offset=2)

# ---- 导入底层模块 (延迟 + 降级) ----
CFG = None
load_config = None
reload_config = None
WaterQualityPredictor = None
WWTPDataPipeline = None

try:
    from config_manager import CFG, load_config as _load_cfg, reload_config as _reload_cfg
    load_config = _load_cfg
    reload_config = _reload_cfg
except ImportError:
    pass

try:
    from predictor_adapter import WaterQualityPredictor as _WQP
    WaterQualityPredictor = _WQP
except ImportError:
    pass

try:
    from data_pipeline import WWTPDataPipeline as _DP
    WWTPDataPipeline = _DP
except ImportError:
    pass


# ============================================================
class SystemAdapter:
    """
    系统服务适配器。
    所有方法均为 @classmethod，作为全局统一网关。
    """

    # ---- 单例缓存 ----
    _gateway_instance = None
    _pipeline_instance = None

    # ==========================================
    # 1. 单例管理
    # ==========================================
    @classmethod
    def get_gateway(cls):
        """获取预测网关单例 (线程安全由 GIL 保证)"""
        if cls._gateway_instance is None:
            if WaterQualityPredictor is None:
                raise RuntimeError("WaterQualityPredictor module not found")
            cls._gateway_instance = WaterQualityPredictor()
        return cls._gateway_instance

    @classmethod
    def get_pipeline(cls):
        """获取数据管道单例"""
        if cls._pipeline_instance is None:
            if WWTPDataPipeline is None:
                raise RuntimeError("WWTPDataPipeline module not found")
            cls._pipeline_instance = WWTPDataPipeline()
        return cls._pipeline_instance

    # ==========================================
    # 2. 系统健康检查 (纯数据，无 Streamlit 副作用)
    # ==========================================
    @classmethod
    def get_system_health(cls) -> Dict[str, Any]:
        """
        返回系统健康状态字典。
        [Fix] 使用单例而非每次 new 新实例。
        [Fix] 不再直接调用 st.sidebar.*, 由调用方自行渲染。
        """
        health = {
            "gateway": "offline",
            "pipeline": "offline",
            "tft_ok": False,
            "lgbm_ok": False,
            "gateway_error": None,
            "pipeline_error": None,
        }

        # --- 预测网关 ---
        try:
            if WaterQualityPredictor is not None:
                gateway = cls.get_gateway()
                health["tft_ok"] = Path(gateway.tft_path).exists()
                health["lgbm_ok"] = Path(gateway.lgbm_path).exists()
                health["gateway"] = "online" if (health["tft_ok"] or health["lgbm_ok"]) else "offline"
        except Exception as e:
            health["gateway_error"] = str(e)

        # --- 数据管道 ---
        try:
            if WWTPDataPipeline is not None:
                cls.get_pipeline()  # 验证可实例化
                health["pipeline"] = "online"
        except Exception as e:
            health["pipeline_error"] = str(e)

        return health

    # ==========================================
    # 3. 看板数据聚合 (缓存 30 秒, 避免高频 Rerun 轰炸底层)
    # ==========================================
    @classmethod
    @st.cache_data(ttl=30)
    def get_dashboard_data(_cls) -> Dict[str, Any]:
        result = {"scada_df": pd.DataFrame(), "predictions": {}, "model_metrics": {}}

        # --- SCADA 数据 ---
        try:
            scada_path = PROJECT_ROOT / "data" / "scada_data.csv"
            if scada_path.exists():
                result["scada_df"] = pd.read_csv(scada_path)
        except Exception as e:
            result["scada_error"] = str(e)

        # --- AI 预测 ( [Fix] 构造正确的 Dict[str, List[float]] 格式 ) ---
        try:
            gateway = _cls.get_gateway()
            scada_df = result["scada_df"]
            if not scada_df.empty:
                seq_len = getattr(getattr(CFG, 'model', None), 'tft_seq_len', 24)
                feat_names = getattr(getattr(CFG, 'model', None), 'tft_feature_names', [])

                # 取最近 seq_len 行构建历史字典
                recent = scada_df.tail(seq_len)
                history_data = {}
                for col in feat_names:
                    if col in recent.columns:
                        history_data[col] = recent[col].tolist()
                    else:
                        history_data[col] = [0.0] * seq_len

                # LGBM 额外列
                for extra in ["flow", "do_meas", "eff_cod", "eff_nh3"]:
                    if extra not in history_data and extra in recent.columns:
                        history_data[extra] = recent[extra].tolist()

                pred_res = gateway.predict(history_data)
                result["predictions"] = pred_res.get("predictions", {})
                result["model_metrics"] = {
                    "inference_time_ms": pred_res.get("inference_time_ms", 0),
                    "status": pred_res.get("status", "unknown"),
                }
        except Exception as e:
            result["prediction_error"] = str(e)

        return result

    @classmethod
    def get_realtime_scada_data(cls) -> pd.DataFrame:
        data = cls.get_dashboard_data()
        return data.get("scada_df", pd.DataFrame())

    # ==========================================
    # 4. 配置管理 (原子写入 + 热重载)
    # ==========================================
    @classmethod
    @st.cache_data(ttl=60)
    def get_current_config(_cls) -> Dict[str, Any]:
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml_rt.load(f)
                return dict(data) if data else {}
        except Exception:
            return {}

    @classmethod
    def update_and_reload_config(cls, section: str, updates: Dict[str, Any]):
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml_rt.load(f) or {}

        if section not in config_data:
            config_data[section] = {}

        for k, v in updates.items():
            if k in config_data[section]:
                config_data[section][k] = v
            else:
                st.warning(f"Ignored unknown field [{section}.{k}]")

        # 原子写入
        dir_name = config_path.parent
        fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.yaml')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_f:
                yaml_rt.dump(config_data, tmp_f)
            if config_path.exists():
                config_path.unlink()
            shutil.move(temp_path, config_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        cls._reload_cfg()
        cls._invalidate_caches()

    @classmethod
    def force_overwrite_config(cls, raw_yaml_str: str):
        # 预校验
        try:
            parsed = yaml_rt.load(raw_yaml_str)
            if not isinstance(parsed, dict):
                raise ValueError("YAML must be a dict")
        except Exception as e:
            raise ValueError(f"YAML validation failed: {e}")

        config_path = PROJECT_ROOT / "config.yaml"
        dir_name = config_path.parent
        fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.yaml')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_f:
                tmp_f.write(raw_yaml_str)
            if config_path.exists():
                config_path.unlink()
            shutil.move(temp_path, config_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        cls._reload_cfg()
        cls._invalidate_caches()

    @classmethod
    def reset_all_instances(cls):
        cls._gateway_instance = None
        cls._pipeline_instance = None
        cls.get_current_config.clear()
        cls.get_dashboard_data.clear()

    @classmethod
    def _reload_cfg(cls):
        """内部：触发热重载"""
        if reload_config is not None:
            try:
                reload_config()
            except Exception:
                pass

    @classmethod
    def _invalidate_caches(cls):
        """内部：重置单例与 Streamlit 缓存"""
        cls._gateway_instance = None
        cls._pipeline_instance = None
        cls.get_current_config.clear()

    # ==========================================
    # 5. 后台训练任务管理
    # ==========================================
    @classmethod
    def trigger_training_task(cls, task_type: str, params: Dict[str, Any]) -> str:
        """
        启动后台训练进程。
        [Fix] 根据 task_type 选择正确的训练脚本 (而非统一用 run_pipeline.py)
        """
        script_map = {
            "lgbm": "run_lgbm.py",
            "tft": "train_tft.py",
            "ppo": "train.py",
        }
        script_name = script_map.get(task_type)
        if not script_name:
            raise ValueError(f"Unknown task_type: {task_type}")

        script_path = PROJECT_ROOT / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")

        cmd = [sys.executable, str(script_path)]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])

        task_id = f"{task_type}_{int(time.time())}"
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{task_id}.log"

        with open(log_file, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,  # 浏览器关闭后继续运行
            )

        if "background_tasks" not in st.session_state:
            st.session_state.background_tasks = {}

        st.session_state.background_tasks[task_id] = {
            "pid": process.pid,
            "status": "running",
            "task_type": task_type,
            "log_file": str(log_file),
            "start_time": time.time(),
            "params": params,
        }
        return task_id

    @classmethod
    def check_task_status(cls, task_id: str) -> Dict[str, Any]:
        tasks = st.session_state.get("background_tasks", {})
        task_info = tasks.get(task_id)
        if task_info is None:
            return {"status": "unknown", "error": "Task ID not found"}

        if task_info["status"] in ("completed", "failed"):
            return task_info

        pid = task_info["pid"]
        log_file = task_info.get("log_file", "")

        # 检查进程是否存活
        is_running = False
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x100000, False, pid)
                if handle:
                    is_running = True
                    kernel32.CloseHandle(handle)
            else:
                os.kill(pid, 0)
                is_running = True
        except (OSError, Exception):
            is_running = False

        if is_running:
            return {"status": "running", "pid": pid}

        # 进程已结束，判定成功/失败
        try:
            time.sleep(0.3)
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                log_content = f.read()
        except Exception as e:
            log_content = f"Failed to read log: {e}"

        failed_markers = ("Traceback (most recent call last)", "Error:", "FATAL", "FAILED")
        is_failed = any(m in log_content for m in failed_markers)

        task_info["status"] = "failed" if is_failed else "completed"
        if is_failed:
            task_info["error"] = log_content
        else:
            task_info["output"] = log_content

        return task_info
