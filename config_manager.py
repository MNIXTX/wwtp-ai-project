# ==========================================
# config_manager.py (完整修复版 - Part 1 / 2)
# ==========================================
import os
import sys

# --- 修复 Windows 终端 UTF-8 乱码（必须在任何 loguru 输出之前执行）---
if sys.platform == 'win32':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8')
            except Exception:
                pass

import yaml
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Any
from loguru import logger
from pydantic import BaseModel, Field

# ==========================================
# Pydantic V1/V2 严谨兼容性处理
# ==========================================
try:
    import pydantic
    from pydantic import BaseModel, ValidationError, validator
    
    PYDANTIC_V2 = pydantic.VERSION.startswith('2.')
    
    if PYDANTIC_V2:
        from pydantic import ConfigDict, field_validator
    else:
        ConfigDict = None 
        field_validator = None
        
    if not PYDANTIC_V2:
        logger.warning(f"⚠️ 检测到 Pydantic V1 ({pydantic.VERSION})，建议升级至 V2 以获得最佳性能。")
        
except ImportError as e:
    raise ImportError(
        f"缺失核心依赖 Pydantic！请先运行: pip install pydantic。详细错误: {e}"
    ) from e

# ==========================================
# 全局项目根目录与用户数据目录
# ==========================================
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    PROJECT_ROOT = Path(__file__).parent.resolve()

if sys.platform == 'win32':
    USER_DATA_DIR = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'WWTP_AI'
else:
    USER_DATA_DIR = Path.home() / '.local' / 'share' / 'WWTP_AI'

# ==========================================
# 🚀 跨平台安全的 YAML PATH 序列化器
# ==========================================
def _path_representer(dumper, data):
    """将 Path 对象统一转换为 POSIX 风格的字符串 (正斜杠 /)，确保跨平台兼容"""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data.as_posix())

yaml.add_representer(Path, _path_representer)
yaml.add_representer(PurePosixPath, _path_representer)
yaml.add_representer(PureWindowsPath, _path_representer)

# ==========================================
# 配置数据结构定义 (Pydantic)
# ==========================================
class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    output_dir: Path = Path("outputs")
    artifacts_dir: Path = Path("artifacts")
    scada_data_csv: Path = Path("data/scada_data.csv")
    calibration_scada_csv: Path = Path("data/calibration_scada.csv")
    calibration_plot_path: Path = Path("outputs/calibration_plot.png")
    resume_model_path: Path = Path("models/ppo_wwtp_resume.zip")
    lgbm_model_dir: Path = Path("models/lgbm")
    lgbm_model_file: str = "lgbm_wwtp.txt"  # 👈 注意：这里是 str，不是 Path

class SensorsStateMapping(BaseModel):
    S_S: str = "eff_cod"       
    S_NH: str = "NH3_reactor"
    S_O: str = "DO_reactor"
    X_H: str = "MLSS_reactor"
    X_A: str = "X_A_reactor"

class SensorsConfig(BaseModel):
    state_mapping: SensorsStateMapping = SensorsStateMapping()
    history_fields: List[str] = [
        "inf_cod", "inf_nh3", "DO_reactor", "MLSS_reactor", 
        "flow", "do_meas", "eff_cod"                        
    ]
    S_S_fraction_of_eff_cod: float = 0.15 

class HeuristicRules(BaseModel):
    high_cod_threshold: float = 500.0
    action_high: List[float] = [0.6, 0.3]
    action_normal: List[float] = [0.0, 0.0]

class RLConfig(BaseModel):
    heuristic_rules: HeuristicRules = HeuristicRules()

class LGBMFeatureConfig(BaseModel):
    target_column: str = "eff_cod"      
    # 🚀 使用 Field(default_factory=list) 完美支持空列表，且符合 Pydantic 最佳实践
    feature_columns: List[str] = Field(default_factory=list) 
    rolling_feature_columns: List[str] = Field(default_factory=list)
    lag_hours: List[int] = Field(default_factory=lambda: [1, 3, 6, 12, 24])
    rolling_windows: List[int] = Field(default_factory=lambda: [6, 12, 24])

class LGBMDemoDataColumns(BaseModel):
    inf_cod: str = "inf_cod"
    flow: str = "flow"                  
    do_meas: str = "do_meas"            
    eff_cod: str = "eff_cod"            

class LGBMConfig(BaseModel):
    features: LGBMFeatureConfig = LGBMFeatureConfig()
    demo_samples: int = 8760
    demo_data_columns: LGBMDemoDataColumns = LGBMDemoDataColumns()

class ASM1Config(BaseModel):
    volume: float = 5000.0
    default_kla: float = 80.0
    default_flow: float = 10000.0
    default_cod_in: float = 300.0
    default_nh3_in: float = 30.0
    default_do_in: float = 1.5
    default_cod_limit: float = 50.0
    default_nh3_limit: float = 5.0
    saturation_do: float = 9.0
    Y_H: float = 0.67
    Y_A: float = 0.24
    i_XB: float = 0.086
    X_H_guess: float = 2500.0
    X_A_guess: float = 100.0
    mu_h: float = 4.0
    k_s: float = 20.0
    k_oh: float = 0.5
    k_no: float = 0.5
    b_h: float = 0.3
    mu_a: float = 0.8
    k_nh: float = 1.5
    k_oa: float = 0.8
    b_a: float = 0.05
    param_bounds_mu_H: List[float] = [2.0, 8.0]
    param_bounds_K_S: List[float] = [5.0, 60.0]
    param_bounds_K_OH: List[float] = [0.1, 1.0]
    param_bounds_mu_A: List[float] = [0.2, 1.5]
    param_bounds_K_NH: List[float] = [0.5, 5.0]
    param_bounds_K_OA: List[float] = [0.2, 1.5]
    rl_kla_bounds: List[float] = [10.0, 200.0]
    rl_w_aeration: float = 2.0
    rl_w_smooth: float = 1.0
    rl_penalty_cod: float = 5.0
    rl_penalty_nh3: float = 8.0
    rl_reward_clip: float = 100.0
    rl_r_bounds: List[float] = [0.3, 1.5]
    rl_kla_center: float = 80.0
    rl_r_center: float = 0.8
    rl_kla_scale: float = 60.0
    rl_r_scale: float = 0.4
    rl_flow_range: List[float] = [5000.0, 20000.0]
    rl_cod_in_range: List[float] = [150.0, 600.0]
    rl_nh3_in_range: List[float] = [15.0, 50.0]
    rl_init_state_ranges: List[List[float]] = [
        [20.0, 80.0], [2.0, 10.0], [1.0, 4.0], [1500.0, 3500.0], [50.0, 250.0]
    ]
    rl_biomass_warn: float = 1000.0
    rl_biomass_crash: float = 500.0
    rl_obs_high: float = 100000.0

class ModelConfig(BaseModel):
    tft_hidden_size: int = 64
    tft_seq_len: int = 24
    tft_num_features: int = 4
    tft_lstm_layers: int = 2
    tft_dropout: float = 0.1
    tft_batch_size: int = 128 
    ppo_learning_rate: float = 3e-4
    ppo_batch_size: int = 256
    ppo_n_steps: int = 2048
    ppo_gamma: float = 0.99
    ppo_episode_steps: int = 168
    tft_feature_names: List[str] = ["inf_cod", "inf_nh3", "DO_reactor", "MLSS_reactor"]
    onnx_opset_version: int = 18  # PyTorch 2.5+ 最低要求
    onnx_dynamic_axes: Dict[str, Dict[str, str]] = {
        "input_data": {0: "batch_size", 1: "sequence_length", 2: "num_features"},
        "prediction": {0: "batch_size"},
        "feature_weights": {0: "batch_size", 1: "sequence_length", 2: "num_features"}
    }
    device: str = "cpu"
    model_fusion_weight_asm1: float = 0.4
    model_fusion_weight_lgbm: float = 0.3
    model_fusion_weight_tft: float = 0.3
    lgbm_feature_order: List[int] = [6, 7, 8]
    pred_horizon: int = 24
    @property
    def lgbm_feature_names(self) -> List[str]:
        from config_manager import CFG  # <--- 这个 import 绝对不能移到文件顶部！
        return CFG.lgbm.features.feature_columns
    @property
    def lgbm_lag_times(self) -> List[int]:
        from config_manager import CFG
        return CFG.lgbm.features.lag_hours

class TrainingConfig(BaseModel):
    tft_epochs: int = 50
    ppo_timesteps: int = 500000
    ppo_n_envs: int = 4
    ppo_total_timesteps: int = 500000
    ppo_eval_freq: int = 1000
    divergence_threshold: float = 3.0 
    ppo_energy_weight: float = 10.0
    tft_lr: float = 1e-3
    lgbm_n_estimators: int = 1000         
    lgbm_early_stop_rounds: int = 50      
    lgbm_log_every: int = 100             
    
    class PPOParams(BaseModel):
        policy: str = "MlpPolicy"
        learning_rate_start: float = 3e-4
        learning_rate_end: float = 3e-6
        clip_range_start: float = 0.2
        clip_range_end: float = 0.05
        n_steps: int = 2048
        batch_size: int = 256
        n_epochs: int = 10
        gamma: float = 0.99
        gae_lambda: float = 0.95
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        device: str = "auto"
    
    ppo_params: PPOParams = PPOParams()
    lgbm_params: Dict[str, Any] = {
        "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, 
        "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, 
        "bagging_freq": 5, "verbose": -1
    }

class CalibrationConfig(BaseModel):
    de_popsize: int = 15
    de_maxiter: int = 50
    de_tol: float = 1e-4
    nh3_penalty_weight: float = 2.0
    
    class ScadaColumnMapping(BaseModel):
        flow: str = "flow"
        inf_cod: str = "inf_cod"
        inf_nh3: str = "inf_nh3"
        eff_cod: str = "eff_cod"
        eff_nh3: str = "eff_nh3"
        do_meas: str = "do_meas"
        kla_meas: str = "kla_meas"
    
    scada_column_mapping: ScadaColumnMapping = ScadaColumnMapping()

class PipelineConfig(BaseModel):
    freq: str = "1h"
    lookback: int = 24
    horizon: int = 24
    max_interp_gap: int = 6
    test_ratio: float = 0.2
    
    physical_limits: Dict[str, List[float]] = {
        'flow': [0.0, 100000.0], 
        'inf_cod': [0.0, 2000.0],     
        'eff_cod': [0.0, 500.0],      
        'inf_nh3': [0.0, 100.0],      
        'eff_nh3': [0.0, 50.0],       
        'do_meas': [0.0, 15.0],       
        'DO_reactor': [0.0, 12.0],
        'MLSS_reactor': [0.0, 10000.0], 
        'ph': [4.0, 10.0], 
        'temp': [0.0, 45.0]
    }
    
    # 🚀 【修复点 1】重写校验器，兼容 V1/V2，并采用白名单防止“自杀式”拦截
    if PYDANTIC_V2:
        @field_validator('physical_limits')
        @classmethod
        def validate_column_names(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
            allowed_keys = {
                'flow', 'inf_cod', 'eff_cod', 'inf_nh3', 'eff_nh3', 
                'do_meas', 'DO_reactor', 'MLSS_reactor', 'ph', 'temp'
            }
            for key in v:
                if key not in allowed_keys:
                    raise ValueError(f"❌ 物理限幅配置错误: 发现非法列名 '{key}'。白名单: {allowed_keys}")
            return v
    else:
        @validator('physical_limits')
        def validate_column_names(cls, v):
            allowed_keys = {
                'flow', 'inf_cod', 'eff_cod', 'inf_nh3', 'eff_nh3', 
                'do_meas', 'DO_reactor', 'MLSS_reactor', 'ph', 'temp'
            }
            for key in v:
                if key not in allowed_keys:
                    raise ValueError(f"❌ 物理限幅配置错误: 发现非法列名 '{key}'。白名单: {allowed_keys}")
            return v

# ==========================================
# 兼容 Pydantic V1/V2 的 AppConfig 组装
# ==========================================
if PYDANTIC_V2:
    class AppConfig(BaseModel):
        model_config = ConfigDict(validate_assignment=True)
        paths: PathsConfig = PathsConfig()
        asm1: ASM1Config = ASM1Config()
        model: ModelConfig = ModelConfig()
        training: TrainingConfig = TrainingConfig()
        calibration: CalibrationConfig = CalibrationConfig()
        pipeline: PipelineConfig = PipelineConfig()
        rl: RLConfig = RLConfig()
        sensors: SensorsConfig = SensorsConfig()
        lgbm: LGBMConfig = LGBMConfig()
else:
    class AppConfig(BaseModel):
        class Config:
            validate_assignment = True
        paths: PathsConfig = PathsConfig()
        asm1: ASM1Config = ASM1Config()
        model: ModelConfig = ModelConfig()
        training: TrainingConfig = TrainingConfig()
        calibration: CalibrationConfig = CalibrationConfig()
        pipeline: PipelineConfig = PipelineConfig()
        rl: RLConfig = RLConfig()
        sensors: SensorsConfig = SensorsConfig()
        lgbm: LGBMConfig = LGBMConfig()
# ==========================================
# config_manager.py (完整修复版 - Part 2 / 2)
# ==========================================

# ==========================================
# 配置加载与校验逻辑
# ==========================================
# 🚀 全局日志句柄追踪器，防止热重载时文件句柄泄漏
_file_log_handler_id = None

def setup_logger(log_dir: Path):
    """统一配置 loguru 日志系统 (带句柄生命周期管理)"""
    global _file_log_handler_id
    
    # 1. 控制台日志防重入
    if not hasattr(setup_logger, "_console_initialized"):
        logger.remove()
        logger.add(
            sys.stderr, 
            level="INFO", 
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}", 
            colorize=sys.stderr.isatty(),
            encoding="utf-8"  # ✅ 强制使用 UTF-8 编码，防止 Windows 控制台 GBK 乱码
        )
        setup_logger._console_initialized = True
        
    # 2. 文件日志：先移除旧的 Handler，再添加新的，防止句柄泄漏和文件锁死
    if _file_log_handler_id is not None:
        try:
            logger.remove(_file_log_handler_id)
        except ValueError:
            pass  # Handler 可能已经失效，忽略
            
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # 写入测试，提前暴露权限问题
        test_file = log_dir / ".write_test"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
        
        _file_log_handler_id = logger.add(
            log_dir / "app_{time:YYYY-MM-DD}.log", 
            rotation="00:00", 
            retention="30 days", 
            level="DEBUG",
            encoding="utf-8"
        )
    except (PermissionError, OSError) as e:
        logger.warning(f"⚠️ 无法在 {log_dir} 创建日志文件 (权限不足或磁盘满)，将仅输出到控制台。原因: {e}")
        _file_log_handler_id = None

def _resolve_paths(paths_cfg: PathsConfig) -> PathsConfig:
    """
    将 PathsConfig 中的相对路径强制转换为基于 PROJECT_ROOT 的绝对路径。
    🚀 【终极修复】引入“类型感知”，完美解决 str 与 Path 混合定义导致的 Pydantic V2 校验崩溃。
    """
    data = paths_cfg.model_dump() if PYDANTIC_V2 else paths_cfg.dict()
    
    # 动态获取模型字段的注解类型
    if PYDANTIC_V2:
        field_types = {name: field.annotation for name, field in PathsConfig.model_fields.items()}
    else:
        field_types = {name: field.outer_type_ for name, field in PathsConfig.__fields__.items()}

    for key, value in data.items():
        expected_type = field_types.get(key)
        
        # 🚀 【核心修复】如果字段定义为 str (如 lgbm_model_file)，强制保持字符串，绝不转为 Path
        if expected_type is str:
            data[key] = str(value)
            continue

        # 对于定义为 Path 的字段，进行标准的绝对路径解析
        p = Path(value) if isinstance(value, str) else value
        if isinstance(p, Path) and not p.is_absolute():
            resolved_path = PROJECT_ROOT / p
        else:
            resolved_path = p
            
        data[key] = resolved_path

    return PathsConfig(**data)

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """加载并校验配置文件 (带完美的权限回退机制)"""
    path = config_path if config_path else (PROJECT_ROOT / "config.yaml")
    
    # 确保早期控制台日志可用
    if not hasattr(setup_logger, "_console_initialized"):
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}", colorize=sys.stderr.isatty())
        setup_logger._console_initialized = True

    if not path.exists():
        logger.warning(f"⚠️ 配置文件 '{path}' 不存在，正在生成默认配置...")
        # mode='json' 在 V2 中会自动将 Path 转为字符串
        default_config = AppConfig().model_dump(mode='json') if PYDANTIC_V2 else AppConfig().dict()
        
        # 兼容 V1 的手动路径序列化
        if not PYDANTIC_V2 and 'paths' in default_config:
            for k, v in default_config['paths'].items():
                if isinstance(v, (str, Path)):
                    default_config['paths'][k] = Path(v).as_posix()
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"✅ 默认配置文件已创建: {path}")
        except (PermissionError, OSError):
            logger.warning(f"⚠️ 项目目录无写入权限，配置文件将生成在用户目录: {USER_DATA_DIR}")
            USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = USER_DATA_DIR / "config.yaml"
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f) or {}
        config = AppConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(f"配置文件 '{path.name}' 格式或类型校验失败，请检查 YAML 语法。详细错误:\n{e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件 '{path.name}' YAML 语法解析失败。详细错误:\n{e}") from e
    except Exception as e:
        raise RuntimeError(f"加载配置文件 '{path.name}' 时发生未知错误: {e}") from e

    # 🚀 注入类型感知后的绝对路径
    config.paths = _resolve_paths(config.paths)

    # 自动创建必要的目录
    dirs_to_create = [
        config.paths.data_dir, config.paths.model_dir, config.paths.log_dir, 
        config.paths.output_dir, config.paths.artifacts_dir, config.paths.lgbm_model_dir
    ]
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.warning(f"⚠️ 无法创建目录 {dir_path}: {e}")
        
    setup_logger(config.paths.log_dir)
    logger.info(f"✅ 配置文件 '{path.name}' 加载成功，日志与目录已就绪。")
    logger.debug(f"📂 项目根目录锚定为: {PROJECT_ROOT}")
    return config

def save_config(config: AppConfig, config_path: Optional[Path] = None):
    """将当前配置安全地保存回 YAML 文件 (Windows 安全的原子写入)"""
    path = config_path if config_path else (PROJECT_ROOT / "config.yaml")
    
    if PYDANTIC_V2:
        data = config.model_dump(mode='json') 
    else:
        data = config.dict()
        
    # 跨平台路径统一：强制将 paths 下的 Path 对象转为 POSIX 字符串
    if not PYDANTIC_V2 and 'paths' in data:
        for k, v in data['paths'].items():
            if isinstance(v, (str, Path)):
                data['paths'][k] = Path(v).as_posix()

    try:
        temp_path = path.with_suffix('.yaml.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # 🚀 Windows 安全的原子替换
        os.replace(str(temp_path), str(path))
        logger.info(f"✅ 配置已成功保存至: {path}")
    except (PermissionError, OSError) as e:
        logger.error(f"❌ 保存配置文件失败 (文件可能被其他程序占用): {e}")
        raise PermissionError(f"无法保存配置文件到 {path}，请关闭可能占用该文件的程序后重试。") from e

def reload_config(config_path: Optional[Path] = None) -> AppConfig:
    """热重载配置文件"""
    global CFG
    logger.info("🔄 正在热重载配置文件...")
    CFG = load_config(config_path)
    return CFG

# ==========================================
# 全局单例导出
# ==========================================
CFG = load_config()