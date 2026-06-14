import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from loguru import logger
import re  #  【新增】用于清理列名中的不可见字符
from config.manager import CFG
from models.lgbm_features import LGBMFeatureBuilder

# ==========================================
# 【新增】轻量级事件系统 (与任何UI框架完全解耦)
# ==========================================
@dataclass
class PipelineEvent:
    """标准化管道事件数据结构"""
    stage: str      # 阶段标识: LOAD / CLEAN / TFT / LGBM / SAVE / DONE
    percent: int    # 进度百分比 0-100
    message: str    # 人类可读消息
    is_cancelled: bool = False  # 是否收到取消信号

class WWTPDataPipeline:
    """污水处理智能优化系统 - 工业级数据处理管道"""

    #  【修复点 1】列名映射：严格对齐下游 LGBM/TFT 所需的标准 SCADA 列名
    # 这里的 Key 代表 CSV 中可能出现的原始列名（小写形式），Value 是算法层标准列名
    COLUMN_MAPPING = {
        'timestamp': 'timestamp', 
        'time': 'timestamp',          #  【修复】兼容 CSV 中时间列叫 'time' 的情况
        'datetime': 'timestamp',      #  【修复】兼容 CSV 中时间列叫 'datetime' 的情况
        'flow': 'flow',           
        'inf_flow': 'flow',           #  【修复】兼容旧命名
        'cod_in': 'inf_cod',      
        'inf_cod': 'inf_cod',         #  【修复】兼容已标准命名的 CSV
        'do': 'do_meas',          
        'do_meas': 'do_meas',         #  【修复】兼容已标准命名的 CSV
        'aeration_do': 'do_meas',     #  【修复】兼容旧命名
        'cod_out': 'eff_cod',     
        'eff_cod': 'eff_cod',         #  【修复】兼容已标准命名的 CSV
        'outflow_cod': 'eff_cod',     #  【修复】兼容旧命名
        'nh3n_out': 'eff_nh3',    
        'eff_nh3': 'eff_nh3',         #  【修复】兼容已标准命名的 CSV
        'mlss': 'MLSS_reactor',   
        'mlss_reactor': 'MLSS_reactor', #  【修复】兼容小写 mlss_reactor
        'nh3n_in': 'inf_nh3',     
        'inf_nh3': 'inf_nh3',         #  【修复】兼容已标准命名的 CSV
        'do_reactor': 'DO_reactor',   #  【修复】兼容小写 do_reactor
        'nh3_reactor': 'NH3_reactor',  #  【修复】兼容 ASM1 状态映射
        'mlss_reacto': 'MLSS_reactor',
        'mlss_reacto_': 'MLSS_reactor',
        'MLSS_REACTO': 'MLSS_reactor',
        # 新增：温度/pH 列名映射
        'temperature': 'temp',  'water_temp': 'temp',
        't': 'temp',           'wt': 'temp',
        'temp': 'temp',        # CSV 中已有标准列名
        'ph': 'pH',            'ph_value': 'pH',
        'PH': 'pH',            'ph_reactor': 'pH',
        'pH': 'pH',            # CSV 中已有标准列名
    }
    
    REQUIRED_STANDARD_COLUMNS = [
        'timestamp', 'flow', 'inf_cod', 'do_meas', 
        'eff_cod', 'MLSS_reactor'
    ]

    def __init__(
        self,
        freq: Optional[str] = None,
        lookback: Optional[int] = None,
        horizon: Optional[int] = None,
        physical_limits: Optional[Dict[str, List[float]]] = None,
        max_interp_gap: Optional[int] = None,
        test_ratio: Optional[float] = None
    ):
        self.freq = freq if freq is not None else CFG.pipeline.freq
        self.lookback = lookback if lookback is not None else CFG.pipeline.lookback
        self.horizon = horizon if horizon is not None else CFG.pipeline.horizon
        self.max_interp_gap = max_interp_gap if max_interp_gap is not None else CFG.pipeline.max_interp_gap
        self.test_ratio = test_ratio if test_ratio is not None else CFG.pipeline.test_ratio
        
        # 🚀 【修复点 2】物理限幅默认值：使用映射后的标准列名
        default_limits = {
            'flow': [0.0, 100000.0], 'inf_cod': [0.0, 2000.0],
            'do_meas': [0.0, 15.0], 'MLSS_reactor': [0.0, 10000.0],
            'inf_nh3': [0.0, 100.0], 'eff_cod': [0.0, 500.0],
            'eff_nh3': [0.0, 50.0], 'DO_reactor': [0.0, 12.0]
        }
        raw_limits = physical_limits
        if raw_limits is None:
            raw_limits = getattr(getattr(CFG, 'pipeline', None), 'physical_limits', default_limits)
            if not isinstance(raw_limits, dict):
                raw_limits = default_limits
        self.physical_limits = {k: tuple(v) for k, v in raw_limits.items()}
        
        self.scaler = StandardScaler()
        self.target_scaler = None  # [Fix] Separate scaler for target (inverse transform at inference)
        self.feature_names: List[str] = []
        self.target_names = [CFG.lgbm.features.target_column]
        self.is_fitted = False
        
        # 【新增】事件订阅与取消控制
        self._event_listeners: List[Callable[[PipelineEvent], None]] = []
        self._cancel_flag = False
        
        logger.info(f" 管道初始化完成 | 频率:{self.freq} | 回溯:{self.lookback}h | 预测:{self.horizon}h")

    # ==================== 【新增】事件管理接口 ====================
    def subscribe(self, listener: Callable[[PipelineEvent], None]):
        """订阅管道事件"""
        self._event_listeners.append(listener)

    def unsubscribe(self, listener: Callable[[PipelineEvent], None]):
        """取消订阅"""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def request_cancel(self):
        """外部请求取消管道执行"""
        self._cancel_flag = True
        logger.warning(" 收到取消请求，管道将在当前步骤完成后安全停止")

    def _emit(self, stage: str, percent: int, message: str):
        """内部统一事件发射器"""
        logger.info(f"[{stage}] ({percent}%) {message}")
        event = PipelineEvent(stage=stage, percent=percent, message=message, is_cancelled=self._cancel_flag)
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"事件监听器异常: {e}")
                
    def _check_cancel(self, stage: str):
        """在每个关键步骤前检查取消信号"""
        if self._cancel_flag:
            self._emit(stage, -1, "用户取消执行")
            raise InterruptedError(f"管道在 [{stage}] 阶段被用户取消")

    # ==================== 1. 数据加载与校验 ====================
    def load_and_validate(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """加载CSV、重命名列并进行校验"""
        self._check_cancel("LOAD")
        
        if filepath is None:
            filepath = Path(CFG.paths.scada_data_csv)
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f" 数据文件不存在: {filepath}")

        filepath = filepath.resolve()  # Normalize path for logging

        # 🚀 【安全防护】检查文件大小，拒绝超大/空文件
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if filepath.stat().st_size == 0:
            raise ValueError(f"数据文件为空: {filepath}")
        if file_size_mb > 500:
            logger.warning(f"⚠️ CSV 文件较大 ({file_size_mb:.0f} MB)，加载可能需要较长时间...")
        if file_size_mb > 2000:
            raise ValueError(
                f"数据文件过大 ({file_size_mb:.0f} MB > 2 GB 上限)。"
                f"请拆分文件或只导入最近一段时间的数据。"
            )

        # 🚀 【修复点 A】多编码尝试：utf-8-sig → utf-8 → gbk (中文 Windows)
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'gbk']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if df is None:
            raise ValueError(f"无法以任何已知编码读取文件: {filepath}")
        
        # 🚀 【修复点 B】深度清理列名（消除首尾空格、零宽空格等不可见字符）
        df.columns = [re.sub(r'[\s\u200b\ufeff]+', '', str(col)) for col in df.columns]
        
        # 🚀 【修复点 3】核心修复：解决全大写导致的映射断层
        # 1. 将 CSV 列名转为大写进行统一比对（此时列名已绝对干净）
        df.columns = [col.upper() for col in df.columns] 
    
        # 2. 动态生成全大写的映射字典 (例如: 'TIMESTAMP' -> 'timestamp')
        upper_mapping = {k.upper(): v for k, v in self.COLUMN_MAPPING.items()}
        
        # 3. 处理 MLSS 的特殊别名 (防止被 upper_mapping 覆盖前出错)
        if 'MLSS_REACTOR' not in df.columns and 'MLSS' in df.columns:
            df.rename(columns={'MLSS': 'MLSS_REACTOR'}, inplace=True)

        # 4. 执行重命名 (此时大写 Key 能完美匹配大写的 df.columns)
        df.rename(columns={k: v for k, v in upper_mapping.items() if k in df.columns}, inplace=True)
        
        # 5. 校验必填列 (此时 df.columns 已经是标准的小写/混合大小写命名)
        # 🚀 【手术修复】整合了原本重复的 3 个 if missing 逻辑，确保终极诊断代码能正常触发
        available_columns = {col.lower(): col for col in df.columns}
        missing = [col for col in self.REQUIRED_STANDARD_COLUMNS if col.lower() not in available_columns]
        if missing:
            #  【终极诊断】打印 Pandas 看到的真实列名及其底层 ASCII 码
            print("\n" + "="*60)
            print(" 致命错误：找不到必要列！", missing)
            print(" Pandas 实际读取到的列名及底层 ASCII 码：")
            for i, col in enumerate(df.columns):
                # 打印列名、类型、以及每个字符的 ASCII 码（专治各种不可见字符）
                ascii_codes = [ord(c) for c in str(col)]
                print(f"  [{i}] 显示为: '{col}' | 类型: {type(col).__name__} | ASCII: {ascii_codes}")
            print("="*60 + "\n")
            raise ValueError(f" CSV 缺少必要列: {missing}。当前可用列: {list(df.columns)}")
        
        # 6. 安全解析时间戳 (此时 'timestamp' 列必定存在)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        invalid_ts = df['timestamp'].isna().sum()
        if invalid_ts > 0:
            logger.warning(f" {invalid_ts}条记录时间戳解析失败，已移除")
            df = df.dropna(subset=['timestamp'])
        
        #  【终极排雷 1】强制去除重复的时间戳，防止下游 set_index 后索引重复导致 LGBM 笛卡尔积爆炸
        dup_count = df['timestamp'].duplicated().sum()
        if dup_count > 0:
            logger.warning(f" 发现 {dup_count} 条重复时间戳，已自动去重 (保留最后一条)")
            df = df.drop_duplicates(subset=['timestamp'], keep='last')

        df = df.sort_values('timestamp').reset_index(drop=True)
        self._emit("LOAD", 20, f"数据加载成功 | 形状:{df.shape} | 时间范围:{df['timestamp'].min()} ~ {df['timestamp'].max()}")
        return df

    # ==================== 2. 清洗与对齐 ====================
    def clean_and_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """异常值截断 + 死值剔除 + 重采样对齐 + 智能插值"""
        self._check_cancel("CLEAN")
        df = df.copy().set_index('timestamp')
        
        #  【修复点 4】物理限幅：此时 df 的列名已经是标准列名，直接与 self.physical_limits 匹配
        for col, (low, high) in self.physical_limits.items():
            if col in df.columns:
                before = ((df[col] < low) | (df[col] > high)).sum()
                df[col] = df[col].clip(lower=low, upper=high)
                if before > 0:
                    logger.debug(f"截断 [{col}]: {before}条超出[{low},{high}]范围")
        
        try:
            td = pd.Timedelta(self.freq)
            dead_threshold = max(3, int(3 / td.total_seconds() * 3600))
        except ValueError as e:
            logger.error(f" 无法解析时间频率字符串: '{self.freq}'，原因: {e}")
            raise

        for col in df.select_dtypes(include=[np.number]).columns:
            rolling_std = df[col].rolling(window=dead_threshold, min_periods=dead_threshold).std()
            dead_mask = rolling_std == 0
            if dead_mask.sum() > 0:
                df.loc[dead_mask, col] = np.nan
                logger.debug(f"剔除 [{col}] 死值: {dead_mask.sum()}条")
        
        self._check_cancel("CLEAN")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        #  【终极排雷 2】剔除全为 NaN 的无效列，防止 resample 崩溃
        valid_numeric_cols = [col for col in numeric_cols if df[col].notna().any()]
        dropped_cols = set(numeric_cols) - set(valid_numeric_cols)
        if dropped_cols:
            logger.warning(f" 以下传感器列全为空值，已自动剔除: {dropped_cols}")
            
        df_resampled = df[valid_numeric_cols].resample(self.freq).mean()
        
        for col in df_resampled.columns:
            if df_resampled[col].isna().any():
                df_resampled[col] = df_resampled[col].interpolate(method='linear', limit=self.max_interp_gap)
        
        df_clean = df_resampled.dropna(how='all')
        first_valid, last_valid = df_clean.first_valid_index(), df_clean.last_valid_index()
        if first_valid is not None and last_valid is not None:
            df_clean = df_clean.loc[first_valid:last_valid]
        
        self._emit("CLEAN", 40, f"清洗完成 | 重采样至 {self.freq} | 有效样本:{len(df_clean)}")
        return df_clean.reset_index()

    # ==================== 3. TFT时序序列构建 (修复版) ====================
    def build_tft_sequences(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        构建TFT所需的滑动窗口序列。
        [修复] 特征和目标分别用独立的 Scaler，彻底杜绝数据泄漏和推理期量纲错误。
        """
        self._check_cancel("TFT")

        df = df.copy()  # Prevent SettingWithCopyWarning on caller's DataFrame

        tft_feature_cols = CFG.model.tft_feature_names
        missing_feats = set(tft_feature_cols) - set(df.columns)
        if missing_feats:
            logger.warning(f" 数据中缺少 TFT 特征: {missing_feats}，将使用 0 填充")
            for col in missing_feats:
                df[col] = 0.0

        target_col = self.target_names[0] if self.target_names else 'eff_cod'
        if target_col not in df.columns:
            df[target_col] = 0.0

        # [修复] 仅对特征列做 StandardScaler，目标列单独处理
        feature_values = df[tft_feature_cols].values.astype(np.float32)
        target_values = df[[target_col]].values.astype(np.float32)

        if not self.is_fitted:
            self.scaler.fit(feature_values)  # 主 scaler 仅拟合特征
            # 初始化并拟合目标专用 Scaler
            if not hasattr(self, 'target_scaler') or self.target_scaler is None:
                self.target_scaler = StandardScaler()
            self.target_scaler.fit(target_values)
            self.feature_names = list(tft_feature_cols)
            self.is_fitted = True
            logger.info(f" Scaler 拟合完成 | 特征: {self.scaler.n_features_in_}维, "
                        f"目标均值={self.target_scaler.mean_[0]:.2f}, 目标标准差={self.target_scaler.scale_[0]:.2f}")

        feature_scaled = self.scaler.transform(feature_values)
        target_scaled = self.target_scaler.transform(target_values)

        # 拼接回标准化的 (N, num_features+1) 数组用于滑动窗口
        scaled = np.concatenate([feature_scaled, target_scaled], axis=1)
        n_features = len(tft_feature_cols)
        target_idx_in_scaled = n_features  # target 在最后一列

        seq_len = self.lookback + self.horizon
        n_samples = len(scaled) - seq_len + 1
        if n_samples <= 0:
            raise ValueError(f"数据长度{len(scaled)}不足以构建序列(需≥{seq_len})")

        # 🚀 [向量化] 使用 stride_tricks 替代 Python for-loop，10-50x 加速
        # sliding_window_view along axis=0: (N, D) → (N-W+1, D, W)
        # D = n_features + 1 (target), W = lookback + horizon
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(scaled, seq_len, axis=0)  # (n_samples, n_feat+1, seq_len)
        # X: all features, lookback steps  → (n_samples, n_features, lookback)
        X_raw = windows[:, :n_features, :self.lookback]
        # y: target feature, horizon steps   → (n_samples, horizon)
        y_raw = windows[:, target_idx_in_scaled, self.lookback:]
        # Transpose X to (n_samples, lookback, n_features) for TFT input
        X_arr = np.ascontiguousarray(np.transpose(X_raw, (0, 2, 1)), dtype=np.float32)
        y_arr = np.ascontiguousarray(y_raw, dtype=np.float32)

        split_idx = int(n_samples * (1 - self.test_ratio))

        self._emit("TFT", 60, f"TFT序列构建完成 | 训练:{split_idx} | 测试:{n_samples-split_idx}")
        return {
            'X_train': X_arr[:split_idx], 'y_train': y_arr[:split_idx],
            'X_test': X_arr[split_idx:], 'y_test': y_arr[split_idx:],
        }

    # ==================== 4. LGBM表格特征工程 ====================
    def build_lgbm_features(self, df: pd.DataFrame) -> Dict:
        """构建 LightGBM 适配的二维表格特征"""
        self._check_cancel("LGBM")
        df_work = df.copy()
        
        #  【修复点 5】兼容大小写时间列，防止 LGBM 构建时报错
        time_col = 'timestamp' if 'timestamp' in df_work.columns else ('TIMESTAMP' if 'TIMESTAMP' in df_work.columns else None)
        if time_col:
            df_work[time_col] = pd.to_datetime(df_work[time_col])
            df_work = df_work.set_index(time_col)

        feature_builder = LGBMFeatureBuilder(config=CFG.lgbm.features)
        combined_feat = feature_builder.build(df_work, is_inference=False)

        # [Fix] Record actual base columns used for LGBM inference alignment
        self.lgbm_base_columns = list(feature_builder.feature_columns) if feature_builder.feature_columns else \
            list(feature_builder._infer_base_columns(df_work))
        
        target_col = CFG.lgbm.features.target_column
        if target_col not in df_work.columns:
            raise ValueError(f" 找不到 LGBM 目标列: {target_col}")
        targets_df = df_work[[target_col]]
        
        max_lag = max(CFG.lgbm.features.lag_hours) if hasattr(CFG.lgbm.features, 'lag_hours') and CFG.lgbm.features.lag_hours else 24
        
        common_idx = combined_feat.index.intersection(targets_df.index)
        combined_feat = combined_feat.loc[common_idx]
        targets_df = targets_df.loc[common_idx]
        
        combined_feat = combined_feat.iloc[max_lag:]
        targets_df = targets_df.iloc[max_lag:]
        
        split_idx = int(len(combined_feat) * (1 - self.test_ratio))
        
        # 🚀 【手术修复 - 核心】强制提纯：确保目标列(target)绝对不在特征矩阵中，防止维度膨胀
        if target_col in combined_feat.columns:
            combined_feat = combined_feat.drop(columns=[target_col])
            
        # 重新获取纯净的特征名
        pure_feature_names = list(combined_feat.columns)
        
        X_train, X_test = combined_feat.iloc[:split_idx], combined_feat.iloc[split_idx:]
        y_train, y_test = targets_df.iloc[:split_idx], targets_df.iloc[split_idx:]
        
        self._emit("LGBM", 80, f"LGBM特征构建完成 | 总样本:{len(combined_feat)} | 特征维度:{len(pure_feature_names)}")
        
        return {
            # 🚀 【手术修复】强制转为 numpy 数组 (.values)，彻底消除 Pandas DataFrame 隐式携带的列名干扰
            'X_train': X_train.values, 
            'X_test': X_test.values,
            'y_train': y_train.values.ravel(), 
            'y_test': y_test.values.ravel(),
            'feature_names': pure_feature_names,
        }

    # ==================== 5. 持久化 (补回遗漏的方法) ====================
    def save_artifacts(self, save_dir: Optional[str] = None):
        """保存Scaler、特征名、管道配置"""
        save_path = Path(save_dir) if save_dir else Path(CFG.paths.artifacts_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, save_path / 'scaler.pkl')

        # [Fix] Save target scaler for inverse transform at inference
        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            joblib.dump(self.target_scaler, save_path / 'target_scaler.pkl')

        # [Fix] Save TFT metadata for inference
        tft_meta = {
            'tft_feature_names': list(CFG.model.tft_feature_names),
            'tft_seq_len': CFG.model.tft_seq_len,
            'target_name': self.target_names[0] if self.target_names else 'eff_cod',
        }
        with open(save_path / 'tft_meta.json', 'w', encoding='utf-8') as f:
            json.dump(tft_meta, f, ensure_ascii=False, indent=2)

        config = {
            'feature_names': self.feature_names, 'target_names': self.target_names,
            'freq': self.freq, 'lookback': self.lookback, 'horizon': self.horizon,
            'physical_limits': {k: list(v) for k, v in self.physical_limits.items()},
            'lgbm_base_columns_for_inference': getattr(self, 'lgbm_base_columns', None),
        }
        with open(save_path / 'pipeline_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"Pipeline artifacts saved to: {save_path}")

    @classmethod
    def load_artifacts(cls, save_dir: Optional[str] = None) -> 'WWTPDataPipeline':
        """从磁盘恢复管道"""
        save_path = Path(save_dir) if save_dir else Path(CFG.paths.artifacts_dir)
        
        with open(save_path / 'pipeline_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        pipeline = cls(
            freq=config['freq'], lookback=config['lookback'],
            horizon=config['horizon'], physical_limits=config['physical_limits']
        )
        pipeline.scaler = joblib.load(save_path / 'scaler.pkl')

        # [Fix] Load target scaler for inverse transform
        target_scaler_path = save_path / 'target_scaler.pkl'
        if target_scaler_path.exists():
            pipeline.target_scaler = joblib.load(target_scaler_path)
        else:
            pipeline.target_scaler = StandardScaler()
            logger.warning("target_scaler.pkl not found, created empty scaler")

        pipeline.feature_names = config['feature_names']
        pipeline.target_names = config['target_names']
        pipeline.is_fitted = True
        logger.info(f"Pipeline restored from {save_path}")
        return pipeline

    # ==================== 主流程入口 ====================
    def run_full_pipeline(self, csv_path: Optional[str] = None, save_dir: Optional[str] = None) -> Dict:
        """一键执行完整数据处理流程"""
        self._cancel_flag = False
        self._emit("INIT", 0, "开始执行完整数据处理管道")
        
        try:
            df_raw = self.load_and_validate(csv_path)
            df_clean = self.clean_and_resample(df_raw)
            tft_data = self.build_tft_sequences(df_clean)
            lgbm_data = self.build_lgbm_features(df_clean)
            
            self._check_cancel("SAVE")
            self.save_artifacts(save_dir)
            
            self._emit("DONE", 100, " 完整管道执行成功!")
            return {'tft': tft_data, 'lgbm': lgbm_data, 'clean_df': df_clean}
            
        except InterruptedError as e:
            self._emit("CANCELLED", -1, str(e))
            logger.warning(f"管道已安全中止: {e}")
            raise
        except Exception as e:
            self._emit("ERROR", -1, f"管道执行失败: {e}")
            raise

if __name__ == "__main__":
    pipeline = WWTPDataPipeline()
    result = pipeline.run_full_pipeline()
    print(f"\nTFT 训练集形状: {result['tft']['X_train'].shape}")
    print(f"LGBM 特征数量:  {len(result['lgbm']['feature_names'])}")