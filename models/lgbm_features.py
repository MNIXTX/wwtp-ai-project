# lgbm_feature_builder.py
import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from loguru import logger


def _rolling_entropy(series: pd.Series, window: int) -> pd.Series:
    """Rolling signal entropy (complexity measure from wwtpai VSL).

    Measures the 'disorder' of the signal over the window using
    normalized histogram entropy. Higher entropy = more complex dynamics.
    """
    if window <= 2:
        return pd.Series(0.0, index=series.index)

    def _entropy(arr):
        arr = np.asarray(arr, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) < max(window // 2, 3):
            return 0.0
        # Normalize to [0,1] and compute histogram entropy
        rng = valid.max() - valid.min()
        if rng < 1e-10:
            return 0.0
        norm = (valid - valid.min()) / rng
        bins = min(10, len(valid) // 3)
        if bins < 2:
            return 0.0
        counts, _ = np.histogram(norm, bins=bins, range=(0, 1))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    return series.rolling(window, min_periods=max(window//2, 3)).apply(_entropy, raw=True)


def _rolling_crossrate(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean crossing rate (instability measure from wwtpai VSL).

    Counts how many times the signal crosses its rolling mean within the window.
    Higher crossing rate = more oscillation / instability.
    """
    if window <= 2:
        return pd.Series(0.0, index=series.index)

    def _crossrate(arr):
        arr = np.asarray(arr, dtype=np.float64)
        valid = ~np.isnan(arr)
        n_valid = valid.sum()
        if n_valid < max(window // 2, 3):
            return 0.0
        y = arr[valid]
        mean_val = y.mean()
        demeaned = y - mean_val
        crossings = np.sum(np.diff(np.signbit(demeaned)))
        return crossings / max(n_valid - 1, 1)

    return series.rolling(window, min_periods=max(window//2, 3)).apply(_crossrate, raw=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling trend slope: (current - lagged) / window.

    Ultra-fast O(1) vectorized computation using shift.
    Captures net change direction over the window — adequate for tree models.
    """
    if window <= 1:
        return pd.Series(0.0, index=series.index)
    slope = (series - series.shift(window)) / window
    return slope.fillna(0.0)

# ==========================================
# 1. 解耦核心：特征配置数据类 (彻底消灭全局 CFG)
# ==========================================
@dataclass
class FeatureConfig:
    """LGBM 特征构建配置"""
    # 🚀 【修改 1】默认值改为空列表，配合 config.yaml 触发“动态推断”
    feature_columns: List[str] = field(default_factory=list)
    # 🚀 【修改 2】新增滚动特征专属配置列，支持对特定列做滚动，空列表则默认对所有基础列做滚动
    rolling_feature_columns: List[str] = field(default_factory=list) 
    
    lag_hours: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 24])
    target_col: str = 'eff_cod'

# ==========================================
# 2. 工业级特征构建器 (实例化设计，支持状态缓存)
# ==========================================
class LGBMFeatureBuilder:
    """
    LGBM 特征构建器 (训练与推理共用)
    通过依赖注入 FeatureConfig，确保训练与推理维度 100% 咬合。
    """
    
    def __init__(self, config: FeatureConfig):
        self.cfg = config

        # 兼容两种配置来源：dataclass / Pydantic 模型
        self.feature_columns = list(getattr(config, 'feature_columns', []) or [])
        self.rolling_feature_columns = list(getattr(config, 'rolling_feature_columns', []) or [])
        self.lag_hours = list(getattr(config, 'lag_hours', [1, 3, 6, 12, 24]))
        self.rolling_windows = list(getattr(config, 'rolling_windows', [6, 12, 24]))
        self.target_col = getattr(config, 'target_col', getattr(config, 'target_column', 'eff_cod'))

        # 🚀 【修改 3】延迟初始化：如果配置为空（动态推断模式），初始化时无法预计算列名
        if self.feature_columns:
            self._expected_columns = self._compute_expected_columns(self.feature_columns)
            logger.info(f"✅ 特征构建器初始化 (静态模式) | 预期特征维度: {len(self._expected_columns)}")
        else:
            self._expected_columns = []
            logger.info("✅ 特征构建器初始化 (动态推断模式) | 等待首次 build() 以锁定特征维度...")
        
    def _compute_expected_columns(self, base_cols: List[str]) -> List[str]:
        """预计算模型预期的特征列名及严格顺序"""
        expected_cols = list(base_cols)

        for col in base_cols:
            for lag in self.lag_hours:
                expected_cols.append(f'{col}_lag_{lag}')

        # 滚动列逻辑对齐
        roll_cols = self.rolling_feature_columns if self.rolling_feature_columns else base_cols
        for col in roll_cols:
            for win in self.rolling_windows:
                expected_cols.append(f'{col}_roll_mean_{win}')
                # VSL (Variation Sliding Layer) — 捕捉变化幅度/趋势/不稳定性
                expected_cols.append(f'{col}_roll_range_{win}')   # 窗口内波动幅度
                expected_cols.append(f'{col}_roll_std_{win}')    # 短期不稳定性
                expected_cols.append(f'{col}_roll_slope_{win}')  # 变化趋势 (线性回归斜率)

        # 循环时间编码 (替换原始 hour/dayofweek 为 sin/cos 对)
        expected_cols.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                              'month_sin', 'month_cos'])

        # VSL entropy + mean_crossing_rate (来自 wwtpai Water Research 2023)
        for col in roll_cols:
            for win in self.rolling_windows:
                expected_cols.append(f'{col}_roll_entropy_{win}')
                expected_cols.append(f'{col}_roll_crossrate_{win}')

        # 交互特征 (领域知识)
        interaction_pairs = [
            ('flow', 'inf_cod'),     # 有机负荷率
            ('MLSS_reactor', 'DO_reactor'),  # 曝气-生物量耦合
            ('temp', 'pH'),          # 温度-pH 协同效应
        ]
        for a, b in interaction_pairs:
            if a in base_cols and b in base_cols:
                expected_cols.append(f'{a}_x_{b}')

        # 一阶差分特征 (变化率)
        for col in base_cols:
            expected_cols.append(f'{col}_delta_1h')

        return expected_cols

    def _infer_base_columns(self, df: pd.DataFrame) -> List[str]:
        """
        🚀 【新增核心】动态推断基础特征列 (Schema-on-Read)
        当配置文件中 feature_columns 为空时，自动从 DataFrame 中提取所有有效的数值列。
        """
        exclude_cols = {self.target_col}
        numeric_df = df.select_dtypes(include=[np.number])
        inferred_cols = [col for col in numeric_df.columns if col not in exclude_cols]
        
        if not inferred_cols:
            raise ValueError("❌ 动态推断失败：数据集中没有找到任何有效的数值型特征列！")
            
        logger.info(f"💡 [动态推断] 配置为空，自动从数据中提取了 {len(inferred_cols)} 个基础特征: {inferred_cols}")
        return inferred_cols

    def build(self, df: pd.DataFrame, is_inference: bool = False, freq_hours: float = 1.0) -> pd.DataFrame:
        """
        构建滞后、滚动、时间特征。
        """
        # 🚀 【修改 4】支持空列表动态推断
        actual_base_cols = self._infer_base_columns(df) if not self.feature_columns else self.feature_columns
        base_cols = [c for c in actual_base_cols if c in df.columns]
        
        if not base_cols:
            raise ValueError(f"❌ 数据中找不到任何 LGBM 基础特征列: {actual_base_cols}")
            
        feature_blocks = [df[base_cols]]
        
        # 1. 滞后特征 (Lag)
        lag_dict = {}
        for col in base_cols:
            for lag in self.lag_hours:
                lag_dict[f'{col}_lag_{lag}'] = df[col].shift(lag)
        if lag_dict:
            feature_blocks.append(pd.DataFrame(lag_dict, index=df.index))
            
        # 2. 滚动特征 (Rolling) + VSL 变化滑动层
        roll_dict = {}
        roll_cols = self.rolling_feature_columns if self.rolling_feature_columns else base_cols
        for col in roll_cols:
            if col in df.columns:
                roll_series = df[col]
                for win in self.rolling_windows:
                    rw = roll_series.rolling(win, min_periods=1)
                    roll_dict[f'{col}_roll_mean_{win}'] = rw.mean()
                    # VSL: 窗口内波动幅度 (max-min)
                    roll_dict[f'{col}_roll_range_{win}'] = rw.max() - rw.min()
                    # VSL: 窗口内标准差 (短期不稳定性)
                    roll_dict[f'{col}_roll_std_{win}'] = rw.std().fillna(0)
                    # VSL: 变化趋势 (线性回归斜率)
                    roll_dict[f'{col}_roll_slope_{win}'] = _rolling_slope(roll_series, win).fillna(0)
                    # VSL: 信号复杂度 (entropy) — 来自 wwtpai
                    roll_dict[f'{col}_roll_entropy_{win}'] = _rolling_entropy(roll_series, win).fillna(0)
                    # VSL: 均值穿越率 (mean crossing rate) — 来自 wwtpai
                    roll_dict[f'{col}_roll_crossrate_{win}'] = _rolling_crossrate(roll_series, win).fillna(0)
        if roll_dict:
            feature_blocks.append(pd.DataFrame(roll_dict, index=df.index))
            
        # 3. 循环时间编码 (Cyclical Encoding) — 替代原始 hour/dayofweek
        if not isinstance(df.index, pd.DatetimeIndex):
            if is_inference:
                now = pd.Timestamp.now()
                freq_min = max(1, int(freq_hours * 60))
                time_index = pd.date_range(end=now, periods=len(df), freq=f'{freq_min}min')
                hour_vals, dow_vals, month_vals = time_index.hour, time_index.dayofweek, time_index.month
            else:
                hour_vals = [0] * len(df)
                dow_vals = [0] * len(df)
                month_vals = [1] * len(df)
        else:
            hour_vals = df.index.hour
            dow_vals = df.index.dayofweek
            month_vals = df.index.month

        time_dict = {
            'hour_sin':   np.sin(2 * np.pi * hour_vals / 24),
            'hour_cos':   np.cos(2 * np.pi * hour_vals / 24),
            'dow_sin':    np.sin(2 * np.pi * dow_vals / 7),
            'dow_cos':    np.cos(2 * np.pi * dow_vals / 7),
            'month_sin':  np.sin(2 * np.pi * month_vals / 12),
            'month_cos':  np.cos(2 * np.pi * month_vals / 12),
        }
        feature_blocks.append(pd.DataFrame(time_dict, index=df.index))

        # 4. 交互特征 (领域知识驱动的特征交叉)
        interaction_pairs = [
            ('flow', 'inf_cod', 'flow_x_inf_cod'),           # 有机负荷率 (kg COD/h)
            ('MLSS_reactor', 'DO_reactor', 'MLSS_x_DO'),     # 曝气-生物量耦合
            ('temp', 'pH', 'temp_x_pH'),                      # 温度-pH 协同
        ]
        inter_dict = {}
        for a, b, name in interaction_pairs:
            if a in df.columns and b in df.columns:
                inter_dict[name] = df[a] * df[b]
        if inter_dict:
            feature_blocks.append(pd.DataFrame(inter_dict, index=df.index))

        # 5. 一阶差分特征 (变化率，捕捉趋势)
        delta_dict = {}
        for col in base_cols:
            if col in df.columns:
                delta_dict[f'{col}_delta_1h'] = df[col].diff(1)
        if delta_dict:
            feature_blocks.append(pd.DataFrame(delta_dict, index=df.index))

        # 6. 拼接并处理缺失值
        combined = pd.concat(feature_blocks, axis=1).ffill().fillna(0)
        
        # 🚀 【修改 6】关键同步：如果是动态推断模式，在首次 build 后锁定预期列名
        if not self._expected_columns:
            self._expected_columns = list(combined.columns)
            logger.info(f"🔒 [维度锁定] 动态推断完成，已锁定预期特征维度: {len(self._expected_columns)}")
            
        return combined

    def build_single_step(self, history_df: pd.DataFrame, current_row: dict, current_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        工业级单步推理接口 (彻底消灭单行数据导致的全零黑洞)
        """
        current_df = pd.DataFrame([current_row])
        full_df = pd.concat([history_df, current_df], ignore_index=True)
        
        # 使用批量构建逻辑
        built_df = self.build(full_df, is_inference=True, freq_hours=1.0)
        
        # 提取最后一行（即当前时刻的完整特征）
        single_step_features = built_df.iloc[[-1]].copy()
        
        # 强制注入准确的循环时间特征
        ts = current_time or pd.Timestamp.now()
        single_step_features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
        single_step_features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
        single_step_features['dow_sin']  = np.sin(2 * np.pi * ts.dayofweek / 7)
        single_step_features['dow_cos']  = np.cos(2 * np.pi * ts.dayofweek / 7)
        single_step_features['month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        single_step_features['month_cos'] = np.cos(2 * np.pi * ts.month / 12)

        return single_step_features

    def align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """强制对齐特征列顺序，确保推理时传入 LightGBM 的列顺序与训练时 100% 一致"""
        if not self._expected_columns:
            if df.empty or len(df.columns) == 0:
                raise RuntimeError("❌ 尚未锁定预期列名，且输入 DataFrame 没有可用列。请先调用 build() 训练数据或加载已保存的列名配置。")

            # 动态模式回退：从当前输入 DataFrame 自动锁定真实特征维度，避免训练/推理时触发空列名异常
            self._expected_columns = list(df.columns)
            logger.warning(
                f"⚠️ 预期列未锁定，已从输入 DataFrame 自动推断 {len(self._expected_columns)} 个特征列：{self._expected_columns}"
            )
            
        missing_cols = [c for c in self._expected_columns if c not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ 推理数据缺少 {len(missing_cols)} 个特征列，将使用 0 填充")
            
        return df.reindex(columns=self._expected_columns, fill_value=0.0)
        
    @property
    def expected_columns(self) -> List[str]:
        return self._expected_columns