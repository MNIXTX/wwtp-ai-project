# asm1_ppo_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Callable, List, Union
from dataclasses import dataclass, field
from loguru import logger

# ==========================================
# 1. 解耦核心：配置数据类与事件回调
# ==========================================
@dataclass
class RLConfig:
    """RL 环境配置"""
    # 物理边界
    volume: float = 5000.0
    cod_limit: float = 50.0
    nh3n_limit: float = 5.0
    
    # 动作映射参数 (工程单位：KLa 为 1/h，R 为无量纲比例)
    kla_bounds: List[float] = field(default_factory=lambda: [10.0, 200.0])
    r_bounds: List[float] = field(default_factory=lambda: [0.3, 1.5])
    kla_center: float = 80.0
    r_center: float = 0.8
    kla_scale: float = 60.0
    r_scale: float = 0.4
    
    # 奖励权重
    w_aeration: float = 2.0
    w_smooth: float = 1.0
    penalty_cod: float = 5.0
    penalty_nh3: float = 8.0
    reward_clip: float = 100.0
    
    # 工况泛化 (进水扰动)
    flow_range: List[float] = field(default_factory=lambda: [5000.0, 20000.0])
    cod_in_range: List[float] = field(default_factory=lambda: [150.0, 600.0])
    nh3_in_range: List[float] = field(default_factory=lambda: [15.0, 50.0])
    
    # 🚀 【手术 1】修复维度撕裂：将 List 改为 Dict，与 config.yaml 严格对齐，彻底消灭索引错位
    init_state_ranges: Dict[str, List[float]] = field(default_factory=lambda: {
        'S_S': [5.0, 30.0],      # 溶解性 COD (mg/L)
        'S_NH': [1.0, 10.0],     # 氨氮 (mg/L)
        'S_O': [1.5, 4.0],       # 溶解氧 (mg/L)
        'X_H': [1500.0, 3500.0], # 异养菌 (mg/L)
        'X_A': [50.0, 250.0]     # 自养菌 (mg/L)
    })
    
    # 安全阈值
    biomass_warn: float = 1000.0
    biomass_crash: float = 500.0
    
    # 扩充观测空间参考基准 (13维)
    obs_reference_scales: List[float] = field(default_factory=lambda: [
        100.0, 20.0, 4.0, 3000.0, 150.0,   # 内部状态 (5)
        15000.0, 600.0, 50.0,              # 当前进水 (3)
        15000.0, 600.0, 50.0,              # 未来 1 小时进水预测 (3)
        150.0, 1.5                         # 当前动作 (2)
    ])
    
    # 🚀 【手术 2】新增：SCADA 物理转换系数 (从 config.yaml 透传)
    S_S_fraction_of_eff_cod: float = 0.15

@dataclass
class EnvEvent:
    """环境事件载体"""
    event_type: str  
    severity: float
    message: str

EventCallback = Callable[[EnvEvent], None]

# ==========================================
# 2. 工业级 RL 环境引擎
# ==========================================
class WWTPControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: RLConfig,
        asm1_solver: any,
        event_cb: Optional[EventCallback] = None,
        max_steps: int = 168,
        training_mode: bool = True  # 🚀 [优化] 训练模式传 True，ODE 用宽松容差加速
    ):
        super().__init__()
        self.cfg = config
        self.solver = asm1_solver
        self.event_cb = event_cb or (lambda e: None)
        self.max_steps = max_steps
        self.training_mode = training_mode

        self.dt_hours = 1.0
        self.dt_days = self.dt_hours / 24.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.obs_dim = 13
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 🚀 [优化] 预分配观测数组，避免每步 np.concatenate 重新分配
        self.obs_ref_scales = np.array(self.cfg.obs_reference_scales, dtype=np.float32) + 1e-6
        self._obs_buf = np.empty(self.obs_dim, dtype=np.float32)  # 预分配

        self.state = np.zeros(5, dtype=np.float32)
        self.Q_in, self.S_S_in, self.S_NH_in = 0.0, 0.0, 0.0
        self.next_Q_in, self.next_S_S_in, self.next_S_NH_in = 0.0, 0.0, 0.0
        self.prev_action = (self.cfg.kla_center, self.cfg.r_center)

    def _emit_event(self, event_type: str, severity: float, msg: str):
        self.event_cb(EnvEvent(event_type, severity, msg))

    def _map_action(self, action: np.ndarray) -> Tuple[float, float]:
        a_kla, a_r = action[0], action[1]
        target_KLa = self.cfg.kla_center + a_kla * self.cfg.kla_scale
        target_R = self.cfg.r_center + a_r * self.cfg.r_scale
        
        target_KLa = float(np.clip(target_KLa, self.cfg.kla_bounds[0], self.cfg.kla_bounds[1]))
        target_R = float(np.clip(target_R, self.cfg.r_bounds[0], self.cfg.r_bounds[1]))
        return target_KLa, target_R

    def _get_obs(self, Kla: float, R: float) -> np.ndarray:
        # 🚀 [优化] 写入预分配缓冲区，避免每步 np.concatenate 分配新数组
        s = self.state
        buf = self._obs_buf
        buf[0], buf[1], buf[2], buf[3], buf[4] = s[0], s[1], s[2], s[3], s[4]
        buf[5], buf[6], buf[7] = self.Q_in, self.S_S_in, self.S_NH_in
        buf[8], buf[9], buf[10] = self.next_Q_in, self.next_S_S_in, self.next_S_NH_in
        buf[11], buf[12] = Kla, R

        # 原地归一化 (利用 np.tanh 的 out 参数减少一次分配)
        np.divide(buf, self.obs_ref_scales, out=buf)
        np.tanh(buf, out=buf)
        return buf

    def _compute_reward(self, S_S: float, S_NH: float, Kla: float, R: float, biomass: float) -> Tuple[float, Dict]:
        # 🚀 [优化] 局部变量绑定，减少属性查找开销
        cfg = self.cfg
        cod_limit = cfg.cod_limit
        nh3n_limit = cfg.nh3n_limit
        kla_bounds = cfg.kla_bounds
        kla_scale = cfg.kla_scale
        r_scale = cfg.r_scale

        reward = 1.0

        # 水质惩罚
        cod_over = S_S - cod_limit
        if cod_over < 0.0:
            cod_over = 0.0
        nh3n_over = S_NH - nh3n_limit
        if nh3n_over < 0.0:
            nh3n_over = 0.0
        r_quality = -cfg.penalty_cod * (cod_over * cod_over) - cfg.penalty_nh3 * (nh3n_over * nh3n_over)

        # 能耗惩罚
        kla_norm = (Kla - kla_bounds[0]) / (kla_bounds[1] - kla_bounds[0] + 1e-6)
        r_energy = -cfg.w_aeration * (kla_norm * kla_norm)

        # 动作平滑惩罚
        r_smooth = 0.0
        prev = self.prev_action
        if prev is not None:
            delta_Kla = (Kla - prev[0]) / (kla_scale + 1e-6)
            delta_R = (R - prev[1]) / (r_scale + 1e-6)
            r_smooth = -cfg.w_smooth * (delta_Kla * delta_Kla + delta_R * delta_R)

        # 生物量保护
        r_biomass = 0.0
        if biomass < cfg.biomass_warn:
            severity = (cfg.biomass_warn - biomass) / cfg.biomass_warn
            r_biomass = -20.0 * (severity * severity)
            self._emit_event("BIOMASS_WARN", severity, f"MLSS warning: {biomass:.0f}")

        reward += r_quality + r_energy + r_smooth + r_biomass
        if reward > cfg.reward_clip:
            reward = cfg.reward_clip
        elif reward < -cfg.reward_clip:
            reward = -cfg.reward_clip

        info = {
            "r_quality": r_quality, "r_energy": r_energy,
            "r_smooth": r_smooth, "r_biomass": r_biomass
        }
        return reward, info

    def _get_state_from_scada(self, scada_row: dict) -> np.ndarray:
        """🚀 【手术 2】补丁 2 落地：从 SCADA 字典安全构建 ODE 5维状态向量"""
        # 假设 scada_row 包含 config.yaml 中映射的键 (如 'eff_cod', 'NH3_reactor' 等)
        # 这里做兼容处理，如果键不存在则使用默认安全值
        s_s = scada_row.get('eff_cod', 30.0) * self.cfg.S_S_fraction_of_eff_cod
        s_nh = scada_row.get('NH3_reactor', 5.0)
        s_o = scada_row.get('DO_reactor', 2.0)
        x_h = scada_row.get('MLSS_reactor', 2500.0)
        x_a = scada_row.get('X_A_reactor', 150.0)
        
        return np.array([s_s, s_nh, s_o, x_h, x_a], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        self.Q_in = np.random.uniform(*self.cfg.flow_range)
        self.S_S_in = np.random.uniform(*self.cfg.cod_in_range)
        self.S_NH_in = np.random.uniform(*self.cfg.nh3_in_range)
        
        self.next_Q_in = self.Q_in + np.random.normal(0, 500)
        self.next_S_S_in = self.S_S_in + np.random.normal(0, 30)
        self.next_S_NH_in = self.S_NH_in + np.random.normal(0, 3)
        
        # 🚀 【手术 1】修复维度撕裂：支持从 options 注入真实 SCADA 状态 (用于推理/离线评估)
        if options and 'scada_row' in options:
            self.state = self._get_state_from_scada(options['scada_row'])
        else:
            # 训练模式：按字典键名安全随机初始化
            self.state = np.array([
                np.random.uniform(*self.cfg.init_state_ranges['S_S']),
                np.random.uniform(*self.cfg.init_state_ranges['S_NH']),
                np.random.uniform(*self.cfg.init_state_ranges['S_O']),
                np.random.uniform(*self.cfg.init_state_ranges['X_H']),
                np.random.uniform(*self.cfg.init_state_ranges['X_A'])
            ], dtype=np.float32)
        
        init_KLa, init_R = self.cfg.kla_center, self.cfg.r_center
        self.prev_action = (init_KLa, init_R)
        
        return self._get_obs(init_KLa, init_R), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        Kla_h, R = self._map_action(action) # Kla_h 单位是 1/h
        
        # 🚀 【手术 3】修复量纲黑洞：将 KLa 从 1/h 转换为 1/d，与 ODE 求解器的时间基准(天)严格统一！
        Kla_d = Kla_h * 24.0 
        
        try:
            result = self.solver.solve(
                t_span_days=(0, self.dt_days), y0=self.state,
                Q_in=self.Q_in,
                S_S_in=self.S_S_in, S_NH_in=self.S_NH_in, S_O_in=0.0,
                KLa=Kla_d, R=R, dt_hours=self.dt_hours,
                method='auto',  # 🚀 [优化] LSODA 优先，自适应刚/非刚切换
                silent=True,
                training_mode=self.training_mode  # 🚀 [优化] 训练时用宽松容差
            )
            if not result.get('success', False): 
                raise RuntimeError(result.get('message', 'ODE Failed'))
                
            # 兼容新版 Solver 返回的 'y' 矩阵 (N, 5)
            y_out = result.get('y')
            if y_out is None:
                raise KeyError("Solver result missing 'y' key.")
                
            self.state = y_out[-1] if y_out.ndim > 1 else y_out
            
        except Exception as e:
            self._emit_event("ODE_FAILED", 1.0, f"ODE 求解发散: {str(e)}")
            return self._get_obs(Kla_h, R), -self.cfg.reward_clip, True, False, {"error": "ODE_FAILED"}
            
        S_S_out, S_NH_out = self.state[0], self.state[1]
        current_biomass = self.state[3] + self.state[4]
        
        # 注意：奖励计算和观测空间依然使用工程单位 (Kla_h)，对 RL 智能体屏蔽底层量纲转换
        reward, reward_info = self._compute_reward(S_S_out, S_NH_out, Kla_h, R, current_biomass)
        
        self.Q_in = self.next_Q_in
        self.S_S_in = self.next_S_S_in
        self.S_NH_in = self.next_S_NH_in
        
        self.next_Q_in = np.clip(self.Q_in + np.random.normal(0, 500), *self.cfg.flow_range)
        self.next_S_S_in = np.clip(self.S_S_in + np.random.normal(0, 30), *self.cfg.cod_in_range)
        self.next_S_NH_in = np.clip(self.S_NH_in + np.random.normal(0, 3), *self.cfg.nh3_in_range)
        
        terminated = False
        truncated = self.current_step >= self.max_steps 
        
        if current_biomass < self.cfg.biomass_crash:
            reward -= 100.0
            terminated = True
            self._emit_event("BIOMASS_CRASH", 1.0, f"污泥彻底流失: MLSS={current_biomass:.0f}")
            
        self.prev_action = (Kla_h, R)
        
        info = {
            "step": self.current_step, "S_S": S_S_out, "S_NH": S_NH_out, 
            "MLSS": current_biomass, "KLa": Kla_h, "R": R, **reward_info
        }
        return self._get_obs(Kla_h, R), reward, terminated, truncated, info

# ==========================================
# 3. CLI 专属适配器 (Mock Solver)
# ==========================================
class MockASM1Solver:
    def solve(self, t_span_days, y0, Q_in, S_S_in, S_NH_in, S_O_in, KLa, R, dt_hours, method='RK45', silent=True, **kwargs):
        S_S, S_NH, S_O, X_H, X_A = y0
        # 模拟物理反应 (KLa 此时已经是 1/d，所以除以 24 换算回小时级别的影响)
        S_S = max(0, S_S + np.random.normal(0, 5) - (KLa/24) * 0.01)
        S_NH = max(0, S_NH + np.random.normal(0, 1) - (KLa/24) * 0.005)
        X_H = max(0, X_H - (1.0 - R) * 10) 
        
        return {
            "success": True, 
            "y": np.array([[S_S, S_NH, 2.0, X_H, X_A]]) # 对齐新版 Solver 返回 'y'
        }

if __name__ == "__main__":
    cli_config = RLConfig()
    
    def cli_event_handler(event: EnvEvent):
        logger.warning(f"🚨 环境事件 [{event.event_type}]: {event.message}")

    env = WWTPControlEnv(
        config=cli_config,
        asm1_solver=MockASM1Solver(),
        event_cb=cli_event_handler,
        max_steps=50
    )
    
    # 测试 1：纯随机初始化 (训练模式)
    obs, _ = env.reset()
    logger.info(f"✅ 训练模式初始化成功 | 观测维度: {obs.shape} (应为 13 维)")
    
    # 测试 2：注入 SCADA 真实数据 (推理/数字孪生模式)
    fake_scada = {'eff_cod': 40.0, 'NH3_reactor': 3.0, 'DO_reactor': 2.5, 'MLSS_reactor': 3000.0}
    obs, _ = env.reset(options={'scada_row': fake_scada})
    logger.info(f"✅ 推理模式初始化成功 | 注入 SCADA 后 S_S 应为 40*0.15=6.0: {env.state[0]:.2f}")
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        logger.info(f"Step {i+1} | Reward: {reward:.2f} | MLSS: {info['MLSS']:.0f} | KLa_h: {info['KLa']:.1f}")
        if term or trunc: break