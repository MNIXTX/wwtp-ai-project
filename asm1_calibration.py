# asm1_calibration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import os
import time
import threading # 🚀 【手术 2】引入线程锁，防止多进程/多线程下的状态污染
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

# 依赖注入：引入真实的物理基座
from asm1_ode_solver import ASM1Solver, ASM1Parameters, ReactorConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 解耦核心：配置数据类与回调接口
# ==========================================
@dataclass
class CalibrationConfig:
    """校准配置"""
    volume: float = 5000.0
    saturation_do: float = 9.0
    x_h_guess: float = 2500.0
    x_a_guess: float = 100.0
    
    # 🚀 【手术 1】新增：SCADA 物理转换系数 (与 RL 环境严格对齐)
    S_S_fraction: float = 0.15 
    
    # 待优化参数边界 [mu_H, K_S, K_OH, mu_A, K_NH, K_OA]
    bounds: List[List[float]] = field(default_factory=lambda: [
        [2.0, 8.0], [5.0, 60.0], [0.1, 1.0], [0.2, 1.5], [0.5, 5.0], [0.2, 1.5]
    ])
    
    de_popsize: int = 15
    de_maxiter: int = 50
    de_tol: float = 1e-4
    nh3_penalty_weight: float = 2.0
    
    col_map: Dict[str, str] = field(default_factory=lambda: {
        'flow': 'flow', 'inf_cod': 'inf_cod', 'inf_nh3': 'inf_nh3',
        'eff_cod': 'eff_cod', 'eff_nh3': 'eff_nh3', 'do_meas': 'do_meas', 'kla_meas': 'kla_meas'
    })

@dataclass
class OptimizationState:
    """优化状态载体"""
    iteration: int
    max_iter: int
    current_loss: float
    best_loss: float
    elapsed_time: float
    current_params: List[float]

ProgressCallback = Callable[[float, str], None]
StateCallback = Callable[[OptimizationState], None]
LogCallback = Callable[[str, str], None]

# ==========================================
# 2. ASM1 算法引擎 (纯计算，零 UI 依赖)
# ==========================================
class ASM1CalibrationEngine:
    """ASM1 校准引擎 (支持中断、细粒度状态抛出、向量化加速)"""
    
    def __init__(self, config: CalibrationConfig,
                 progress_cb: Optional[ProgressCallback] = None,
                 state_cb: Optional[StateCallback] = None,
                 log_cb: Optional[LogCallback] = None):
        self.cfg = config
        self.progress_cb = progress_cb or (lambda p, t: None)
        self.state_cb = state_cb or (lambda s: None)
        self.log_cb = log_cb or (lambda lvl, msg: logger.log(lvl, msg))
        
        self._stop_requested = False
        self.df = None
        
        self.reactor_cfg = ReactorConfig(volume=self.cfg.volume, S_O_sat=self.cfg.saturation_do)
        self.solver = ASM1Solver(params=ASM1Parameters(), reactor=self.reactor_cfg)
        
        # 🚀 【手术 2】引入线程锁，保护并发状态更新
        self._state_lock = threading.Lock()

    def _log(self, level: str, msg: str):
        self.log_cb(level, msg)

    def request_stop(self):
        self._stop_requested = True
        self._log("WARNING", "⚠️ 收到停止指令，将在当前迭代结束后安全终止...")

    def load_data(self, csv_path: str):
        self._log("INFO", f"正在加载真实数据: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到校准数据文件: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        initial_len = len(self.df)
        self.df = self.df.dropna()
        self._log("SUCCESS", f"加载完成，有效连续数据点: {len(self.df)} (丢弃 {initial_len - len(self.df)} 行)")

    def _ode_system_wrapper(self, t, y, u_interp):
        """桥接函数：将真实的 5 变量 ASM1Solver 适配给 scipy 的 solve_ivp"""
        Q, S_s_in, S_nh_in, Kla_h = u_interp(t)
        
        # 🚀 【手术 1】修复 KLa 量纲黑洞：将 SCADA 的 1/h 转换为 ODE 期望的 1/d
        Kla_d = Kla_h * 24.0 
        
        args = (Q, self.cfg.volume, S_s_in, S_nh_in, 0.0, Kla_d, 0.0)
        return self.solver._ode_system(t, y, args)

    def simulate(self, opt_params):
        """向量化一次性积分"""
        cm = self.cfg.col_map
        n_steps = len(self.df)
        
        dt_days = 1.0 / 24.0 
        t_span = (0.0, float(n_steps - 1) * dt_days)
        t_eval = np.arange(n_steps, dtype=float) * dt_days
        
        u_interp = interp1d(
            t_eval, 
            np.column_stack([
                self.df[cm['flow']].values, 
                self.df[cm['inf_cod']].values, 
                self.df[cm['inf_nh3']].values, 
                self.df[cm['kla_meas']].values # 保持原始 1/h 传入，在 wrapper 中统一转换
            ]),
            axis=0, kind='linear', bounds_error=False, fill_value="extrapolate"
        )
        
        self.solver.p.mu_H = opt_params[0]
        self.solver.p.K_S = opt_params[1]
        self.solver.p.K_OH = opt_params[2]
        self.solver.p.mu_A = opt_params[3]
        self.solver.p.K_NH = opt_params[4]
        self.solver.p.K_OA = opt_params[5]
        
        # 🚀 【补丁落地】修复 y0 物理失真：总 COD 转换为溶解性 COD
        y0 = [
            self.df[cm['eff_cod']].iloc[0] * self.cfg.S_S_fraction, 
            self.df[cm['eff_nh3']].iloc[0], 
            self.df[cm['do_meas']].iloc[0],
            self.cfg.x_h_guess,
            self.cfg.x_a_guess
        ]
        
        try:
            sol = solve_ivp(
                self._ode_system_wrapper, t_span, y0, args=(u_interp,), 
                t_eval=t_eval, method='RK45', rtol=1e-3, atol=1e-4
            )
            
            if not sol.success:
                return None, None, None
                
            sim_cod = np.maximum(sol.y[0], 0.0)
            sim_nh3 = np.maximum(sol.y[1], 0.0)
            sim_do = np.maximum(sol.y[2], 0.0)
            return sim_cod, sim_nh3, sim_do
            
        except Exception as e:
            self._log("DEBUG", f"ODE 求解异常: {str(e)}")
            return None, None, None

    def objective_function(self, opt_params):
        """使用 NRMSE 计算 Loss，并拦截更新全局最优 Loss"""
        if self._stop_requested:
            raise StopIteration("用户手动终止优化")

        sim_cod, sim_nh3, _ = self.simulate(opt_params)
        if sim_cod is None:
            return 1e6 
            
        cm = self.cfg.col_map
        # 🚀 【跨文件手术 1】修复物理失真：将 SCADA 总 COD 转换为溶解性 COD，与 ODE 输出对齐
        real_cod = self.df[cm['eff_cod']].values * self.cfg.S_S_fraction 
        real_nh3 = self.df[cm['eff_nh3']].values
        
        mean_cod = np.mean(real_cod) + 1e-4
        mean_nh3 = np.mean(real_nh3) + 1e-4
        
        nrmse_cod = np.sqrt(np.mean((real_cod - sim_cod)**2)) / mean_cod
        nrmse_nh3 = np.sqrt(np.mean((real_nh3 - sim_nh3)**2)) / mean_nh3
        
        current_loss = nrmse_cod + self.cfg.nh3_penalty_weight * nrmse_nh3
        
        with self._state_lock:
            if current_loss < self._run_best_loss:
                self._run_best_loss = current_loss
                self._best_params_so_far = opt_params.copy()
            
        return current_loss

    def _de_callback(self, xk, convergence):
        """差分进化的回调函数"""
        with self._state_lock:
            self._run_generation += 1
            
            state = OptimizationState(
                iteration=self._run_generation,
                max_iter=self.cfg.de_maxiter,
                current_loss=self._run_best_loss, 
                best_loss=self._run_best_loss,
                elapsed_time=time.time() - self._run_start_time,
                current_params=self._best_params_so_far.tolist()
            )
            
            progress = (self._run_generation / self.cfg.de_maxiter) * 100
            self.progress_cb(progress, f"Generation {self._run_generation}/{self.cfg.de_maxiter} (Pop: {self.cfg.de_popsize}) | Best Loss: {self._run_best_loss:.4f}")
        
        self.state_cb(state)
        return self._stop_requested

    def run_calibration(self) -> Dict[str, Any]:
        """执行差分进化全局优化"""
        self._stop_requested = False
        
        # 状态局部化与线程安全初始化
        with self._state_lock:
            self._run_generation = 0
            self._run_best_loss = 1e6
            self._best_params_so_far = np.zeros(len(self.cfg.bounds))
            self._run_start_time = time.time()
        
        self._log("INFO", "启动差分进化算法进行全局参数寻优...")
        self.progress_cb(0.0, "开始校准...")
        
        bounds = [tuple(b) for b in self.cfg.bounds]
        
        try:
            result = differential_evolution(
                self.objective_function, 
                bounds, 
                popsize=self.cfg.de_popsize, 
                maxiter=self.cfg.de_maxiter, 
                tol=self.cfg.de_tol,
                workers=1,       
                updating='deferred',
                callback=self._de_callback,  
                disp=False                   
            )
            
            self._log("SUCCESS", f"✅ 校准完成！最终加权 NRMSE Loss: {result.fun:.4f}")
            self.progress_cb(100.0, "校准完成！")
            return {"params": result.x, "loss": result.fun, "success": True}
            
        except StopIteration:
            self._log("WARNING", "🛑 校准已被用户手动终止。")
            with self._state_lock:
                return {"params": self._best_params_so_far, "loss": self._run_best_loss, "success": False}

# ==========================================
# 3. CLI 专属适配器
# ==========================================
def cli_state_handler(state: OptimizationState):
    if state.iteration % max(1, state.max_iter // 10) == 0 or state.iteration == 1:
        logger.info(f"Gen {state.iteration}/{state.max_iter} | Best Loss: {state.best_loss:.4f}")

if __name__ == "__main__":
    cli_config = CalibrationConfig(de_popsize=10, de_maxiter=20)
    
    engine = ASM1CalibrationEngine(
        config=cli_config,
        progress_cb=lambda p, t: logger.info(f"[{p:.1f}%] {t}"),
        state_cb=cli_state_handler,
        log_cb=lambda lvl, msg: logger.log(lvl, msg)
    )
    
    fake_csv = "fake_scada.csv"
    pd.DataFrame({
        'flow': np.random.uniform(8000, 12000, 100),
        'inf_cod': np.random.uniform(200, 400, 100),
        'inf_nh3': np.random.uniform(20, 40, 100),
        'eff_cod': np.random.uniform(30, 60, 100),
        'eff_nh3': np.random.uniform(2, 8, 100),
        'do_meas': np.random.uniform(1.5, 3.0, 100),
        'kla_meas': np.random.uniform(60, 100, 100)
    }).to_csv(fake_csv, index=False)
    
    engine.load_data(fake_csv)
    result = engine.run_calibration()
    
    if result['success'] or result['params'] is not None:
        param_names = ['mu_H', 'K_S', 'K_OH', 'mu_A', 'K_NH', 'K_OA']
        logger.success("\n--- 🎯 专属 ASM1 动力学参数 ---")
        for name, val in zip(param_names, result['params']):
            logger.success(f"  {name}: {val:.4f}")
            
    if os.path.exists(fake_csv):
        os.remove(fake_csv)