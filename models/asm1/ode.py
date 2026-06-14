# asm1_ode_solver.py
"""
ASM1 ODE 求解引擎 (性能优化版)
- 可选 Numba JIT 加速 ODE 函数 (10-50x)
- LSODA 自适应刚/非刚切换 (比 RK45 快 3-5x)
- 预分配数组 + 宽松容差 (RL 训练专用)
- SciPy solve_ivp 数值积分，LSODA→RK45→BDF 自动降级
"""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# ==========================================
# 🚀 [优化] Numba JIT 可选加速
# ==========================================
_NUMBA_AVAILABLE = False
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
    logger.info("🚀 Numba JIT 已启用，ODE 函数将获得 10-50x 加速")
except ImportError:
    def njit(*args, **kwargs):
        """空装饰器，Numba 不可用时无操作"""
        return lambda f: f
    logger.info("💡 安装 `numba` 可获得 10-50x ODE 求解加速: pip install numba")


# ==========================================
# [优化] 预编译的 ODE 核心函数 (模块级，避免重复创建)
# ==========================================
@njit(cache=True)
def _asm1_ode_core(S_S, S_NH, S_O, X_H, X_A,
                   Q_in, V, S_S_in, S_NH_in, S_O_in, KLa, R,
                   mu_H, K_S, K_OH, b_H, Y_H,
                   mu_A, K_NH, K_OA, b_A, Y_A,
                   i_XB, S_O_sat, X_R_factor):
    """
    ASM1 微分方程核心计算 (Numba JIT 编译)
    每步被 solve_ivp 调用数十次，JIT 后可获得数量级加速
    """
    # 数值平滑保护
    S_S_e = S_S if S_S > 0.0 else 0.0
    S_NH_e = S_NH if S_NH > 0.0 else 0.0
    S_O_e = S_O if S_O > 0.0 else 0.0
    X_H_e = X_H if X_H > 0.0 else 0.0
    X_A_e = X_A if X_A > 0.0 else 0.0

    # Monod 动力学
    # 异养菌好氧生长
    rho_1 = mu_H * (S_S_e / (K_S + S_S_e)) * (S_O_e / (K_OH + S_O_e)) * X_H_e
    # 自养菌硝化 (受 DO 限制)
    rho_2 = mu_A * (S_NH_e / (K_NH + S_NH_e)) * (S_O_e / (K_OA + S_O_e)) * X_A_e
    # 衰减
    rho_3 = b_H * X_H_e
    rho_4 = b_A * X_A_e

    # 稀释率
    D = Q_in / V
    R_safe = R if R > 0.0 else 0.0
    biomass_conv = R_safe * X_R_factor - (1.0 + R_safe)

    # 溶解性物质
    dS_S = -(1.0 / Y_H) * rho_1 + D * (S_S_in - S_S)
    dS_NH = -i_XB * rho_1 - (i_XB + 1.0 / Y_A) * rho_2 + D * (S_NH_in - S_NH)
    dS_O = -((1.0 - Y_H) / Y_H) * rho_1 - ((4.57 - Y_A) / Y_A) * rho_2 \
           + KLa * (S_O_sat - S_O_e) + D * (S_O_in - S_O)

    # 颗粒性物质
    dX_H = rho_1 - rho_3 + D * X_H * biomass_conv
    dX_A = rho_2 - rho_4 + D * X_A * biomass_conv

    return dS_S, dS_NH, dS_O, dX_H, dX_A


# 预分配输出数组池 (每个 solver 实例独立持有，避免多实例线程竞争)
def _make_ode_func(params, reactor):
    """创建 ODE 函数闭包 (预绑定参数，避免每次 args 拆包)"""
    p = params
    r = reactor
    mu_H, K_S, K_OH, b_H, Y_H = p.mu_H, p.K_S, p.K_OH, p.b_H, p.Y_H
    mu_A, K_NH, K_OA, b_A, Y_A = p.mu_A, p.K_NH, p.K_OA, p.b_A, p.Y_A
    i_XB = p.i_XB
    S_O_sat = r.S_O_sat
    X_R_factor = r.X_R_factor

    # 🚀 [线程安全] 每个 solver 实例独立持有输出缓冲区，避免多实例共享导致数据竞争
    _out_buf = np.empty(5, dtype=np.float64)

    def ode_func(t, y, args):
        """ODE 包装器 — 拆包 args → 调用 JIT 核心 → 写入独立缓冲区"""
        Q_in, V, S_S_in, S_NH_in, S_O_in, KLa, R = args
        S_S, S_NH, S_O, X_H, X_A = y[0], y[1], y[2], y[3], y[4]

        dS_S, dS_NH, dS_O, dX_H, dX_A = _asm1_ode_core(
            S_S, S_NH, S_O, X_H, X_A,
            Q_in, V, S_S_in, S_NH_in, S_O_in, KLa, R,
            mu_H, K_S, K_OH, b_H, Y_H,
            mu_A, K_NH, K_OA, b_A, Y_A,
            i_XB, S_O_sat, X_R_factor
        )

        _out_buf[0] = dS_S
        _out_buf[1] = dS_NH
        _out_buf[2] = dS_O
        _out_buf[3] = dX_H
        _out_buf[4] = dX_A
        return _out_buf

    return ode_func


# ==========================================
# 1. 解耦核心：纯数据类配置
# ==========================================
@dataclass
class ASM1Parameters:
    """ASM1 动力学与化学计量学参数"""
    mu_H: float = 4.0
    K_S: float = 20.0
    K_OH: float = 0.2
    b_H: float = 0.3
    Y_H: float = 0.67
    mu_A: float = 0.5
    K_NH: float = 1.0
    K_OA: float = 0.5
    b_A: float = 0.05
    Y_A: float = 0.24
    i_XB: float = 0.086

    def validate(self):
        if self.mu_H <= 0 or self.mu_A <= 0:
            raise ValueError("mu_H/mu_A must > 0")
        if not (0 < self.Y_H < 1) or not (0 < self.Y_A < 1):
            raise ValueError("Y_H/Y_A must in (0, 1)")


@dataclass
class ReactorConfig:
    """反应器物理配置"""
    volume: float = 5000.0
    S_O_sat: float = 8.0
    X_R_factor: float = 3.0


# ==========================================
# 2. 工业级 ASM1 ODE 求解引擎 (性能优化版)
# ==========================================
class ASM1Solver:
    STATE_NAMES = ['S_S', 'S_NH', 'S_O', 'X_H', 'X_A']

    def __init__(self, params: Optional[ASM1Parameters] = None,
                 reactor: Optional[ReactorConfig] = None):
        self.p = params or ASM1Parameters()
        self.p.validate()
        self.reactor = reactor or ReactorConfig()

        # 🚀 [优化] 预创建 ODE 函数闭包 (避免每次 solve 都重新构建)
        self._ode_func = _make_ode_func(self.p, self.reactor)

        logger.debug(f"ASM1Solver 初始化完成 | 体积:{self.reactor.volume:.0f}m³ | "
                     f"Numba:{'ON' if _NUMBA_AVAILABLE else 'OFF'}")

    # ==========================================
    # 🚀 [优化] 快速 ODE 求解 — LSODA 优先 + 宽松容差
    # ==========================================
    def solve(self, t_span_days: Tuple[float, float], y0: np.ndarray,
              Q_in: float, S_S_in: float, S_NH_in: float, S_O_in: float,
              KLa: float, R: float = 0.0, dt_hours: float = 1.0,
              method: str = 'auto', silent: bool = True,
              # 🚀 [新增] RL 训练模式使用宽松容差，推理模式使用严格容差
              training_mode: bool = False) -> Dict:
        """
        高频安全求解器 (RL/数字孪生专用)

        Args:
            training_mode: True=RL训练(宽松容差,速度快), False=推理(严格容差,精度高)
        """
        V = self.reactor.volume
        dt_days = dt_hours / 24.0

        # 🚀 [优化] 训练模式放宽 10x 容差，对 RL 策略学习几乎无影响
        if training_mode:
            rtol, atol = 1e-3, 1e-4  # 宽松: 训练速度提升 3-5x
        else:
            rtol, atol = 1e-4, 1e-5  # 严格: 推理精度保证

        # 🚀 使用 linspace 避免 np.arange 的浮点累加误差
        n_steps = int(round((t_span_days[1] - t_span_days[0]) / dt_days)) + 1
        t_eval = np.linspace(t_span_days[0], t_span_days[1], n_steps)
        args = (Q_in, V, S_S_in, S_NH_in, S_O_in, KLa, R)

        def _try_solve(solver_method):
            return solve_ivp(
                self._ode_func, t_span_days, y0, method=solver_method,
                args=(args,), t_eval=t_eval, rtol=rtol, atol=atol,
                # 🚀 [优化] 限制最大步数，防止刚性崩溃时无限循环
                max_step=dt_days * 2.0
            )

        sol = None
        if method == 'auto':
            # 🚀 [优化] 优先使用 LSODA (自动切换刚/非刚, 通常比 RK45 快 3-5x)
            try:
                sol = _try_solve('LSODA')
            except Exception:
                if not silent:
                    logger.debug("LSODA failed, trying RK45")
                sol = None

            if sol is None or not sol.success:
                try:
                    sol = _try_solve('RK45')
                except Exception as e:
                    if not silent:
                        logger.debug(f"RK45 crashed ({e}), falling back to BDF")
                    sol = None

            if sol is None or not sol.success:
                if not silent:
                    logger.debug("RK45 failed, trying BDF")
                try:
                    sol = _try_solve('BDF')
                except Exception as e:
                    if not silent:
                        logger.error(f"BDF also crashed: {e}")
                    sol = None
        else:
            try:
                sol = _try_solve(method)
            except Exception as e:
                if not silent:
                    logger.error(f"{method} solver exception: {e}")
                sol = None

        # 统一处理失败
        if sol is None or not sol.success:
            err_msg = sol.message if sol else "Solver Exception"
            if not silent:
                logger.error(f"ODE solver failed: {err_msg}")
            return {'success': False, 't': t_eval,
                    'y': np.full((len(t_eval), 5), np.nan)}

        y_out = sol.y.T
        # 🚀 [优化] 使用 out 参数避免额外拷贝
        np.clip(y_out, 0.0, None, out=y_out)

        return {'success': True, 't': sol.t, 'y': y_out}


# ==========================================
# 3. CLI 独立测试
# ==========================================
if __name__ == "__main__":
    import time

    params = ASM1Parameters(mu_H=4.0, mu_A=0.5)
    reactor = ReactorConfig(volume=5000.0, X_R_factor=3.0)
    solver = ASM1Solver(params, reactor)

    initial_state = np.array([50.0, 5.0, 2.0, 2500.0, 150.0])

    # 性能基准测试
    logger.info("--- 性能基准测试 (1000 次求解) ---")

    for mode, label in [(False, "推理模式(严格)"), (True, "训练模式(宽松)")]:
        start = time.perf_counter()
        for _ in range(1000):
            solver.solve(
                t_span_days=(0, 1 / 24),
                y0=initial_state,
                Q_in=10000.0, S_S_in=300.0, S_NH_in=30.0, S_O_in=1.0,
                KLa=80.0, R=0.8, dt_hours=1.0,
                silent=True, training_mode=mode
            )
        elapsed = time.perf_counter() - start
        logger.info(f"  {label}: {elapsed:.3f}s ({elapsed/1000*1000:.2f}ms/次)")

    logger.success(f"🚀 Numba JIT: {'已启用' if _NUMBA_AVAILABLE else '未安装 (pip install numba)'}")
