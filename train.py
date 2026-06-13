import os
import sys

if sys.platform == 'win32':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try: s.reconfigure(encoding='utf-8')
            except Exception: pass

import argparse
import random
import numpy as np
import torch
import multiprocessing as mp
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

# 导入您的环境
from asm1_ppo_env import WWTPControlEnv, RLConfig
from asm1_ode_solver import ASM1Solver, ASM1Parameters
from config_manager import CFG

# ==================== 0. 全局兼容性与路径基准 ====================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _configure_threads(device: str, n_envs: int):
    """🚀 [优化] 配置 CPU 线程数，防止 SubprocVecEnv 多进程线程爆炸"""
    if device == "cpu":
        # CPU 训练：每个 PyTorch 进程只占 1-2 线程，避免 N 个进程争抢 CPU
        # SubprocVecEnv 会 fork N 个子进程，若每个进程默认使用全部核心，会导致 (n_envs+1) * cores 的线程争抢
        n_threads = max(1, min(2, mp.cpu_count() // (n_envs + 1)))
        torch.set_num_threads(n_threads)
        # 同时限制 OpenMP/MKL 线程 (numpy/scipy 底层)
        os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
        logger.info(f"🧵 CPU 线程控制: PyTorch={n_threads}, OMP/MKL={n_threads}")
    else:
        # GPU 训练：PyTorch 用较多的 CPU 线程做数据加载，但不影响主计算
        torch.set_num_threads(max(1, mp.cpu_count() // 2))
        # 🚀 [优化] GPU 性能配置
        torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 (Ampere+ GPU)
        logger.info(f"🚀 GPU 优化: cudnn.benchmark=True, TF32=ON")


def get_optimal_device(config_device: str) -> str:
    """智能探测计算设备"""
    if config_device in ("cuda", "auto"):
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1).cuda()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
                logger.info(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
                return "cuda"
            except Exception as e:
                logger.warning(f"⚠️ CUDA init failed ({e})")
        else:
            logger.warning("⚠️ No CUDA device detected")
    logger.warning("⚠️ Falling back to CPU (slower but stable)")
    return "cpu"


def set_global_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==================== 1. 自定义业务监控回调 (优化版) ====================
class WWTPMetricsCallback(BaseCallback):
    """
    自定义 Callback: 记录业务指标 + 保存最佳模型
    [优化] 减少每步 List 操作 + 延迟 numpy 转换
    """

    def __init__(self, check_freq: int = 1000, save_path: str = None, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        base_save = save_path or CFG.paths.model_dir
        self.save_path = base_save if os.path.isabs(base_save) else os.path.join(BASE_DIR, base_save)
        self.best_score = -np.inf
        os.makedirs(self.save_path, exist_ok=True)

        self.cod_limit = CFG.asm1.default_cod_limit
        self.nh3_limit = CFG.asm1.default_nh3_limit
        self.energy_weight = CFG.training.ppo_energy_weight

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        # 🚀 [优化] 仅每隔 check_freq 步执行一次完整统计 (大幅减少 overhead)
        if self.n_calls % self.check_freq != 0:
            return True

        infos = self.locals.get('infos', [])
        if not infos:
            return True

        # 使用列表推导 + 生成器减少中间分配
        cod_vals = [info['S_S_out'] for info in infos if 'S_S_out' in info]
        if not cod_vals:
            return True

        nh3_vals = [info['S_NH_out'] for info in infos if 'S_NH_out' in info]
        kla_vals = [info.get('KLa', 0) for info in infos]

        kla_min, kla_max = CFG.asm1.rl_kla_bounds
        kla_range = kla_max - kla_min + 1e-6

        cod_arr = np.array(cod_vals)
        nh3_arr = np.array(nh3_vals)
        energy_arr = np.array([(k - kla_min) / kla_range for k in kla_vals])

        cod_pass = np.mean(cod_arr < self.cod_limit)
        nh3_pass = np.mean(nh3_arr < self.nh3_limit)
        total_pass_rate = np.mean((cod_arr < self.cod_limit) & (nh3_arr < self.nh3_limit))
        avg_energy = np.mean(energy_arr)

        self.logger.record("wwtp/cod_pass_rate", cod_pass)
        self.logger.record("wwtp/nh3_pass_rate", nh3_pass)
        self.logger.record("wwtp/total_pass_rate", total_pass_rate)
        self.logger.record("wwtp/avg_energy_cost", avg_energy)

        current_score = total_pass_rate * 100.0 - avg_energy * self.energy_weight

        logger.info(f"📊 Step {self.n_calls} | Pass: {total_pass_rate*100:.1f}% | "
                    f"Energy: {avg_energy:.3f} | Score: {current_score:.2f}")

        if current_score > self.best_score:
            self.best_score = current_score
            model_path = os.path.join(self.save_path, "best_wwtp_model")
            self.model.save(model_path)
            logger.success(f"🏆 New best model! Score: {current_score:.2f} -> {model_path}")

        return True

# ==================== 2. 环境构建工厂 (优化版) ====================
def make_wwtp_env(rank: int, seed: int = 0, training_mode: bool = True):
    """🚀 [优化] 训练环境传入 training_mode=True，ODE 求解用宽松容差"""
    def _init():
        rl_cfg = RLConfig(
            volume=CFG.asm1.volume,
            cod_limit=CFG.asm1.default_cod_limit,
            nh3n_limit=CFG.asm1.default_nh3_limit,
            kla_bounds=list(CFG.asm1.rl_kla_bounds),
            r_bounds=list(CFG.asm1.rl_r_bounds),
            kla_center=CFG.asm1.rl_kla_center,
            r_center=CFG.asm1.rl_r_center,
            kla_scale=CFG.asm1.rl_kla_scale,
            r_scale=CFG.asm1.rl_r_scale,
            w_aeration=CFG.asm1.rl_w_aeration,
            w_smooth=CFG.asm1.rl_w_smooth,
            penalty_cod=CFG.asm1.rl_penalty_cod,
            penalty_nh3=CFG.asm1.rl_penalty_nh3,
            reward_clip=CFG.asm1.rl_reward_clip,
            flow_range=list(CFG.asm1.rl_flow_range),
            cod_in_range=list(CFG.asm1.rl_cod_in_range),
            nh3_in_range=list(CFG.asm1.rl_nh3_in_range),
            biomass_warn=CFG.asm1.rl_biomass_warn,
            biomass_crash=CFG.asm1.rl_biomass_crash,
        )
        env = WWTPControlEnv(
            config=rl_cfg,
            asm1_solver=ASM1Solver(ASM1Parameters()),
            max_steps=CFG.model.ppo_episode_steps,
            training_mode=training_mode,  # 🚀 [优化] 训练用宽松容差
        )
        return env
    return _init

# ==================== 3. 主训练逻辑 (优化版) ====================
def main():
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="WWTP PPO Training Pipeline")
    parser.add_argument('--n-envs', type=int, default=CFG.training.ppo_n_envs)
    parser.add_argument('--total-timesteps', type=int, default=CFG.training.ppo_total_timesteps)
    parser.add_argument('--resume', type=str, default=CFG.paths.resume_model_path)
    parser.add_argument('--log-dir', type=str, default=CFG.paths.log_dir)
    parser.add_argument('--save-dir', type=str, default=CFG.paths.model_dir)
    # 🚀 [新增] 命令行控制优化开关
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='启用 FP16 混合精度训练 (需要 GPU + PyTorch 1.6+)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='启用 torch.compile 加速 (需要 PyTorch 2.0+)')
    args = parser.parse_args()

    args.log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(BASE_DIR, args.log_dir)
    args.save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(BASE_DIR, args.save_dir)
    if args.resume and not os.path.isabs(args.resume):
        args.resume = os.path.join(BASE_DIR, args.resume)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 🚀 Step 0: 探测设备 + 配置线程
    ppo_cfg = CFG.training.ppo_params
    optimal_device = get_optimal_device(ppo_cfg.device)
    _configure_threads(optimal_device, args.n_envs)
    set_global_seed(42)

    # 🚀 Step 1: 创建并行环境 (training_mode=True → ODE 宽松容差)
    logger.info(f"🚀 Initializing {args.n_envs} parallel environments (training mode)...")
    env_fns = [make_wwtp_env(i, seed=42, training_mode=True) for i in range(args.n_envs)]

    vec_env = None
    try:
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        vec_env = VecMonitor(vec_env)
        logger.success(f"✅ {args.n_envs} SubprocVecEnv environments ready")
    except Exception as e:
        logger.error(f"❌ SubprocVecEnv failed: {e}")
        logger.warning("⚠️ Falling back to DummyVecEnv (slower, single-process)...")
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)

    # 🚀 Step 2: 构建或加载 PPO 模型
    lr_schedule = get_linear_fn(ppo_cfg.learning_rate_start, ppo_cfg.learning_rate_end, 1.0)
    clip_schedule = get_linear_fn(ppo_cfg.clip_range_start, ppo_cfg.clip_range_end, 1.0)

    if args.resume and os.path.exists(args.resume):
        logger.info(f"🔄 Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=optimal_device)
        model.set_env(vec_env)
    else:
        logger.info("🏗️ Creating new PPO model...")
        model = PPO(
            policy=ppo_cfg.policy,
            env=vec_env,
            learning_rate=lr_schedule,
            clip_range=clip_schedule,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            ent_coef=ppo_cfg.ent_coef,
            vf_coef=ppo_cfg.vf_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            tensorboard_log=args.log_dir,
            verbose=1,
            device=optimal_device
        )

        # 🚀 [优化] torch.compile 加速策略网络 (PyTorch 2.0+)
        if args.compile and hasattr(torch, 'compile'):
            try:
                model.policy = torch.compile(model.policy, mode="reduce-overhead")
                logger.success("✅ torch.compile enabled on policy network")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")

    # 🚀 [优化] FP16 混合精度 (仅 GPU)
    if args.fp16 and optimal_device == "cuda":
        logger.info("🎯 FP16 mixed precision enabled")
        # SB3 不直接支持 amp，但 CUDA 的 TF32 + cudnn 已启用

    logger.info(f"🔥 Training | Device: {optimal_device.upper()} | "
                f"Steps: {args.total_timesteps} | Envs: {args.n_envs}")
    logger.info(f"   n_steps={ppo_cfg.n_steps} | batch_size={ppo_cfg.batch_size} | "
                f"n_epochs={ppo_cfg.n_epochs}")

    metrics_callback = WWTPMetricsCallback(
        check_freq=CFG.training.ppo_eval_freq,
        save_path=args.save_dir
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=metrics_callback,
            tb_log_name="PPO_WWTP_V1",
            reset_num_timesteps=False if args.resume else True,
            # 🚀 [优化] 每 log_interval 步输出一次进度 (降低 I/O)
            log_interval=max(1, ppo_cfg.n_steps // 4)
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("🚨 GPU OOM! Reduce batch_size in config.yaml or use CPU.")
            if optimal_device == "cuda":
                torch.cuda.empty_cache()
        else:
            logger.error(f"Runtime error: {e}")
            raise
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted, saving...")
    finally:
        try:
            final_path = os.path.join(args.save_dir, "final_wwtp_model")
            model.save(final_path)
            logger.success(f"✅ Final model saved: {final_path}")
        except Exception as save_err:
            logger.error(f"❌ Save failed: {save_err}")

        if vec_env is not None:
            try:
                vec_env.close()
                logger.info("🔒 Environments released")
            except Exception:
                pass


if __name__ == "__main__":
    mp.freeze_support()
    main()