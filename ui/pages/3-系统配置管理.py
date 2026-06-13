# ui/pages/3-系统配置管理.py
import streamlit as st
import sys
import subprocess
import shutil
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.core.service_adapter import SystemAdapter

# [Fix] 移除 st.set_page_config — 页面文件中不应调用，仅在主 app.py 调用

st.header("系统配置管理 (热重载)")
st.caption("修改底层 config.yaml。保存后系统将执行原子级写入、清除缓存并触发热重载，无需重启服务。")

# ==========================================
# 1. 通过 Adapter 获取当前配置
# ==========================================
try:
    current_config = SystemAdapter.get_current_config()
    if not current_config:
        st.warning("配置文件为空或不存在，将使用默认值。")
        current_config = {}
except Exception as e:
    st.error(f"读取配置文件失败: {e}")
    st.stop()

# 提取真实配置
pipeline_cfg = current_config.get('pipeline', {})
model_cfg = current_config.get('model', {})
training_cfg = current_config.get('training', {})

# [Fix] 定义表单 widget 的 key 列表，用于强制刷新 UI
FORM_KEYS = [
    "cfg_lookback", "cfg_max_gap", "cfg_horizon", "cfg_test_ratio",
    "cfg_tft_hidden", "cfg_tft_epochs", "cfg_lgbm_trees", "cfg_div_thresh"
]

# ==========================================
# 2. 结构化配置修改表单
# ==========================================
with st.form("config_form"):
    st.subheader("数据管道与时序配置 (Pipeline)")
    col1, col2 = st.columns(2)
    with col1:
        lookback = st.number_input(
            "历史回溯窗口 (小时)",
            value=int(pipeline_cfg.get('lookback', 24)), step=1, key="cfg_lookback"
        )
        max_gap = st.number_input(
            "最大允许插值间隙 (小时)",
            value=int(pipeline_cfg.get('max_interp_gap', 6)), step=1, key="cfg_max_gap"
        )
    with col2:
        horizon = st.number_input(
            "预测未来窗口 (小时)",
            value=int(pipeline_cfg.get('horizon', 24)), step=1, key="cfg_horizon"
        )
        test_ratio = st.number_input(
            "时序测试集划分比例",
            value=float(pipeline_cfg.get('test_ratio', 0.2)),
            step=0.05, format="%.2f", key="cfg_test_ratio"
        )

    st.divider()
    st.subheader("核心模型与训练超参 (Model & Training)")
    col3, col4 = st.columns(2)
    with col3:
        tft_hidden = st.number_input(
            "TFT 隐层维度",
            value=int(model_cfg.get('tft_hidden_size', 64)), step=32, key="cfg_tft_hidden"
        )
        tft_epochs = st.number_input(
            "TFT 训练轮数 (Epochs)",
            value=int(training_cfg.get('tft_epochs', 50)), step=10, key="cfg_tft_epochs"
        )
    with col4:
        lgbm_trees = st.number_input(
            "LGBM 最大树数量",
            value=int(training_cfg.get('lgbm_n_estimators', 1000)), step=100, key="cfg_lgbm_trees"
        )
        div_thresh = st.number_input(
            "双模型分歧阈值 (mg/L)",
            value=float(training_cfg.get('divergence_threshold', 3.0)),
            step=0.5, format="%.1f", key="cfg_div_thresh"
        )

    st.divider()
    submitted = st.form_submit_button("保存配置并无损热重载", type="primary", use_container_width=True)

    if submitted:
        updates_pipeline = {
            "lookback": lookback, "horizon": horizon,
            "max_interp_gap": max_gap, "test_ratio": test_ratio
        }
        updates_model = {"tft_hidden_size": tft_hidden}
        updates_training = {
            "tft_epochs": tft_epochs,
            "lgbm_n_estimators": lgbm_trees,
            "divergence_threshold": div_thresh
        }

        try:
            SystemAdapter.update_and_reload_config("pipeline", updates_pipeline)
            SystemAdapter.update_and_reload_config("model", updates_model)
            SystemAdapter.update_and_reload_config("training", updates_training)
            st.success("配置已无损保存！所有工程注释已保留，底层实例已重置。")

            # 同步更新高级选项的文本框内容
            config_path = PROJECT_ROOT / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    st.session_state.raw_yaml_input = f.read()
        except Exception as e:
            st.error(f"保存失败: {e}")

# ==========================================
# 3. 高级开发者选项 (危险区域)
# ==========================================
st.divider()
with st.expander("高级开发者选项 (直接编辑 YAML)", expanded=False):
    st.warning("直接编辑原始 YAML 文本。语法错误将导致系统崩溃！修改后请务必点击下方的强制覆盖按钮。")

    if "raw_yaml_input" not in st.session_state:
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                st.session_state.raw_yaml_input = f.read()
        else:
            st.session_state.raw_yaml_input = ""

    st.text_area("原始 config.yaml", height=400, key="raw_yaml_input")

    if st.button("强制覆盖写入并重置系统", type="secondary"):
        raw_yaml_str = st.session_state.raw_yaml_input
        if not raw_yaml_str or not raw_yaml_str.strip():
            st.error("拒绝执行：YAML 内容为空！")
        else:
            try:
                SystemAdapter.force_overwrite_config(raw_yaml_str)
                for key in FORM_KEYS:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("强制覆盖成功！所有底层单例与缓存已重置。")
                st.rerun()
            except Exception as e:
                # [Fix] 捕获所有异常（ruamel.yaml.YAMLError 是 ImportError 的子类，包含在 Exception 中）
                st.error(f"写入失败: {e}")

# ==========================================
# 4. 环境维护与卸载工具
# ==========================================
st.divider()
st.header("环境维护与卸载工具")
st.caption("危险操作：以下功能将删除运行数据或卸载依赖，请谨慎使用。")

with st.expander("点击展开：系统清理与维护面板", expanded=False):
    tab_maintain, tab_uninstall = st.tabs(["运行数据清理", "依赖卸载与环境重置"])

    # ---- Tab 1: 运行数据清理 ----
    with tab_maintain:
        st.info("此操作仅删除运行时生成的临时文件，不会删除核心代码和配置文件。")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("清理所有日志 (logs/)", type="secondary", key="btn_clean_logs"):
                log_dir = PROJECT_ROOT / "logs"
                if log_dir.exists():
                    shutil.rmtree(log_dir)
                    st.success(f"已删除: logs/")
                else:
                    st.info("logs/ 目录不存在")

            if st.button("清理模型缓存 (models/)", type="secondary", key="btn_clean_models"):
                model_dir = PROJECT_ROOT / "models"
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    st.success(f"已删除: models/")
                    st.warning("模型已删除，需要重新训练：TFT → train_tft.py | LGBM → run_lgbm.py")
                else:
                    st.info("models/ 目录不存在")

        with col_m2:
            if st.button("清理输出结果 (outputs/)", type="secondary", key="btn_clean_outputs"):
                out_dir = PROJECT_ROOT / "outputs"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                    st.success(f"已删除: outputs/")
                else:
                    st.info("outputs/ 目录不存在")

            if st.button("清理构建产物 (build/dist/)", type="secondary", key="btn_clean_build"):
                for d in ["build", "dist"]:
                    dd = PROJECT_ROOT / d
                    if dd.exists():
                        shutil.rmtree(dd)
                        st.success(f"已删除: {d}/")
                spec_files = list(PROJECT_ROOT.glob("*.spec"))
                for sf in spec_files:
                    sf.unlink()
                    st.success(f"已删除: {sf.name}")
                if not any((PROJECT_ROOT / d).exists() for d in ["build", "dist"]) and not spec_files:
                    st.info("没有构建产物")

            if st.button("重置配置文件为默认", type="secondary", key="btn_reset_config"):
                # [Fix] 删除 config.yaml → 系统自动重新生成完整默认配置
                config_path = PROJECT_ROOT / "config.yaml"
                if config_path.exists():
                    backup_path = config_path.with_suffix(".yaml.bak")
                    shutil.copy2(config_path, backup_path)
                    config_path.unlink()
                    st.success(f"原配置已备份为 config.yaml.bak，默认配置已生成。")
                else:
                    st.info("配置文件不存在，无需重置。")

                # 清除所有缓存并触发热重载
                SystemAdapter.reset_all_instances()
                from config_manager import reload_config
                reload_config()
                st.rerun()

    # ---- Tab 2: 依赖卸载 ----
    with tab_uninstall:
        st.error("卸载操作不可逆！这将删除 Python 依赖包，系统将无法运行。")
        st.write("建议仅在需要彻底重装或迁移环境时使用。")

        # [Fix] 检查是否在虚拟环境中
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        if not in_venv:
            st.error(
                "**未检测到虚拟环境！**\n\n"
                "当前 Python 是系统级安装。执行卸载将删除全局 Python 包，"
                "可能影响本机其他项目。\n\n"
                "请先在终端运行:\n```bash\nvenv\\Scripts\\activate\n```\n"
                "然后在 venv 中重启 Streamlit。"
            )
        else:
            st.success(f"已检测到虚拟环境: {sys.prefix}")

        req_file = PROJECT_ROOT / "requirements.txt"
        uninstall_disabled = not in_venv

        if st.button(
            "卸载所有项目依赖包",
            type="primary",
            disabled=uninstall_disabled,
            key="btn_uninstall_deps"
        ):
            if not req_file.exists():
                st.error("未找到 requirements.txt，无法执行卸载。")
            else:
                with st.spinner("正在卸载依赖... (可能需要几分钟)"):
                    try:
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "uninstall", "-y", "-r", str(req_file)],
                            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120
                        )
                        if result.returncode == 0:
                            st.success("依赖卸载完成！请重启 Streamlit 服务。")
                            with st.expander("查看卸载日志"):
                                st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                        else:
                            st.error("卸载过程中出现错误:")
                            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("卸载命令超时（120秒），请手动在终端执行: pip uninstall -y -r requirements.txt")
                    except Exception as e:
                        st.error(f"执行卸载命令失败: {e}")

        st.divider()
        st.caption("如需删除整个项目文件夹，请在文件资源管理器中手动删除 `WWTP_AI_System` 文件夹。")
