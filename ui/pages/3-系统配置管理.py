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
from ui.i18n import t

# [Fix] 移除 st.set_page_config — 页面文件中不应调用，仅在主 app.py 调用

st.header(t("config_header"))
st.caption(t("config_caption"))

# ==========================================
# 1. 通过 Adapter 获取当前配置
# ==========================================
try:
    current_config = SystemAdapter.get_current_config()
    if not current_config:
        st.warning(t("config_empty_warn"))
        current_config = {}
except Exception as e:
    st.error(t("config_read_err", e))
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
    st.subheader(t("config_pipeline_section"))
    col1, col2 = st.columns(2)
    with col1:
        lookback = st.number_input(
            t("config_lookback"),
            value=int(pipeline_cfg.get('lookback', 24)), step=1, key="cfg_lookback"
        )
        max_gap = st.number_input(
            t("config_max_interp"),
            value=int(pipeline_cfg.get('max_interp_gap', 6)), step=1, key="cfg_max_gap"
        )
    with col2:
        horizon = st.number_input(
            t("config_horizon"),
            value=int(pipeline_cfg.get('horizon', 24)), step=1, key="cfg_horizon"
        )
        test_ratio = st.number_input(
            t("config_test_ratio"),
            value=float(pipeline_cfg.get('test_ratio', 0.2)),
            step=0.05, format="%.2f", key="cfg_test_ratio"
        )

    st.divider()
    st.subheader(t("config_model_section"))
    col3, col4 = st.columns(2)
    with col3:
        tft_hidden = st.number_input(
            t("config_tft_hidden"),
            value=int(model_cfg.get('tft_hidden_size', 64)), step=32, key="cfg_tft_hidden"
        )
        tft_epochs = st.number_input(
            t("config_tft_epochs"),
            value=int(training_cfg.get('tft_epochs', 50)), step=10, key="cfg_tft_epochs"
        )
    with col4:
        lgbm_trees = st.number_input(
            t("config_lgbm_trees"),
            value=int(training_cfg.get('lgbm_n_estimators', 1000)), step=100, key="cfg_lgbm_trees"
        )
        div_thresh = st.number_input(
            t("config_divergence"),
            value=float(training_cfg.get('divergence_threshold', 3.0)),
            step=0.5, format="%.1f", key="cfg_div_thresh"
        )

    st.divider()
    submitted = st.form_submit_button(t("btn_save_config"), type="primary", use_container_width=True)

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
            st.success(t("success_config_saved"))

            # 同步更新高级选项的文本框内容
            config_path = PROJECT_ROOT / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    st.session_state.raw_yaml_input = f.read()
        except Exception as e:
            st.error(t("err_config_save", e))

# ==========================================
# 3. 高级开发者选项 (危险区域)
# ==========================================
st.divider()
with st.expander(t("advanced_expander"), expanded=False):
    st.warning(t("advanced_warn"))

    if "raw_yaml_input" not in st.session_state:
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                st.session_state.raw_yaml_input = f.read()
        else:
            st.session_state.raw_yaml_input = ""

    st.text_area(t("yaml_editor_label"), height=400, key="raw_yaml_input")

    if st.button(t("btn_force_overwrite"), type="secondary"):
        raw_yaml_str = st.session_state.raw_yaml_input
        if not raw_yaml_str or not raw_yaml_str.strip():
            st.error(t("err_yaml_empty"))
        else:
            try:
                SystemAdapter.force_overwrite_config(raw_yaml_str)
                for key in FORM_KEYS:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success(t("success_yaml_overwritten"))
                st.rerun()
            except Exception as e:
                # [Fix] 捕获所有异常（ruamel.yaml.YAMLError 是 ImportError 的子类，包含在 Exception 中）
                st.error(t("err_yaml_write", e))

# ==========================================
# 4. 环境维护与卸载工具
# ==========================================
st.divider()
st.header(t("maintenance_header"))
st.caption(t("maintenance_caption"))

with st.expander(t("cleanup_expander"), expanded=False):
    tab_maintain, tab_uninstall = st.tabs([t("tab_cleanup"), t("tab_uninstall")])

    # ---- Tab 1: 运行数据清理 ----
    with tab_maintain:
        st.info(t("cleanup_info"))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button(t("btn_clean_logs"), type="secondary", key="btn_clean_logs"):
                log_dir = PROJECT_ROOT / "logs"
                if log_dir.exists():
                    from config_manager import delete_log_files, reacquire_log_handler
                    deleted, locked = delete_log_files(log_dir)
                    if deleted:
                        st.success(t("success_logs_deleted") + (" " + t("locked_files_skipped", locked) if locked else ""))
                    elif locked:
                        st.warning(t("all_files_locked", locked))
                    reacquire_log_handler(log_dir)
                else:
                    st.info(t("info_logs_not_exist"))

            if st.button(t("btn_clean_models"), type="secondary", key="btn_clean_models"):
                model_dir = PROJECT_ROOT / "models"
                if model_dir.exists():
                    from config_manager import safe_rmtree
                    deleted, locked = safe_rmtree(model_dir)
                    if deleted:
                        st.success(t("success_models_deleted") + (" " + t("locked_files_skipped", locked) if locked else ""))
                        st.warning(t("warn_models_need_retrain"))
                    elif locked:
                        st.warning(t("all_files_locked", locked))
                else:
                    st.info(t("info_models_not_exist"))

        with col_m2:
            if st.button(t("btn_clean_outputs"), type="secondary", key="btn_clean_outputs"):
                out_dir = PROJECT_ROOT / "outputs"
                if out_dir.exists():
                    from config_manager import safe_rmtree
                    deleted, locked = safe_rmtree(out_dir)
                    if deleted:
                        st.success(t("success_outputs_deleted") + (" " + t("locked_files_skipped", locked) if locked else ""))
                    elif locked:
                        st.warning(t("all_files_locked", locked))
                else:
                    st.info(t("info_outputs_not_exist"))

            if st.button(t("btn_clean_build"), type="secondary", key="btn_clean_build"):
                from config_manager import safe_rmtree
                cleaned = 0
                for d in ["build", "dist"]:
                    dd = PROJECT_ROOT / d
                    if dd.exists():
                        safe_rmtree(dd)
                        cleaned += 1
                        st.success(t("success_build_deleted", d))
                spec_files = list(PROJECT_ROOT.glob("*.spec"))
                for sf in spec_files:
                    sf.unlink()
                    st.success(t("success_build_deleted", sf.name))
                    cleaned += 1
                if cleaned == 0:
                    st.info(t("info_no_build"))

            if st.button(t("btn_reset_config"), type="secondary", key="btn_reset_config"):
                # [Fix] 删除 config.yaml → 系统自动重新生成完整默认配置
                config_path = PROJECT_ROOT / "config.yaml"
                if config_path.exists():
                    backup_path = config_path.with_suffix(".yaml.bak")
                    shutil.copy2(config_path, backup_path)
                    config_path.unlink()
                    st.success(t("success_config_reset"))
                else:
                    st.info(t("info_config_not_exist"))

                # 清除所有缓存并触发热重载
                SystemAdapter.reset_all_instances()
                from config_manager import reload_config
                reload_config()
                st.rerun()

        # Soft reset section
        st.divider()
        st.caption(t("soft_reset_caption"))
        if st.button(t("btn_soft_reset"), type="secondary", key="btn_soft_reset"):
            from config_manager import safe_rmtree
            with st.spinner(t("spinner_cleaning")):
                deleted_total, locked_total = 0, 0
                # 仅清理临时/缓存目录，不碰 venv（运行中无法删除自身）
                for d in ["logs", "artifacts", "outputs", "__pycache__", "build", "dist"]:
                    path = PROJECT_ROOT / d
                    if path.exists():
                        try:
                            dl, lk = safe_rmtree(path)
                            deleted_total += dl
                            locked_total += lk
                        except Exception as e:
                            st.warning(t("delete_dir_failed", d, str(e)))
                # 清理 .spec 文件
                for sf in PROJECT_ROOT.glob("*.spec"):
                    try:
                        sf.unlink()
                        deleted_total += 1
                    except Exception:
                        pass
                if deleted_total:
                    msg = t("cleanup_done", deleted_total)
                    if locked_total:
                        msg += " " + t("locked_files_skipped", locked_total)
                    st.success(msg)
                elif locked_total:
                    st.warning(t("partial_cleanup_locked"))
                else:
                    st.info(t("no_files_to_clean"))

    # ---- Tab 2: 依赖卸载 ----
    with tab_uninstall:
        st.error(t("uninstall_warn"))
        st.write(t("uninstall_hint"))

        # 全面的 venv 健康检查
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        venv_cfg = Path(sys.prefix) / "pyvenv.cfg" if in_venv else None
        venv_healthy = in_venv and venv_cfg and venv_cfg.exists()

        if not in_venv:
            st.error(t("no_venv_error"))
        elif not venv_healthy:
            st.error(t("venv_broken_error"))
        else:
            st.success(t("venv_detected", sys.prefix))

        req_file = PROJECT_ROOT / "requirements.txt"
        uninstall_disabled = not venv_healthy

        if st.button(
            t("btn_uninstall_deps"),
            type="primary",
            disabled=uninstall_disabled,
            key="btn_uninstall_deps"
        ):
            if not req_file.exists():
                st.error(t("err_no_requirements"))
            else:
                with st.spinner(t("spinner_uninstalling")):
                    try:
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "uninstall", "-y", "-r", str(req_file)],
                            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120
                        )
                        if result.returncode == 0:
                            st.success(t("success_uninstall_done"))
                            with st.expander(t("uninstall_log_expander")):
                                st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                        else:
                            st.error(t("uninstall_err"))
                            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error(t("uninstall_timeout"))
                    except Exception as e:
                        st.error(t("uninstall_cmd_fail", e))

        st.divider()
        st.caption(t("manual_delete_hint"))
