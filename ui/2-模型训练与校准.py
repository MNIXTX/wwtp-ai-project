# ui/pages/2-模型训练与校准.py
import streamlit as st
import sys, time, subprocess
from pathlib import Path
from collections import deque
from ui.init import setup_page, t, get_adapter, get_config
PROJECT_ROOT = setup_page()

cfg = get_config()

st.header(t("training_header"))
st.caption(t("training_caption"))


# ═══════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════
with st.expander(t("data_prep_expander"), expanded=False):
    st.markdown(t("data_prep_markdown"))

    try:
        from config.manager import CFG
        current_csv = str(getattr(CFG.paths, 'scada_data_csv', t("not_configured")))
    except Exception:
        current_csv = t("not_loaded")
    st.caption(t("current_source", current_csv))

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    csv_names = [f.name for f in csv_files] if csv_files else []

    if csv_names:
        st.subheader(t("select_file_header"))
        sel = st.selectbox(t("select_file_label"), csv_names, index=None, placeholder=t("select_placeholder"), key="sel_csv")
        if sel and st.button(t("btn_set_source", sel), key="btn_sel"):
            try:
                from config.manager import reload_config
                new_path = (data_dir / sel).as_posix()
                get_adapter().update_and_reload_config("paths", {"scada_data_csv": new_path})
                reload_config()
                st.success(t("success_source_updated", sel)); time.sleep(0.3); st.rerun()
            except Exception as e:
                st.error(t("err_update_failed", e))

    st.subheader(t("upload_header"))
    up = st.file_uploader(t("upload_label"), type=["csv"], key="up_csv")
    if up is not None:
        last = st.session_state.get("_up_done", "")
        if up.name != last:
            st.session_state._up_done = up.name
            dest = data_dir / up.name
            try:
                from config.manager import reload_config
                dest.write_bytes(up.getbuffer())
                st.success(t("success_file_saved", up.name))
                get_adapter().update_and_reload_config("paths", {"scada_data_csv": dest.as_posix()})
                reload_config(); time.sleep(0.3); st.rerun()
            except Exception as e:
                st.error(t("err_save_failed", e))

    st.subheader(t("custom_path_header"))
    cpath = st.text_input(t("custom_path_label"), placeholder=t("custom_path_placeholder"), key="txt_path")
    if cpath:
        p = Path(cpath.strip())
        if not p.is_absolute(): p = PROJECT_ROOT / p
        if p.exists():
            if st.button(t("btn_set_source", cpath.strip()), key="btn_cpath"):
                try:
                    from config.manager import reload_config
                    get_adapter().update_and_reload_config("paths", {"scada_data_csv": p.as_posix()})
                    reload_config(); st.success(t("success_updated")); time.sleep(0.3); st.rerun()
                except Exception as e:
                    st.error(t("err_update_failed", e))
        else:
            st.warning(t("warn_file_not_exist", str(p)))

# Pipeline button
col_p1, col_p2 = st.columns([1, 3])
with col_p1:
    if st.button(t("btn_run_pipeline"), type="secondary", help=t("btn_run_pipeline_help")):
        with st.spinner(t("spinner_pipeline")):
            try:
                from config.manager import CFG
                pipe_csv = str(CFG.paths.scada_data_csv)
            except Exception: pipe_csv = None
            cmd = [sys.executable, str(PROJECT_ROOT / "training" / "run_pipeline.py")]
            if pipe_csv: cmd.extend(["--csv", pipe_csv])
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300)
            if result.returncode == 0:
                st.success(t("success_pipeline_done"))
            else:
                st.error(t("err_pipeline_failed", result.stderr[-500:] if result.stderr else t("err_pipeline_no_output")))
with col_p2:
    st.caption(t("pipeline_hint"))

st.divider()

# ═══════════════════════════════════════════════════════════
# Auto-refresh toggle
# ═══════════════════════════════════════════════════════════
if "training_auto_refresh" not in st.session_state:
    st.session_state.training_auto_refresh = False

cbtn, ctog = st.columns([1, 3])
with cbtn:
    if st.button(t("btn_manual_refresh"), icon="🔄", use_container_width=True): st.rerun()
with ctog:
    st.session_state.training_auto_refresh = st.toggle(t("toggle_auto_poll"), value=st.session_state.training_auto_refresh)
st.divider()

# Auto-refresh
if st.session_state.training_auto_refresh:
    time.sleep(3); st.rerun()


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def read_log_tail(log_file: str, lines: int = 50) -> str:
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            return "".join(deque(f, maxlen=lines))
    except Exception: return t("not_loaded")


def render_training_tab(task_name_key: str, task_type: str, params_config: dict, warning_msg_key: str = None):
    task_display = t(task_name_key)
    if warning_msg_key: st.warning(t(warning_msg_key))

    with st.form(f"form_{task_type}"):
        st.caption(t("train_params_label"))
        cols = st.columns(len(params_config))
        params = {}
        for i, (key, cfg) in enumerate(params_config.items()):
            with cols[i]:
                if cfg['type'] == 'int':
                    params[key] = st.number_input(t(cfg['label_key']), value=cfg['default'], step=1, key=f"{task_type}_{key}")
                else:
                    params[key] = st.number_input(t(cfg['label_key']), value=cfg['default'], step=0.01, format="%.4f", key=f"{task_type}_{key}")

        if st.form_submit_button(t("btn_start_training", task_display), type="primary", use_container_width=True):
            with st.spinner(t("spinner_submitting")):
                try:
                    task_id = get_adapter().trigger_training_task(task_type, params)
                    st.success(t("success_task_submitted", task_id)); time.sleep(0.5); st.rerun()
                except Exception as e:
                    st.error(t("err_launch_failed", e))

    st.divider()
    st.caption(t("training_history"))

    all_tasks = st.session_state.get("background_tasks", {})
    type_tasks = {tid: info for tid, info in all_tasks.items() if info.get("task_type") == task_type}
    if not type_tasks:
        st.info(t("no_history")); return

    for task_id, info in sorted(type_tasks.items(), key=lambda x: x[1].get("start_time", 0), reverse=True):
        status_info = get_adapter().check_task_status(task_id)
        status = status_info.get("status", "unknown")
        log_file = info.get("log_file", "")
        icon = {"running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")
        label = t(f"status_{status}")

        with st.expander(f"{icon} {task_display} | `{task_id}` | {label}", expanded=(status == "running")):
            c1, c2 = st.columns([1, 3])
            with c1:
                pid_val = info.get('pid', 'N/A')
                st.caption(t("label_pid", pid_val))
                st.caption(t("label_task_type", info.get('task_type', 'N/A')))
            with c2:
                if log_file:
                    with st.popover(t("popover_view_log"), use_container_width=True):
                        st.code(read_log_tail(log_file), language="log", line_numbers=False)


# ═══════════════════════════════════════════════════════════
# Training Tabs
# ═══════════════════════════════════════════════════════════
tab_lgbm, tab_tft, tab_ppo = st.tabs([t("tab_lgbm"), t("tab_tft"), t("tab_ppo")])

with tab_lgbm:
    st.caption(t("lgbm_baseline"))
    render_training_tab("lgbm_baseline", "lgbm", {
        "n_estimators": {"type": "int", "default": 2000, "label_key": "param_n_estimators"},
        "learning_rate": {"type": "float", "default": 0.03, "label_key": "param_learning_rate"},
        "early_stop_rounds": {"type": "int", "default": 100, "label_key": "param_early_stop"},
    })

with tab_tft:
    st.caption(t("tft_temporal"))
    st.warning(t("warn_tft_slow"))
    render_training_tab("tft_temporal", "tft", {
        "epochs": {"type": "int", "default": 50, "label_key": "param_epochs"},
        "batch_size": {"type": "int", "default": 128, "label_key": "param_batch_size"},
        "hidden_size": {"type": "int", "default": 128, "label_key": "param_hidden_size"},
    })

with tab_ppo:
    st.caption(t("ppo_rl"))
    st.warning(t("warn_ppo_slow"))
    render_training_tab("ppo_rl", "ppo", {
        "total_timesteps": {"type": "int", "default": 500000, "label_key": "param_total_timesteps"},
        "n_envs": {"type": "int", "default": 4, "label_key": "param_n_envs"},
    })
