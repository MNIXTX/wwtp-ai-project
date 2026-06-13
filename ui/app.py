import streamlit as st
import sys
import os
import time
import subprocess
from pathlib import Path

# --- 修复 Windows 终端 UTF-8 乱码 ---
if sys.platform == 'win32':
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, 'reconfigure'):
            try: s.reconfigure(encoding='utf-8')
            except Exception: pass

# ---- 路径引导 ----
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.i18n import t, lang_selector

# ---- 页面全局配置 (必须放在所有 st.* 调用之前) ----
st.set_page_config(
    page_title=t("page_title"),
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _kill_process(pid: int):
    """跨平台终止进程"""
    try:
        if sys.platform == "win32":
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                           capture_output=True, timeout=10)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


# ---- 侧边栏 ----
with st.sidebar:
    st.title(t("sidebar_title"))

    # ---- 安全退出（折叠，置顶）----
    with st.expander("🔴 " + t("sidebar_maintenance"), expanded=False):
        st.caption("⚠️ " + t("shutdown_warning"))
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(t("btn_stop_tasks"), type="secondary", use_container_width=True, key="btn_stop_tasks"):
                tasks = st.session_state.get("background_tasks", {})
                stopped = 0
                for task_id, info in list(tasks.items()):
                    if info.get("status") == "running":
                        pid = info.get("pid")
                        if pid:
                            _kill_process(pid)
                            info["status"] = "stopped"
                            stopped += 1
                if stopped:
                    st.success(t("tasks_stopped", stopped))
                else:
                    st.info(t("no_running_tasks"))
        with col_b:
            if st.button(t("btn_shutdown"), type="primary", use_container_width=True, key="btn_shutdown"):
                tasks = st.session_state.get("background_tasks", {})
                for info in tasks.values():
                    if info.get("status") == "running":
                        pid = info.get("pid")
                        if pid:
                            _kill_process(pid)
                for pf in (PROJECT_ROOT / "server.pid", PROJECT_ROOT / ".server.pid"):
                    try:
                        if pf.exists():
                            pf.unlink()
                    except Exception:
                        pass
                st.success(t("shutdown_done"))
                # Auto-close browser tab after shutdown
                st.markdown("""
                <script>
                setTimeout(function() {
                    window.open('', '_self', '');
                    window.close();
                }, 600);
                </script>
                """, unsafe_allow_html=True)
                time.sleep(1.5)  # Allow Streamlit to flush response to browser before exit
                os._exit(0)

    st.divider()
    st.caption(t("v1.3.0_label"))
    lang_selector()

# ---- 系统底层状态（主页顶部）----
try:
    from ui.core.service_adapter import SystemAdapter
    health = SystemAdapter.get_system_health()
    gateway_status = t("gateway_online") if health.get("gateway") == "online" else t("gateway_offline")
    pipeline_status = t("gateway_online") if health.get("pipeline") == "online" else t("gateway_offline")

    st.subheader(t("sidebar_system_status"))
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric(t("predictor_gateway"), gateway_status)
    with col_s2:
        st.metric(t("data_pipeline_label"), pipeline_status)
    with col_s3:
        tft_ok = health.get("tft_ok", False)
        lgbm_ok = health.get("lgbm_ok", False)
        model_status = "✅" if (tft_ok or lgbm_ok) else "⚠️"
        model_label = "TFT+LGBM" if (tft_ok and lgbm_ok) else ("TFT" if tft_ok else ("LGBM" if lgbm_ok else t("no_model")))
        st.metric(t("model_status_label"), model_status + " " + model_label)
    if health.get("gateway_error"):
        st.caption(t("gateway_error_prefix") + str(health['gateway_error']))
    st.divider()
except Exception:
    pass

# ---- 主页内容 ----
st.title(t("app_title"))
st.markdown(f"""
{t("app_intro")}

---

### {t("app_features_title")}

| {t("app_feature_page")} | {t("app_feature_func")} |
|------|------|
| **{t("label_live_board")}** | {t("app_feature_dashboard")} |
| **{t("nav_training")}** | {t("app_feature_training")} |
| **{t("nav_config")}** | {t("app_feature_config")} |
| **{t("nav_predict")}** | {t("app_feature_predict")} |

---

### {t("app_guide_title")}

1. {t("app_guide_first")}
2. {t("app_guide_daily")}
3. {t("app_guide_maintain")}
""")
