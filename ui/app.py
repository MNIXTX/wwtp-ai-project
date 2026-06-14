import streamlit as st
import sys
import os
import time
import subprocess
from pathlib import Path
from ui.init import setup_page, get_system_health
from ui.i18n import t, lang_selector

PROJECT_ROOT = setup_page()

# --- Page config (must be first st.* call) ---
st.set_page_config(
    page_title=t("page_title"),
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _kill_process(pid: int):
    try:
        if sys.platform == "win32":
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                           capture_output=True, timeout=10)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.title(t("sidebar_title"))

    # --- Maintenance ---
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
                        if pid: _kill_process(pid)
                for pf in (PROJECT_ROOT / "server.pid", PROJECT_ROOT / ".server.pid"):
                    try:
                        if pf.exists(): pf.unlink()
                    except Exception: pass
                st.success(t("shutdown_done"))
                # Auto-close: components.html guarantees JS execution before server dies
                st.components.v1.html("""
                <html><body><script>
                (function() {
                    try { window.top.close(); } catch(e) {}
                    setTimeout(function() {
                        try { window.open('', '_self', ''); window.top.close(); } catch(e) {}
                    }, 400);
                    setTimeout(function() {
                        document.body.innerHTML = '<h3 style=\"text-align:center;margin-top:40px;font-family:sans-serif\">Server stopped.<br>You may close this tab.</h3>';
                    }, 2000);
                })();
                </script></body></html>
                """, height=80)
                import threading
                def _delayed_kill():
                    time.sleep(2.5)
                    os._exit(0)
                threading.Thread(target=_delayed_kill, daemon=True).start()
                st.stop()

    st.divider()
    st.caption(t("v1.3.0_label"))
    lang_selector()

# ═══════════════════════════════════════════════════════════
# System Status (cached, 30s TTL)
# ═══════════════════════════════════════════════════════════
with st.spinner(t("sidebar_system_status") + "..."):
    health = get_system_health()

st.subheader(t("sidebar_system_status"))
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric(t("predictor_gateway"), t("gateway_online") if health.get("gateway") == "online" else t("gateway_offline"))
with col_s2:
    st.metric(t("data_pipeline_label"), t("gateway_online") if health.get("pipeline") == "online" else t("gateway_offline"))
with col_s3:
    tft_ok = health.get("tft_ok", False)
    lgbm_ok = health.get("lgbm_ok", False)
    model_icon = "✅" if (tft_ok or lgbm_ok) else "⚠️"
    model_label = "TFT+LGBM" if (tft_ok and lgbm_ok) else ("TFT" if tft_ok else ("LGBM" if lgbm_ok else t("no_model")))
    st.metric(t("model_status_label"), f"{model_icon} {model_label}")

if health.get("gateway_error"):
    st.caption(t("gateway_error_prefix") + str(health['gateway_error']))
st.divider()

# ═══════════════════════════════════════════════════════════
# Main Page
# ═══════════════════════════════════════════════════════════
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
