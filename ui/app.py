import streamlit as st
import sys
import os
import time
import signal
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

from ui.i18n import t, get_lang, set_lang, lang_selector

# ---- 页面全局配置 (必须放在所有 st.* 调用之前) ----
st.set_page_config(
    page_title=t("page_title"),
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 侧边栏 ----
with st.sidebar:
    st.title(t("sidebar_title"))

    # 系统底层状态
    try:
        from ui.core.service_adapter import SystemAdapter
        health = SystemAdapter.get_system_health()
        tft_ok = health.get("tft_ok", False)
        lgbm_ok = health.get("lgbm_ok", False)
        gateway_status = t("gateway_online") if health.get("gateway") == "online" else t("gateway_offline")
        pipeline_status = t("gateway_online") if health.get("pipeline") == "online" else t("gateway_offline")

        st.subheader(t("sidebar_system_status"))
        st.metric(t("predictor_gateway"), gateway_status)
        st.metric(t("data_pipeline_label"), pipeline_status)
        if health.get("gateway_error"):
            st.caption(t("gateway_error_prefix") + str(health['gateway_error']))
    except Exception:
        st.metric(t("predictor_gateway"), t("gateway_unknown"))
        st.metric(t("data_pipeline_label"), t("gateway_unknown"))

    st.divider()
    st.subheader(t("sidebar_nav"))
    st.page_link("pages/1-数据浏览与看板.py", label=t("nav_dashboard"), icon="📊")
    st.page_link("pages/2-模型训练与校准.py", label=t("nav_training"), icon="🧠")
    st.page_link("pages/3-系统配置管理.py", label=t("nav_config"), icon="🛠️")
    st.page_link("pages/4-手动预测.py", label=t("nav_predict"), icon="🧪")

    st.divider()
    st.caption(t("v1.3.0_label"))

    # ---- 语言切换 ----
    lang_selector()

    # ---- 安全退出 ----
    st.divider()
    with st.expander(t("sidebar_maintenance"), expanded=False):
        st.warning(t("maintenance_warning"))

        # 停止后台训练任务
        if st.button(t("btn_stop_tasks"), type="secondary", use_container_width=True, key="btn_stop_tasks"):
            tasks = st.session_state.get("background_tasks", {})
            stopped = 0
            for task_id, info in list(tasks.items()):
                if info.get("status") == "running":
                    pid = info.get("pid")
                    if pid:
                        try:
                            if sys.platform == "win32":
                                subprocess.run(
                                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                                    capture_output=True, timeout=10
                                )
                            else:
                                os.kill(pid, signal.SIGTERM)
                            info["status"] = "stopped"
                            stopped += 1
                        except Exception:
                            pass
            if stopped:
                st.success(t("tasks_stopped", stopped))
            else:
                st.info(t("no_running_tasks"))

        # 停止 Streamlit 服务
        st.divider()
        st.error(t("shutdown_warning"))
        if st.button(t("btn_shutdown"), type="primary", use_container_width=True, key="btn_shutdown"):
            # 先清理后台任务
            tasks = st.session_state.get("background_tasks", {})
            for info in tasks.values():
                if info.get("status") == "running":
                    pid = info.get("pid")
                    if pid:
                        try:
                            if sys.platform == "win32":
                                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True, timeout=10)
                            else:
                                os.kill(pid, signal.SIGTERM)
                        except Exception:
                            pass
            st.success(t("shutdown_done"))
            time.sleep(1)
            os._exit(0)

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
