"""
Streamlit Session State 统一键名管理 (来自 awesome-streamlit 最佳实践)

将所有 st.session_state 键名集中定义，避免跨页面键名不一致导致的 bug。

Usage:
    from ui.session_state import KEYS, init_session_state
    init_session_state()
    st.session_state[KEYS.dashboard_file] = "my_data.csv"
"""

from dataclasses import dataclass
import streamlit as st


@dataclass
class SessionKeys:
    """所有 st.session_state 键名的唯一来源。"""
    # Dashboard
    dashboard_df: str = "dashboard_df"
    dashboard_file: str = "dashboard_file"

    # Training
    training_auto_refresh: str = "training_auto_refresh"
    background_tasks: str = "background_tasks"
    _up_done: str = "_up_done"

    # Config
    raw_yaml_input: str = "raw_yaml_input"

    # Shared
    lang: str = "lang"


KEYS = SessionKeys()


def init_session_state():
    """Initialize all session state defaults once per session."""
    defaults = {
        KEYS.dashboard_df: None,
        KEYS.dashboard_file: None,
        KEYS.training_auto_refresh: False,
        KEYS.background_tasks: {},
        KEYS.raw_yaml_input: "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
