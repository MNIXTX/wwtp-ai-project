"""
UI 公共初始化模块 — 消除样板 + 懒加载优化

Usage:
    from ui.init import setup_page, t, get_adapter, get_config
    PROJECT_ROOT = setup_page()
"""

import sys
from pathlib import Path
import streamlit as st


def setup_page():
    """Common UI initialization (no heavy imports)."""
    if sys.platform == 'win32':
        for s in (sys.stdout, sys.stderr):
            if hasattr(s, 'reconfigure'):
                try: s.reconfigure(encoding='utf-8')
                except Exception: pass

    if getattr(sys, 'frozen', False):
        PROJECT_ROOT = Path(sys.executable).parent.resolve()
    else:
        PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    return PROJECT_ROOT


# ── Lazy-load t() to avoid circular import at module level ──
def t(key: str, *args):
    from ui.i18n import t as _t
    return _t(key, *args)


# ── Cached singletons (lazy import inside) ──

@st.cache_resource(show_spinner=False)
def _load_adapter():
    from ui.core.service_adapter import SystemAdapter
    return SystemAdapter


def get_adapter():
    return _load_adapter()


@st.cache_data(ttl=30, show_spinner=False)
def get_system_health():
    try:
        return get_adapter().get_system_health()
    except Exception:
        return {"gateway": "unknown", "pipeline": "unknown", "tft_ok": False, "lgbm_ok": False}


@st.cache_data(ttl=10, show_spinner=False)
def get_config():
    from config.manager import CFG
    return {
        "cod_limit": getattr(getattr(CFG, 'asm1', None), 'default_cod_limit', 50.0),
        "nh3_limit": getattr(getattr(CFG, 'asm1', None), 'default_nh3_limit', 5.0),
        "tft_seq_len": getattr(getattr(CFG, 'model', None), 'tft_seq_len', 24),
        "tft_feature_names": list(getattr(getattr(CFG, 'model', None), 'tft_feature_names', [])),
    }
