"""
可复用 UI 组件库 (来自 awesome-streamlit 最佳实践, 领域无关)

Usage:
    from ui.components import metric_card, status_badge, section_header
"""

import streamlit as st
from typing import Optional


def status_badge(ok: bool, label_ok: str = "Online", label_fail: str = "Offline") -> str:
    """Return HTML status badge."""
    color = "#28a745" if ok else "#dc3545"
    label = label_ok if ok else label_fail
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{label}</span>'


def metric_card(
    label: str, value: str, delta: Optional[str] = None,
    delta_ok: bool = True, help_text: Optional[str] = None,
):
    """Styled metric card with optional delta indicator."""
    delta_color = "normal" if delta_ok else "inverse"
    st.metric(
        label=label, value=value, delta=delta,
        delta_color=delta_color, help=help_text,
    )


def section_header(title: str, description: str = "", divider: bool = True):
    """Consistent section header with optional description."""
    if divider:
        st.divider()
    st.subheader(title)
    if description:
        st.caption(description)


def compliance_indicator(
    predicted_value: float, limit: float, label: str = "COD",
    unit: str = "mg/L",
) -> None:
    """Show compliance status for a predicted effluent value."""
    from ui.i18n import t
    compliant = predicted_value <= limit
    if compliant:
        st.success(t("success_compliant", predicted_value, limit))
    else:
        st.error(t("error_exceeded", predicted_value, limit))


def model_status_panel(tft_ok: bool, lgbm_ok: bool, gateway_ok: bool):
    """Consistent model status display across pages."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("TFT", "✅ Loaded" if tft_ok else "❌ Not Found")
    with c2:
        st.metric("LGBM", "✅ Loaded" if lgbm_ok else "❌ Not Found")
    with c3:
        st.metric("Gateway", "✅ Online" if gateway_ok else "❌ Offline")


def data_summary_bar(file_name: str, rows: int, time_range: str, columns: int):
    """Four-column data summary bar."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", file_name)
    c2.metric("Rows", f"{rows:,}")
    c3.metric("Time Range", time_range)
    c4.metric("Columns", columns)
