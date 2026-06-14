# ui/pages/1-数据浏览与看板.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- Init ---
from ui.init import setup_page, t, get_adapter, get_config
PROJECT_ROOT = setup_page()

cfg = get_config()
COD_LIMIT = cfg["cod_limit"]
NH3_LIMIT = cfg["nh3_limit"]

st.header(t("dashboard_header"))
st.caption(t("dashboard_caption"))


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Loading CSV...")
def _load_csv(filepath: str) -> pd.DataFrame:
    """Cached CSV loader (5min TTL)."""
    for enc in ['utf-8-sig', 'utf-8', 'gbk']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode: {filepath}")


def _find_time_col(df: pd.DataFrame):
    for c in ['timestamp', 'time', 'datetime', 'date']:
        matches = [x for x in df.columns if c in str(x).lower()]
        if matches: return matches[0]
    return None


def _smart_resample(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Auto-resample to readable granularity based on time span."""
    df = df.set_index(time_col).sort_index()
    span_h = (df.index.max() - df.index.min()).total_seconds() / 3600
    if span_h > 24 * 60:        rule, fmt = "1D", "%m-%d"
    elif span_h > 24 * 7:       rule, fmt = "6h", "%m-%d %H:00"
    elif span_h > 24 * 2:       rule, fmt = "2h", "%m-%d %H:00"
    else:                        rule, fmt = None, "%m-%d %H:%M"
    if rule and len(df) > 2:
        df = df.resample(rule).mean()
    df.index = df.index.strftime(fmt)
    return df


# ═══════════════════════════════════════════════════════════
# Data Source
# ═══════════════════════════════════════════════════════════
st.divider()
data_dir = PROJECT_ROOT / "data"
csv_files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
csv_names = [f.name for f in csv_files] if csv_files else []

if not csv_names:
    st.warning(t("dashboard_no_csv"))
    st.stop()

col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    selected_file = st.selectbox(t("dashboard_source"), csv_names, index=0, help=t("dashboard_source_help"))
with col_f2:
    st.write(""); st.write("")
    load_btn = st.button(t("btn_load_data"), use_container_width=True, type="primary")

# Initialize state
if "dashboard_file" not in st.session_state:
    st.session_state.dashboard_file = None

filepath = str(data_dir / selected_file)

if load_btn or st.session_state.dashboard_file != selected_file:
    try:
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        if Path(filepath).stat().st_size == 0:
            st.error(t("err_file_empty")); st.stop()
        if size_mb > 500:
            st.warning(t("warn_file_large", f"{size_mb:.0f}"))
        scada_df = _load_csv(filepath)
        st.session_state.dashboard_df = scada_df
        st.session_state.dashboard_file = selected_file
    except Exception as e:
        st.error(t("err_read_failed", e))
        st.stop()
else:
    scada_df = st.session_state.get("dashboard_df")

if scada_df is None or scada_df.empty:
    st.info(t("info_click_load"))
    st.stop()

# ═══════════════════════════════════════════════════════════
# Overview
# ═══════════════════════════════════════════════════════════
st.divider()
time_col = _find_time_col(scada_df)
if time_col:
    scada_df[time_col] = pd.to_datetime(scada_df[time_col], errors='coerce')
    time_min, time_max = scada_df[time_col].min(), scada_df[time_col].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("metric_file"), selected_file)
c2.metric(t("metric_rows"), f"{len(scada_df):,}")
c3.metric(t("metric_time_range"), f"{time_min.strftime('%Y-%m-%d')} ~ {time_max.strftime('%Y-%m-%d')}" if time_col else "N/A")
c4.metric(t("metric_columns"), len(scada_df.columns))
st.info(t("info_csv_not_realtime"), icon="💡")

# ═══════════════════════════════════════════════════════════
# Latest Record
# ═══════════════════════════════════════════════════════════
st.divider()
st.subheader(t("latest_record"))
latest = scada_df.iloc[-1].to_dict()
flow_val = float(latest.get('flow', 0))
inf_cod_val = float(latest.get('inf_cod', 0))
eff_cod_val = float(latest.get('eff_cod', 0))
eff_nh3_val = float(latest.get('eff_nh3', 0))

k1, k2, k3, k4 = st.columns(4)
k1.metric(t("metric_flow"), f"{flow_val:,.0f} m³/h" if flow_val else "N/A")
k2.metric(t("metric_inf_cod"), f"{inf_cod_val:.1f} mg/L" if inf_cod_val else "N/A")
k3.metric(t("metric_eff_cod"), f"{eff_cod_val:.1f} mg/L" if eff_cod_val else "N/A",
          delta=t("delta_compliant") if eff_cod_val <= COD_LIMIT else t("delta_exceeded"),
          delta_color="normal" if eff_cod_val <= COD_LIMIT else "inverse")
k4.metric(t("metric_eff_nh3"), f"{eff_nh3_val:.2f} mg/L" if eff_nh3_val else "N/A",
          delta=t("delta_compliant") if eff_nh3_val <= NH3_LIMIT else t("delta_exceeded"),
          delta_color="normal" if eff_nh3_val <= NH3_LIMIT else "inverse")

# ═══════════════════════════════════════════════════════════
# AI Prediction (cached)
# ═══════════════════════════════════════════════════════════
st.divider()
st.subheader(t("ai_pred_title"))

try:
    gateway = get_adapter().get_gateway()
    seq_len = cfg["tft_seq_len"]
    feat_names = cfg["tft_feature_names"]

    recent = scada_df.tail(seq_len)
    history_data = {}
    for col in feat_names:
        history_data[col] = recent[col].tolist() if col in recent.columns else [0.0] * seq_len

    with st.spinner("Running AI prediction..."):
        pred_result = gateway.predict(history_data)
    predictions = pred_result.get("predictions", {})

    pc1, pc2, pc3 = st.columns(3)
    tft_val = predictions.get("tft_cod")
    lgbm_val = predictions.get("lgbm_cod")
    final_val = predictions.get("final_cod")
    pc1.metric(t("metric_tft_pred"), f"{tft_val:.1f} mg/L" if tft_val is not None else "N/A")
    pc2.metric(t("metric_lgbm_pred"), f"{lgbm_val:.1f} mg/L" if lgbm_val is not None else "N/A")
    pc3.metric(t("metric_final_pred"), f"{final_val:.1f} mg/L" if final_val is not None else "N/A",
               delta=t("delta_divergence", f"{predictions.get('divergence', 0):.1f}") if predictions.get('divergence') else None,
               delta_color="off")

    if final_val is not None:
        if final_val <= COD_LIMIT:
            st.success(t("success_compliant", final_val, COD_LIMIT))
        else:
            st.error(t("error_exceeded", final_val, COD_LIMIT))
except Exception as e:
    st.info(t("info_ai_unavailable", str(e)[:120]))

# ═══════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════
st.divider()
st.subheader(t("history_trend"))

if time_col:
    chart_df = _smart_resample(scada_df.copy(), time_col)
else:
    chart_df = scada_df

tab1, tab2 = st.tabs([t("tab_cod_trend"), t("tab_nh3_trend")])
with tab1:
    cod_cols = [c for c in ['inf_cod', 'eff_cod'] if c in chart_df.columns]
    st.line_chart(chart_df[cod_cols], height=350, use_container_width=True) if cod_cols else st.info(t("info_no_cod_col"))
with tab2:
    nh3_cols = [c for c in ['inf_nh3', 'eff_nh3'] if c in chart_df.columns]
    st.line_chart(chart_df[nh3_cols], height=350, use_container_width=True) if nh3_cols else st.info(t("info_no_nh3_col"))

# ═══════════════════════════════════════════════════════════
# Raw Data
# ═══════════════════════════════════════════════════════════
with st.expander(t("raw_data_table"), expanded=False):
    st.dataframe(scada_df.head(500), use_container_width=True, hide_index=True)
    if len(scada_df) > 500:
        st.caption(f"... {len(scada_df) - 500} more rows")
