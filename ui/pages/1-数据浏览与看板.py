# ui/pages/1-数据浏览与看板.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# 路径引导
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.core.service_adapter import SystemAdapter
from ui.i18n import t

# 排放限值
try:
    from config_manager import CFG
    COD_LIMIT = getattr(getattr(CFG, 'asm1', None), 'default_cod_limit', 50.0)
    NH3_LIMIT = getattr(getattr(CFG, 'asm1', None), 'default_nh3_limit', 5.0)
except Exception:
    COD_LIMIT = 50.0
    NH3_LIMIT = 5.0

st.header(t("dashboard_header"))
st.caption(t("dashboard_caption"))

# ---- 数据源选择 ----
st.divider()

data_dir = PROJECT_ROOT / "data"
csv_files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
csv_names = [f.name for f in csv_files] if csv_files else []

if not csv_names:
    st.warning(t("dashboard_no_csv"))
    st.stop()

col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    selected_file = st.selectbox(
        t("dashboard_source"), csv_names, index=0, help=t("dashboard_source_help"),
    )
with col_f2:
    st.caption("")
    st.caption("")
    load_btn = st.button(t("btn_load_data"), use_container_width=True, type="primary")

if "dashboard_df" not in st.session_state:
    st.session_state.dashboard_df = None
    st.session_state.dashboard_file = None

if load_btn or st.session_state.dashboard_df is None:
    filepath = data_dir / selected_file
    try:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if filepath.stat().st_size == 0:
            st.error(t("err_file_empty"))
            st.stop()
        if size_mb > 500:
            st.warning(t("warn_file_large", f"{size_mb:.0f}"))
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df.columns = [c.strip() for c in df.columns]
        st.session_state.dashboard_df = df
        st.session_state.dashboard_file = selected_file
    except Exception as e:
        st.error(t("err_read_failed", e))

scada_df = st.session_state.dashboard_df

if scada_df is None or scada_df.empty:
    st.info(t("info_click_load"))
    st.stop()

# ---- 数据概览 ----
st.divider()
time_col = None
for candidate in ['timestamp', 'time', 'datetime', 'date']:
    matches = [c for c in scada_df.columns if candidate in c.lower()]
    if matches:
        time_col = matches[0]
        break

if time_col:
    scada_df[time_col] = pd.to_datetime(scada_df[time_col], errors='coerce')
    time_min = scada_df[time_col].min()
    time_max = scada_df[time_col].max()

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric(t("metric_file"), st.session_state.dashboard_file or "—")
col_info2.metric(t("metric_rows"), f"{len(scada_df):,}")
if time_col:
    col_info3.metric(t("metric_time_range"), f"{time_min.strftime('%Y-%m-%d')} ~ {time_max.strftime('%Y-%m-%d')}")
col_info4.metric(t("metric_columns"), len(scada_df.columns))

st.info(t("info_csv_not_realtime"), icon="💡")

# ---- 最新记录 ----
st.divider()
st.subheader(t("latest_record"))
latest = scada_df.iloc[-1].to_dict()
c1, c2, c3, c4 = st.columns(4)

flow_val = float(latest.get('flow', 0))
inf_cod_val = float(latest.get('inf_cod', 0))
eff_cod_val = float(latest.get('eff_cod', 0))
eff_nh3_val = float(latest.get('eff_nh3', 0))

c1.metric(t("metric_flow"), f"{flow_val:,.0f} m³/h" if flow_val else "N/A")
c2.metric(t("metric_inf_cod"), f"{inf_cod_val:.1f} mg/L" if inf_cod_val else "N/A")
c3.metric(t("metric_eff_cod"), f"{eff_cod_val:.1f} mg/L" if eff_cod_val else "N/A",
          delta=t("delta_compliant") if eff_cod_val <= COD_LIMIT else t("delta_exceeded"),
          delta_color="normal" if eff_cod_val <= COD_LIMIT else "inverse")
c4.metric(t("metric_eff_nh3"), f"{eff_nh3_val:.2f} mg/L" if eff_nh3_val else "N/A",
          delta=t("delta_compliant") if eff_nh3_val <= NH3_LIMIT else t("delta_exceeded"),
          delta_color="normal" if eff_nh3_val <= NH3_LIMIT else "inverse")

# ---- AI 预测 ----
st.divider()
st.subheader(t("ai_pred_title"))

try:
    gateway = SystemAdapter.get_gateway()
    seq_len = getattr(getattr(CFG, 'model', None), 'tft_seq_len', 24)
    feat_names = getattr(getattr(CFG, 'model', None), 'tft_feature_names', [])

    recent = scada_df.tail(seq_len)
    history_data = {}
    for col in feat_names:
        history_data[col] = recent[col].tolist() if col in recent.columns else [0.0] * seq_len
    for extra in ["flow", "do_meas", "eff_cod", "eff_nh3"]:
        if extra not in history_data and extra in recent.columns:
            history_data[extra] = recent[extra].tolist()

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
    st.info(t("info_ai_unavailable", e))

# ---- 趋势图 ----
st.divider()
st.subheader(t("history_trend"))

if time_col:
    df_timed = scada_df.set_index(time_col).sort_index()
    default_end = df_timed.index.max()
    default_start = max(df_timed.index.min(), default_end - pd.Timedelta(days=7))
    date_range = st.date_input(
        t("date_range_label"),
        value=(default_start.date(), default_end.date()),
        min_value=df_timed.index.min().date(),
        max_value=df_timed.index.max().date(),
    )
    if len(date_range) == 2:
        start_ts = pd.Timestamp(date_range[0])
        end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        chart_df = df_timed[(df_timed.index >= start_ts) & (df_timed.index <= end_ts)]
    else:
        chart_df = df_timed

    total_hours = (chart_df.index.max() - chart_df.index.min()).total_seconds() / 3600 if len(chart_df) > 1 else 0
    if total_hours > 24 * 14:
        rule = "1D"; fmt = "%m-%d"
    elif total_hours > 24 * 3:
        rule = "4h"; fmt = "%m-%d %H:00"
    else:
        rule = None; fmt = "%m-%d %H:%M"
    if rule and len(chart_df) > 1:
        chart_df = chart_df.resample(rule).mean()
    chart_df.index = chart_df.index.strftime(fmt)
else:
    chart_df = scada_df

tab1, tab2 = st.tabs([t("tab_cod_trend"), t("tab_nh3_trend")])
with tab1:
    cod_cols = [c for c in ['inf_cod', 'eff_cod'] if c in chart_df.columns]
    if cod_cols:
        st.line_chart(chart_df[cod_cols], height=350, use_container_width=True)
    else:
        st.info(t("info_no_cod_col"))
with tab2:
    nh3_cols = [c for c in ['inf_nh3', 'eff_nh3'] if c in chart_df.columns]
    if nh3_cols:
        st.line_chart(chart_df[nh3_cols], height=350, use_container_width=True)
    else:
        st.info(t("info_no_nh3_col"))

with st.expander(t("raw_data_table"), expanded=False):
    st.dataframe(scada_df, use_container_width=True, hide_index=True)
