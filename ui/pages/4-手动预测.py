# pages/4-手动预测.py

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.i18n import t

st.title(t("predict_title"))

# --- 导入依赖 ---
try:
    from ui.core.service_adapter import SystemAdapter
    from config_manager import CFG
except ImportError as e:
    st.error(t("err_import_failed", e))
    st.stop()

# --- 模型状态检查 ---
try:
    predictor = SystemAdapter.get_gateway()
    tft_loaded = getattr(predictor, 'tft_session', None) is not None
    lgbm_loaded = getattr(predictor, 'lgbm_model', None) is not None
    tft_file_ok = False
    lgbm_file_ok = False
    if hasattr(predictor, 'tft_path') and predictor.tft_path:
        tft_path = Path(predictor.tft_path)
        tft_file_ok = tft_path.exists() and tft_path.stat().st_size > 0
    if hasattr(predictor, 'lgbm_path') and predictor.lgbm_path:
        lgbm_path = Path(predictor.lgbm_path)
        lgbm_file_ok = lgbm_path.exists() and lgbm_path.stat().st_size > 0
    model_ready = tft_loaded or lgbm_loaded
    need_refresh = (tft_file_ok and not tft_loaded) or (lgbm_file_ok and not lgbm_loaded)
except Exception as e:
    st.error(t("err_model_init", e))
    model_ready = False
    tft_loaded = lgbm_loaded = tft_file_ok = lgbm_file_ok = need_refresh = False

# ---- 特征元数据 ----
try:
    FEATURE_NAMES = CFG.model.tft_feature_names
except Exception:
    FEATURE_NAMES = ["inf_cod", "inf_nh3", "DO_reactor", "MLSS_reactor"]

FEATURE_META = {
    "inf_cod":       (t("feat_inf_cod"),     "mg/L", "化学需氧量，反映进水有机污染物浓度"),
    "inf_nh3":       (t("feat_inf_nh3"),     "mg/L", "进水氨氮浓度，影响硝化反应负荷"),
    "DO_reactor":    (t("feat_do_reactor"),  "mg/L", "好氧区溶解氧浓度，影响COD去除与硝化效率"),
    "MLSS_reactor":  (t("feat_mlss_reactor"),"mg/L", "混合液悬浮固体浓度，代表活性污泥微生物量"),
    "flow":          (t("feat_flow"),         "m³/h","单位时间进入厂区的污水体积"),
    "do_meas":       (t("feat_do_meas"),     "mg/L", "溶解氧仪表在线测量值"),
    "eff_cod":       (t("feat_eff_cod"),     "mg/L", "处理后出水的化学需氧量"),
    "eff_nh3":       (t("feat_eff_nh3"),     "mg/L", "处理后出水的氨氮浓度"),
}

DEFAULTS = {
    "inf_cod": 300.0, "inf_nh3": 30.0, "DO_reactor": 2.0, "MLSS_reactor": 3000.0,
    "flow": 10000.0, "do_meas": 2.0, "eff_cod": 40.0, "eff_nh3": 3.0,
}

LIMITS = {
    "inf_cod":      (0, 2000,   t("feat_inf_cod")),
    "inf_nh3":      (0, 100,    t("feat_inf_nh3")),
    "DO_reactor":   (0, 15,     t("feat_do_reactor")),
    "MLSS_reactor": (500, 15000,t("feat_mlss_reactor")),
    "flow":         (0, 100000, t("feat_flow")),
    "do_meas":      (0, 15,     t("feat_do_meas")),
}

# ---- 模型状态卡片 ----
st.divider()
if not model_ready:
    st.info(t("info_model_not_ready"), icon="💡")
    st.page_link("pages/2-模型训练与校准.py", label=t("link_to_training"), icon="🏃", use_container_width=True)
    st.stop()

col_tft, col_lgbm = st.columns(2)
with col_tft:
    if tft_loaded:
        st.success(t("tft_loaded_ok"))
    elif tft_file_ok:
        st.warning(t("tft_file_but_not_loaded"))
    else:
        st.error(t("tft_not_found"))
with col_lgbm:
    if lgbm_loaded:
        st.success(t("lgbm_loaded_ok"))
    elif lgbm_file_ok:
        st.warning(t("lgbm_file_but_not_loaded"))
    else:
        st.error(t("lgbm_not_found"))

if need_refresh:
    st.info(t("refresh_hint"), icon="🔄")

# ---- 输入表单 ----
st.divider()
st.subheader(t("process_params"))

with st.expander(t("usage_guide_expander"), expanded=False):
    st.markdown(t("usage_guide_text", CFG.model.tft_seq_len))

with st.form(key="prediction_form"):
    cols = st.columns(2)
    user_inputs = {}
    for i, feature in enumerate(FEATURE_NAMES):
        col_idx = i % 2
        with cols[col_idx]:
            meta = FEATURE_META.get(feature, (feature, "", ""))
            name, unit, desc = meta[0], meta[1], meta[2]
            if "flow" in feature.lower():
                step, vmin, vmax = 100.0, 0.0, 100000.0
            elif "cod" in feature.lower():
                step, vmin, vmax = 5.0, 0.0, 2000.0
            elif "nh3" in feature.lower():
                step, vmin, vmax = 1.0, 0.0, 100.0
            elif "do" in feature.lower():
                step, vmin, vmax = 0.1, 0.0, 15.0
            elif "mlss" in feature.lower():
                step, vmin, vmax = 100.0, 500.0, 15000.0
            else:
                step, vmin, vmax = 1.0, None, None
            user_inputs[feature] = st.number_input(
                label=name, value=DEFAULTS.get(feature, 0.0),
                step=step, min_value=vmin, max_value=vmax,
                help=f"{desc}　　{unit}: {unit}　　[{vmin}, {vmax}]",
                format="%.1f" if step < 1.0 else "%.0f",
            )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submitted = st.form_submit_button(t("btn_predict"), type="primary", use_container_width=True)
    st.caption(t("predict_caption", CFG.model.tft_seq_len))

# ---- 处理预测 ----
if submitted:
    errors = []
    for feat, val in user_inputs.items():
        if feat in LIMITS:
            lo, hi, label = LIMITS[feat]
            if val < lo or val > hi:
                errors.append(t("err_param_out_of_range", label, feat, lo, hi))
    if errors:
        st.error(t("err_param_anomaly") + "\n".join(f"- {e}" for e in errors))
        st.stop()

    with st.spinner(t("spinner_predicting")):
        try:
            seq_len = CFG.model.tft_seq_len
            history_data = {}
            for feat in FEATURE_NAMES:
                history_data[feat] = [user_inputs[feat]] * seq_len
            for extra_col in ["flow", "do_meas", "eff_cod", "eff_nh3"]:
                if extra_col not in history_data:
                    history_data[extra_col] = [DEFAULTS.get(extra_col, 0.0)] * seq_len

            result = predictor.predict(history_data)
            st.divider()
            st.subheader(t("predict_result_title"))

            predictions = result.get("predictions", {})
            tft_cod = predictions.get("tft_cod")
            lgbm_cod = predictions.get("lgbm_cod")
            final_cod = predictions.get("final_cod")
            divergence = predictions.get("divergence")
            status = result.get("status", "unknown")
            warnings = result.get("warnings", [])
            importance = result.get("feature_importance", {})
            inference_ms = result.get("inference_time_ms", 0)

            col_t, col_l, col_f = st.columns(3)
            with col_t:
                st.metric(t("metric_tft_pred"), f"{tft_cod:.1f} mg/L" if tft_cod is not None else "—")
            with col_l:
                st.metric(t("metric_lgbm_pred"), f"{lgbm_cod:.1f} mg/L" if lgbm_cod is not None else "—")
            with col_f:
                st.metric(t("metric_final_pred"), f"{final_cod:.1f} mg/L" if final_cod is not None else "—",
                          delta=t("delta_divergence", f"{divergence:.1f}") if divergence is not None else None,
                          delta_color="off")

            st.caption(t("inference_time", inference_ms))
            st.divider()

            if final_cod is not None:
                cod_limit = CFG.asm1.default_cod_limit
                if final_cod <= cod_limit:
                    st.success(t("success_cod_compliant", final_cod, cod_limit))
                else:
                    st.error(t("error_cod_exceeded", final_cod, cod_limit))

            if status == "warning_divergence":
                st.warning(t("warn_model_divergence"))

            with st.expander(t("detail_expander")):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.write(t("detail_predictions"))
                    st.write(t("detail_tft_pred", tft_cod) if tft_cod else t("detail_tft_na"))
                    st.write(t("detail_lgbm_pred", lgbm_cod) if lgbm_cod else t("detail_lgbm_na"))
                    st.write(t("detail_final_pred", final_cod) if final_cod else t("detail_final_na"))
                    st.write(t("detail_divergence", divergence) if divergence else t("detail_div_na"))
                with dc2:
                    st.write(t("detail_status"))
                    st.write(t("detail_status_code", status))
                    st.write(t("detail_inference_time", inference_ms))
                    st.write(t("detail_warnings", len(warnings)))

            if importance:
                with st.expander(t("importance_expander")):
                    imp_df = pd.DataFrame({
                        t("importance_col_feature"): list(importance.keys()),
                        t("importance_col_weight"): list(importance.values()),
                    }).sort_values(t("importance_col_weight"), ascending=False)
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)

            if warnings:
                with st.expander(t("warnings_expander", len(warnings))):
                    for i, w in enumerate(warnings, 1):
                        st.caption(f"{i}. {w}")

        except Exception as e:
            st.error(t("err_predict_service", e))
            st.caption(t("err_type", type(e).__name__))
