import streamlit as st
import joblib

st.set_page_config(
    page_title="Prediksi Hujan Jawa Timur",
    page_icon="🌧️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model    = joblib.load('model/model_xgboost_jatim.pkl')
    features = joblib.load('model/features_list.pkl')
    info     = joblib.load('model/model_info.pkl')
    return model, features, info

model, features, info = load_artifacts()

st.title("🌧️ Dashboard Prediksi Hujan — Jawa Timur")
st.markdown("Model: **XGBoost Classifier** | Wilayah: **Jawa Timur** | Target: RR > 0 mm")

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC (Test)", f"{info['roc_auc']:.4f}")
col2.metric("CV ROC-AUC", f"{info['cv_roc_auc_mean']:.4f} ± {info['cv_roc_auc_std']:.4f}")
col3.metric("Jumlah Fitur", len(info['features']))