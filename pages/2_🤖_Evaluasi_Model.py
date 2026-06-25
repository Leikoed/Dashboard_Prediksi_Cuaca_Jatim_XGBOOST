import streamlit as st
import joblib

info = joblib.load('model/model_info.pkl')

st.title("🤖 Evaluasi Model XGBoost")

col1, col2 = st.columns(2)
col1.metric("ROC-AUC Test", f"{info['roc_auc']:.4f}")
col2.metric("Best Iteration", info['best_iteration'])

st.subheader("Confusion Matrix & ROC Curve")
st.image("assets/evaluasi_model.png")

st.subheader("Feature Importance (XGBoost Gain)")
st.image("assets/feature_importance_xgb.png")

st.subheader("SHAP Feature Importance")
tab1, tab2 = st.tabs(["Bar Chart", "Beeswarm"])
with tab1:
    st.image("assets/shap_importance.png")
with tab2:
    st.image("assets/shap_beeswarm.png")