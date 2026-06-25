import streamlit as st
import joblib
import numpy as np
import pandas as pd

model    = joblib.load('model/model_xgboost_jatim.pkl')
features = joblib.load('model/features_list.pkl')

st.title("🔮 Prediksi Hujan — Input Manual")
st.markdown("Masukkan data cuaca hari ini untuk memprediksi kemungkinan hujan.")

with st.form("form_prediksi"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Suhu & Kelembapan**")
        Tn    = st.number_input("Suhu Min (Tn) °C", 15.0, 35.0, 24.0)
        Tx    = st.number_input("Suhu Max (Tx) °C", 20.0, 40.0, 32.0)
        Tavg  = st.number_input("Suhu Rata-rata (Tavg) °C", 18.0, 38.0, 28.0)
        RH    = st.number_input("Kelembapan (RH_avg) %", 40.0, 100.0, 78.0)

    with col2:
        st.markdown("**Angin & Penyinaran**")
        ss    = st.number_input("Lama Penyinaran (ss) jam", 0.0, 12.0, 6.0)
        ff_x  = st.number_input("Kec. Angin Max (ff_x) m/s", 0.0, 30.0, 5.0)
        ff_avg= st.number_input("Kec. Angin Rata-rata (ff_avg) m/s", 0.0, 20.0, 3.0)
        ddd_x = st.number_input("Arah Angin Max (ddd_x) °", 0, 360, 180)

    with col3:
        st.markdown("**Data Historis**")
        month = st.selectbox("Bulan", range(1, 13), index=0)
        RR_lag1 = st.number_input("Curah hujan kemarin (RR_lag1) mm", 0.0, 200.0, 0.0)
        RR_lag2 = st.number_input("Curah hujan 2 hari lalu (RR_lag2) mm", 0.0, 200.0, 0.0)
        RR_lag3 = st.number_input("Curah hujan 3 hari lalu (RR_lag3) mm", 0.0, 200.0, 0.0)
        rain_streak = st.number_input("Hari hujan berturut-turut", 0, 30, 0)

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    # Hitung fitur turunan
    from datetime import date
    today = date.today()
    day_of_year  = today.timetuple().tm_yday
    week_of_year = today.isocalendar()[1]
    season = 1 if month in [12,1,2] else (2 if month in [6,7,8] else 3)

    # Asumsi nilai lag RH (bisa dikembangkan jadi input terpisah)
    RH_lag1 = RH_lag2 = RH_lag3 = RH

    # Rolling sederhana: rata-rata dari lag yang tersedia
    RR_roll3 = np.mean([RR_lag1, RR_lag2, RR_lag3])
    RR_roll7 = RR_roll3  # approx
    RH_avg_roll3 = RH_avg_roll7 = RH
    Tavg_roll3 = Tavg_roll7 = Tavg
    ff_avg_roll3 = ff_avg_roll7 = ff_avg

    wind_sin = np.sin(np.radians(ddd_x))
    wind_cos = np.cos(np.radians(ddd_x))

    input_dict = {
        'Tn': Tn, 'Tx': Tx, 'Tavg': Tavg, 'RH_avg': RH,
        'ss': ss, 'ff_x': ff_x, 'ff_avg': ff_avg, 'ddd_x': ddd_x,
        'month': month, 'day_of_year': day_of_year,
        'week_of_year': week_of_year, 'season': season,
        'wind_sin': wind_sin, 'wind_cos': wind_cos,
        'RR_lag1': RR_lag1, 'RR_lag2': RR_lag2, 'RR_lag3': RR_lag3,
        'RH_lag1': RH_lag1, 'RH_lag2': RH_lag2, 'RH_lag3': RH_lag3,
        'RR_roll3': RR_roll3, 'RR_roll7': RR_roll7,
        'RH_avg_roll3': RH_avg_roll3, 'RH_avg_roll7': RH_avg_roll7,
        'Tavg_roll3': Tavg_roll3, 'Tavg_roll7': Tavg_roll7,
        'ff_avg_roll3': ff_avg_roll3, 'ff_avg_roll7': ff_avg_roll7,
        'rain_streak': rain_streak,
    }

    X_input = pd.DataFrame([input_dict])[features]
    prob  = model.predict_proba(X_input)[0][1]
    label = model.predict(X_input)[0]

    st.divider()
    if label == 1:
        st.error(f"🌧️ **HUJAN** — Probabilitas: {prob:.1%}")
    else:
        st.success(f"☀️ **TIDAK HUJAN** — Probabilitas hujan: {prob:.1%}")
    st.progress(float(prob))