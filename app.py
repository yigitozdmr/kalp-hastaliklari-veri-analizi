import streamlit as st
import numpy as np
import joblib

# Sayfa ayarları
st.set_page_config(
    page_title="Kalp Hastalığı Tahmin Sistemi",
    layout="centered"
)

st.title("🫀 Kalp Hastalığı Risk Tahmini")
st.write(
    "Bu uygulama, makine öğrenmesi modeli kullanarak "
    "kalp hastalığı riskini tahmin eder."
)

# Modeli yükle
model = joblib.load("model/heart_model.joblib")

st.header("📋 Hasta Bilgileri")

age = st.number_input("Yaş", min_value=1, max_value=120, value=50)
sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
cp = st.selectbox("Göğüs Ağrısı Tipi (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Dinlenme Kan Basıncı", value=120)
chol = st.number_input("Kolesterol", value=200)
fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl", [0, 1])
restecg = st.selectbox("EKG Sonucu", [0, 1, 2])
thalach = st.number_input("Maksimum Kalp Atış Hızı", value=150)
exang = st.selectbox("Egzersizle Angina", [0, 1])
oldpeak = st.number_input("ST Depresyonu", value=1.0)
slope = st.selectbox("ST Segment Eğimi", [0, 1, 2])
ca = st.selectbox("Büyük Damar Sayısı", [0, 1, 2, 3])
thal = st.selectbox("Thal Değeri", [0, 1, 2, 3])

sex = 1 if sex == "Erkek" else 0

# Tahmin butonu
if st.button("🔍 Tahmin Yap"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak,
                           slope, ca, thal]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ Kalp hastalığı riski YÜKSEK")
    else:
        st.success("✅ Kalp hastalığı riski DÜŞÜK")
        
