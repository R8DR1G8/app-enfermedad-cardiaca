import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("modelo_cardiaco_final.pkl")

st.title("❤️ Predicción de Enfermedad Cardíaca")

st.header("🧾 Ingresá los datos del paciente:")

# Diccionarios de traducción al inglés
opciones_sexo = {"Masculino": "Male", "Femenino": "Female"}
opciones_cp = {
    "Angina típica": "Typical angina",
    "Angina atípica": "Atypical angina",
    "Dolor no anginoso": "Non-anginal pain",
    "Asintomático": "Asymptomatic"
}
opciones_fbs = {
    "Menor a 120 mg/ml": "Lower than 120 mg/ml",
    "Mayor a 120 mg/ml": "Greater than 120 mg/ml"
}
opciones_rest_ecg = {
    "Normal": "Normal",
    "Anormalidad ST-T": "ST-T wave abnormality",
    "Hipertrofia ventricular izquierda": "Left ventricular hypertrophy"
}
opciones_exang = {"No": "No", "Sí": "Yes"}
opciones_slope = {
    "Plano": "Flat",
    "Ascendente": "Upsloping",
    "Descendente": "Downsloping"
}
opciones_vessels = {
    "Cero": "Zero",
    "Uno": "One",
    "Dos": "Two",
    "Tres": "Three",
    "Cuatro": "Four"
}
opciones_thal = {
    "Normal": "Normal",
    "Defecto fijo": "Fixed Defect",
    "Defecto reversible": "Reversable Defect"
}

# Formulario
age = st.number_input("Edad", min_value=1, max_value=120, value=50)
sexo = st.selectbox("Sexo", list(opciones_sexo.keys()))
cp = st.selectbox("Tipo de dolor en el pecho", list(opciones_cp.keys()))
resting_bp = st.number_input("Presión en reposo", min_value=80, max_value=200, value=120)
cholestoral = st.number_input("Colesterol", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Azúcar en ayunas", list(opciones_fbs.keys()))
rest_ecg = st.selectbox("ECG en reposo", list(opciones_rest_ecg.keys()))
max_hr = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=250, value=150)
exang = st.selectbox("¿Angina inducida por ejercicio?", list(opciones_exang.keys()))
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Pendiente ST", list(opciones_slope.keys()))
vessels = st.selectbox("N° de vasos coloreados", list(opciones_vessels.keys()))
thal = st.selectbox("Talasemia", list(opciones_thal.keys()))

# Preparar datos
entrada = pd.DataFrame({
    'age': [age],
    'sex': [1 if opciones_sexo[sexo] == "Male" else 0],
    'chest_pain_type': [opciones_cp[cp]],
    'resting_blood_pressure': [resting_bp],
    'cholestoral': [cholestoral],
    'fasting_blood_sugar': [1 if opciones_fbs[fbs] == "Greater than 120 mg/ml" else 0],
    'rest_ecg': [opciones_rest_ecg[rest_ecg]],
    'Max_heart_rate': [max_hr],
    'exercise_induced_angina': [1 if opciones_exang[exang] == "Yes" else 0],
    'oldpeak': [oldpeak],
    'slope': [opciones_slope[slope]],
    'vessels_colored_by_flourosopy': [opciones_vessels[vessels]],
    'thalassemia': [opciones_thal[thal]]
})

# Codificación one-hot
entrada = pd.get_dummies(entrada)

# Alinear con las columnas del entrenamiento
columnas_entrenadas = modelo.feature_names_in_
for col in columnas_entrenadas:
    if col not in entrada.columns:
        entrada[col] = 0
entrada = entrada[columnas_entrenadas]

# Predicción
if st.button("🔍 Predecir"):
    proba = modelo.predict_proba(entrada)[0][1]
    umbral = 0.3  # umbral ajustado

    if proba >= umbral:
        st.error(f"⚠️ Posible enfermedad cardíaca detectada. (Probabilidad: {proba:.2f})")
    else:
        st.success(f"✅ Sin señales de enfermedad cardíaca. (Probabilidad: {proba:.2f})")

    st.write("🧠 Modelo utilizado: modelo_cardiaco_final.pkl")