import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y scaler
paquete = joblib.load("modelo_cardiaco_definitivo.pkl")
modelo = paquete["modelo"]
columnas_entrenadas = paquete["columnas"]
scaler = joblib.load("scaler.pkl")

# Configuración de la app
st.set_page_config(page_title="Predicción Cardíaca", page_icon="❤️")
st.title("❤️ Predicción de Enfermedad Cardíaca")
st.header("🧾 Ingresa los datos del paciente:")

# Opciones categóricas traducidas
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
    "Ascendente": "Upsloping",
    "Plano": "Flat",
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

# Formularios de entrada
edad = st.number_input("Edad", 1, 120, 50)
sexo = st.selectbox("Sexo", list(opciones_sexo.keys()))
cp = st.selectbox("Tipo de dolor en el pecho", list(opciones_cp.keys()))
rest_bp = st.number_input("Presión en reposo (mm Hg)", 80, 200, 120)
colesterol = st.number_input("Colesterol (mg/dl)", 100, 600, 250)
fbs = st.selectbox("Azúcar en ayunas", list(opciones_fbs.keys()))
rest_ecg = st.selectbox("ECG en reposo", list(opciones_rest_ecg.keys()))
max_hr = st.number_input("Frecuencia cardíaca máxima", 60, 250, 150)
exang = st.selectbox("¿Angina inducida por ejercicio?", list(opciones_exang.keys()))
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Pendiente del segmento ST", list(opciones_slope.keys()))
vessels = st.selectbox("N° de vasos coloreados", list(opciones_vessels.keys()))
thal = st.selectbox("Talasemia", list(opciones_thal.keys()))

# Crear dataframe de entrada
entrada = pd.DataFrame({
    'age': [edad],
    'sex': [1 if opciones_sexo[sexo] == "Male" else 0],
    'chest_pain_type': [opciones_cp[cp]],
    'resting_blood_pressure': [rest_bp],
    'cholestoral': [colesterol],
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

# Asegurar columnas esperadas por el modelo
for col in columnas_entrenadas:
    if col not in entrada.columns:
        entrada[col] = 0
entrada = entrada[columnas_entrenadas]

# Escalar la entrada
entrada_scaled = scaler.transform(entrada)

# Botón de predicción
if st.button("🔍 Predecir"):
    probabilidad = modelo.predict_proba(entrada_scaled)[0][1]
    
    st.subheader("🔎 Resultado del análisis:")
    if probabilidad >= 0.77:
        st.error(f"⚠️ Alta probabilidad de enfermedad cardíaca. Recomendado acudir al hospital.\n\nProbabilidad: {probabilidad:.2f}")
    elif probabilidad > 0.63:
        st.warning(f"⁉️ Riesgo moderado. Considere una revisión médica.\n\nProbabilidad: {probabilidad:.2f}")
    elif probabilidad > 0.55:
        st.info(f"🤨 Riesgo leve. Manténgase en observación.\n\nProbabilidad: {probabilidad:.2f}")
    else:
        st.success(f"✅ Sin señales significativas de enfermedad cardíaca.\n\nProbabilidad: {probabilidad:.2f}")
    
    st.caption("📦 Modelo: modelo_cardiaco_definitivo.pkl | Escalado: scaler.pkl")