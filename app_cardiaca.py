import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo previamente entrenado
modelo = joblib.load('modelo_cardiaco_balanceado.pkl')

# Título de la app
st.title("❤️ Predicción de Enfermedad Cardíaca")

# Formulario de entrada
st.header("🧾 Ingresá los datos del paciente:")

age = st.number_input("Edad", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sexo", ["Male", "Female"])
cp = st.selectbox("Tipo de dolor en el pecho", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
resting_bp = st.number_input("Presión en reposo", min_value=80, max_value=200, value=120)
cholestoral = st.number_input("Colesterol", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Azúcar en sangre en ayunas", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"])
rest_ecg = st.selectbox("ECG en reposo", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_hr = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=250, value=150)
exang = st.selectbox("Angina inducida por ejercicio", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Pendiente del segmento ST", ["Flat", "Upsloping", "Downsloping"])
vessels = st.selectbox("N° de vasos coloreados", ["Zero", "One", "Two", "Three", "Four"])
thal = st.selectbox("Talasemia", ["Normal", "Fixed Defect", "Reversable Defect"])

# Convertir variables categóricas
entrada = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'Male' else 0],
    'chest_pain_type': [cp],
    'resting_blood_pressure': [resting_bp],
    'cholestoral': [cholestoral],
    'fasting_blood_sugar': [1 if fbs == "Greater than 120 mg/ml" else 0],
    'rest_ecg': [rest_ecg],
    'Max_heart_rate': [max_hr],
    'exercise_induced_angina': [1 if exang == "Yes" else 0],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'vessels_colored_by_flourosopy': [vessels],
    'thalassemia': [thal]
})

# One-hot encoding (tiene que coincidir con el entrenamiento)
entrada = pd.get_dummies(entrada)

# Asegurarse de que tenga las mismas columnas que durante el entrenamiento
columnas_entrenadas = modelo.feature_names_in_
for col in columnas_entrenadas:
    if col not in entrada.columns:
        entrada[col] = 0
entrada = entrada[columnas_entrenadas]

# Botón de predicción
if st.button("🔍 Predecir"):
    resultado = modelo.predict(entrada)
    if resultado[0] == 1:
        st.error("⚠️ Posible enfermedad cardíaca detectada.")
    else:
        st.success("✅ Sin señales de enfermedad cardíaca.")
        st.write("Modelo: modelo_cardiaco_balanceado.pkl")