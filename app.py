import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo previamente entrenado
modelo = joblib.load('modelo_diabetes.pkl')

# Título
st.title("🩺 Predicción de Diabetes con IA")

# Formulario de entrada
st.header("🧾 Ingresá los datos del paciente:")

pregnancies = st.number_input("N° de embarazos", min_value=0, max_value=20, value=1)
glucose = st.number_input("Nivel de glucosa", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Presión arterial", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Grosor de piel (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulina", min_value=0, max_value=900, value=85)
bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=0.0, max_value=80.0, value=25.0)
dpf = st.number_input("Índice genético (DPF)", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Edad", min_value=1, max_value=120, value=30)

# Botón para predecir
if st.button("🔍 Predecir"):
    entrada = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    resultado = modelo.predict(entrada)

    if resultado[0] == 1:
        st.error("⚠️ Posible diagnóstico: Diabetes")
    else:
        st.success("✅ Diagnóstico: Sin diabetes")
