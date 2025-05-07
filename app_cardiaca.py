import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Cargar el modelo entrenado
modelo = joblib.load('modelo_cardiaco.pkl')

st.title("🫀 Predicción de Enfermedades Cardíacas")
st.markdown("Ingrese los datos del paciente para evaluar su riesgo cardíaco:")

# Entradas del usuario
age = st.number_input("Edad", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sexo", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
cp = st.selectbox("Tipo de dolor en el pecho", options=[0, 1, 2, 3], format_func=lambda x: [
    "Angina típica", "Angina atípica", "Dolor no anginoso", "Asintomático"][x])
resting_bp = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Azúcar en ayunas > 120 mg/dl", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
rest_ecg = st.selectbox("Electrocardiograma en reposo", options=[0, 1, 2], format_func=lambda x: [
    "Normal", "Anormalidad ST-T", "Hipertrofia ventricular"][x])
thalach = st.number_input("Frecuencia cardíaca máxima alcanzada", min_value=60, max_value=220, value=150)
exang = st.selectbox("¿Angina inducida por ejercicio?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
oldpeak = st.number_input("Depresión ST", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Pendiente del ST", options=[0, 1, 2], format_func=lambda x: [
    "Ascendente", "Plana", "Descendente"][x])
vessels = st.selectbox("Nº de vasos coloreados (fluoroscopía)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Talasemia", options=[1, 2, 3, 0], format_func=lambda x: {
    1: "Normal", 2: "Defecto fijo", 3: "Defecto reversible", 0: "Desconocido"}[x])

# Crear DataFrame de entrada con nombres correctos
entrada = pd.DataFrame([[age, sex, cp, resting_bp, chol, fbs, rest_ecg,
                         thalach, exang, oldpeak, slope, vessels, thal]],
                       columns=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral',
                                'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate',
                                'exercise_induced_angina', 'oldpeak', 'slope',
                                'vessels_colored_by_flourosopy', 'thalassemia'])

# Botón para predecir
if st.button("🔍 Predecir"):
    resultado = modelo.predict(entrada)

    if resultado[0] == 1:
        st.error("⚠️ Riesgo de enfermedad cardíaca detectado")
    else:
        st.success("✅ No se detecta enfermedad cardíaca")

    # Guardar resultado en CSV
    os.makedirs("resultados", exist_ok=True)
    resultado_df = entrada.copy()
    resultado_df['Predicción'] = resultado[0]
    archivo_csv = "resultados/predicciones.csv"

    if os.path.exists(archivo_csv):
        resultado_df.to_csv(archivo_csv, mode='a', header=False, index=False)
    else:
        resultado_df.to_csv(archivo_csv, index=False)

    st.info("📁 Resultado guardado en 'resultados/predicciones.csv'")

# Mostrar gráficas si hay resultados guardados
if os.path.exists("resultados/predicciones.csv"):
    st.markdown("### 📊 Estadísticas de predicciones")

    df_resultados = pd.read_csv("resultados/predicciones.csv")
    conteo = df_resultados['Predicción'].value_counts()

    # Gráfico de torta
    fig1, ax1 = plt.subplots()
    ax1.pie(conteo, labels=conteo.index.map(lambda x: "Con enfermedad" if x == 1 else "Sin enfermedad"),
            autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Gráfico de barras
    st.bar_chart(conteo.rename(index={0: "Sin enfermedad", 1: "Con enfermedad"}))
