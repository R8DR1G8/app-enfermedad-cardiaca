import joblib

# Cargar el modelo grande
modelo = joblib.load("modelo_cardiaco_balanceado.pkl")

# Volver a guardar con mayor compresión
joblib.dump(modelo, "modelo_cardiaco_liviano.pkl", compress=3)

print("✅ Modelo liviano guardado como 'modelo_cardiaco_liviano.pkl'")
