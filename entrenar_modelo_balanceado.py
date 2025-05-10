import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Cargar el dataset
df = pd.read_csv("heart_avanzado.csv")

# 2. Convertir variables categóricas a numéricas
df_encoded = pd.get_dummies(df.drop("Patient ID", axis=1))

# 3. Definir variables X (entrada) e y (salida)
X = df_encoded.drop("Heart Attack Risk", axis=1)
y = df_encoded["Heart Attack Risk"]

# 4. Aplicar SMOTE para balancear
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# 5. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# 6. Entrenar modelo
modelo = RandomForestClassifier(n_estimators=150, random_state=42)
modelo.fit(X_train, y_train)

# 7. Evaluación
y_pred = modelo.predict(X_test)
print(f"\n📊 Precisión balanceada: {accuracy_score(y_test, y_pred):.3f}")
print("\n🔍 Reporte de clasificación:")
print(classification_report(y_test, y_pred))
print("\n🧱 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# 8. Guardar modelo
joblib.dump(modelo, "modelo_cardiaco_balanceado.pkl")
print("✅ Modelo balanceado guardado como 'modelo_cardiaco_balanceado.pkl'")
