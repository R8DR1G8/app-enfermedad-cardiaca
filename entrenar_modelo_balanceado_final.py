import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Cargar el dataset
df = pd.read_csv("heart_avanzado.csv")

# 2. Renombrar la columna objetivo para facilitar
df = df.rename(columns={'Heart Attack Risk': 'target'})

# 3. Eliminar columnas que no ayudan
df = df.drop(['Patient ID', 'Country', 'Continent', 'Hemisphere'], axis=1)

# 4. Convertir texto a dummies (One-hot encoding)
df = pd.get_dummies(df)

# 5. Separar variables
X = df.drop("target", axis=1)
y = df["target"]

# 6. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Aplicar SMOTE para balancear
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 8. División de datos
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 9. Entrenar modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 10. Evaluar
y_pred = modelo.predict(X_test)
print("📊 Precisión balanceada:", balanced_accuracy_score(y_test, y_pred))
print("\n🔍 Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("\n🧱 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# 11. Guardar modelo
joblib.dump(modelo, "modelo_cardiaco_balanceado_final.pkl")