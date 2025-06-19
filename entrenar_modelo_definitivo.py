import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Cargar dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# 2. Detectar nombre correcto de la columna objetivo
posibles_columnas_objetivo = ['HeartDisease', 'Heart Disease', 'target']
columna_objetivo = None

for col in posibles_columnas_objetivo:
    if col in df.columns:
        columna_objetivo = col
        break

if columna_objetivo is None:
    raise ValueError("No se encontró una columna objetivo válida en el archivo CSV.")

# 3. One-hot encoding
df = pd.get_dummies(df)

# 4. Separar variables
y = df[columna_objetivo]
X = df.drop(columna_objetivo, axis=1)

# 🔒 Guardar nombres de columnas para la app
columnas = X.columns.tolist()

# 5. División y balanceo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Escalado
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 7. Entrenamiento
modelo = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
modelo.fit(X_train_res, y_train_res)

# 8. Evaluación
y_pred = modelo.predict(X_test)
print("📊 Precisión balanceada:", balanced_accuracy_score(y_test, y_pred))
print("\n🔍 Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("\n🧱 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# 9. Guardar modelo y columnas
paquete = {
    'modelo': modelo,
    'columnas': columnas
}
joblib.dump(paquete, "modelo_cardiaco_definitivo.pkl")
print("✅ Modelo y columnas guardadas en 'modelo_cardiaco_definitivo.pkl'")
