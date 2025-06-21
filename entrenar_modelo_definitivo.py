import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Cargar dataset
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# 2. Detectar columna objetivo
for posible in ['HeartDisease', 'Heart Disease', 'target']:
    if posible in df.columns:
        columna_objetivo = posible
        break
else:
    raise ValueError("No se encontró una columna objetivo válida.")

# 3. Codificación categórica
df = pd.get_dummies(df)

# 4. Separar variables
y = df[columna_objetivo]
X = df.drop(columns=[columna_objetivo])
columnas_modelo = X.columns.tolist()

# 5. Dividir y balancear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 7. Entrenar modelo
modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
modelo.fit(X_train_scaled, y_train_res)

# 8. Evaluación
y_pred = modelo.predict(X_test_scaled)
print("📊 Precisión balanceada:", balanced_accuracy_score(y_test, y_pred))
print("\n🔍 Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("\n🧱 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# 9. Guardar modelo y scaler
joblib.dump({'modelo': modelo, 'columnas': columnas_modelo}, "modelo_cardiaco_definitivo.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Modelo y scaler guardados como 'modelo_cardiaco_definitivo.pkl' y 'scaler.pkl'")
