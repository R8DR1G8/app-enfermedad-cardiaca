# entrenar_modelo_definitivo.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib

# Leer datos
df = pd.read_csv("HeartDiseaseTrain-Test.csv")

# Codificar variables categóricas
df = pd.get_dummies(df)

# Separar variables
y = df['HeartDisease']
X = df.drop('HeartDisease', axis=1)

# Guardar nombres de columnas para la app
columnas = X.columns.tolist()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Escalar
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Entrenar modelo calibrado
base_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
modelo = CalibratedClassifierCV(base_model, cv=5)  # Calibración para mejores probabilidades
modelo.fit(X_train_res, y_train_res)

# Evaluar
y_pred = modelo.predict(X_test)
print("📊 Precisión balanceada:", balanced_accuracy_score(y_test, y_pred))
print("\n🔍 Reporte:\n", classification_report(y_test, y_pred))
print("\n🧱 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Guardar modelo
paquete = {
    'modelo': modelo,
    'columnas': columnas
}
joblib.dump(paquete, "modelo_cardiaco_definitivo.pkl")
