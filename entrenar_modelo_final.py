import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Cargar datos
df = pd.read_csv('heart_avanzado.csv')

# Convertir variables categóricas a numéricas
df = pd.get_dummies(df)

# Eliminar columnas no necesarias
df = df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere'], errors='ignore')

# Verificar columna objetivo
y = df['Heart Attack Risk']
X = df.drop('Heart Attack Risk', axis=1)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Balancear con SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Escalar datos
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test)

# Entrenar modelo (priorizando recall con parámetros)
modelo = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
modelo.fit(X_train_bal, y_train_bal)

# Evaluación
y_pred = modelo.predict(X_test)

print("📊 Precisión balanceada:", balanced_accuracy_score(y_test, y_pred))
print("\n🔍 Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("\n🧱 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Guardar modelo final
joblib.dump(modelo, 'modelo_cardiaco_final.pkl')    