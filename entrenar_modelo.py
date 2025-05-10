import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Cargar el dataset
df = pd.read_csv('heart.csv')

# Mapear columnas de texto a valores numéricos
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

df['chest_pain_type'] = df['chest_pain_type'].map({
    'Typical angina': 0,
    'Atypical angina': 1,
    'Non-anginal pain': 2,
    'Asymptomatic': 3
})

df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
    'Lower than 120 mg/ml': 0,
    'Greater than 120 mg/ml': 1
})

df['rest_ecg'] = df['rest_ecg'].map({
    'Normal': 0,
    'ST-T wave abnormality': 1,
    'Left ventricular hypertrophy': 2
})

df['exercise_induced_angina'] = df['exercise_induced_angina'].map({
    'No': 0,
    'Yes': 1
})

df['slope'] = df['slope'].map({
    'Upsloping': 0,
    'Flat': 1,
    'Downsloping': 2
})

df['vessels_colored_by_flourosopy'] = df['vessels_colored_by_flourosopy'].map({
    'Zero': 0,
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4
})

df['thalassemia'] = df['thalassemia'].map({
    'Normal': 1,
    'Fixed Defect': 2,
    'Reversable Defect': 3,
    'No': 0
})

# Confirmar que no quedan columnas de texto
print("Columnas con texto:", df.select_dtypes(include=['object']).columns.tolist())

# Separar datos (X) y etiquetas (y)
X = df.drop('target', axis=1)
y = df['target']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = MLPClassifier(max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar precisión
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f'✅ Precisión del modelo: {precision:.2f}')

# Guardar modelo entrenado
joblib.dump(modelo, 'modelo_cardiaco.pkl')
