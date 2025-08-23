import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --- Carga de Datos ---
# Este es el mismo conjunto de datos de Churn de Telco que analizamos.
# Para facilitar la ejecución, lo he incluido en un formato de diccionario.
# Simplemente copia y pega este bloque completo en tu script de Python.

data = {
    "gender": [
        "Female",
        "Male",
        "Male",
        "Male",
        "Female",
        "Female",
        "Male",
        "Female",
        "Female",
        "Male",
    ],
    "SeniorCitizen": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Partner": ["Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No"],
    "Dependents": ["No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes"],
    "tenure": [1, 34, 2, 45, 2, 8, 22, 10, 28, 62],
    "PhoneService": ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes"],
    "MultipleLines": [
        "No phone service",
        "No",
        "No",
        "No phone service",
        "No",
        "Yes",
        "Yes",
        "No phone service",
        "Yes",
        "No",
    ],
    "InternetService": [
        "DSL",
        "DSL",
        "DSL",
        "DSL",
        "Fiber optic",
        "Fiber optic",
        "Fiber optic",
        "DSL",
        "Fiber optic",
        "DSL",
    ],
    "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No", "Yes"],
    "OnlineBackup": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes"],
    "DeviceProtection": [
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "No",
        "Yes",
        "No",
    ],
    "TechSupport": ["No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No"],
    "StreamingTV": ["No", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No"],
    "StreamingMovies": ["No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No"],
    "Contract": [
        "Month-to-month",
        "One year",
        "Month-to-month",
        "One year",
        "Month-to-month",
        "Month-to-month",
        "Month-to-month",
        "Month-to-month",
        "Month-to-month",
        "One year",
    ],
    "PaperlessBilling": [
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
    ],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Electronic check",
        "Electronic check",
        "Credit card (automatic)",
        "Mailed check",
        "Electronic check",
        "Bank transfer (automatic)",
    ],
    "MonthlyCharges": [
        29.85,
        56.95,
        53.85,
        42.3,
        70.7,
        99.65,
        89.1,
        29.75,
        104.8,
        56.15,
    ],
    "TotalCharges": [
        "29.85",
        "1889.5",
        "108.15",
        "1840.75",
        "151.65",
        "820.5",
        "1949.4",
        "301.9",
        "3046.05",
        "3487.95",
    ],
    "Churn": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No"],
}
# Nota: He usado una pequeña muestra de los datos para que el código sea manejable.
# Si tienes el archivo CSV completo, puedes cargarlo así:
# df = pd.read_csv('Telco-Customer-Churn.csv')

df = pd.DataFrame(data)

# --- Limpieza de Datos ---
# Convertir 'TotalCharges' a numérico, forzando errores a NaN (Not a Number)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Rellenar los valores NaN con la mediana de la columna
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convertir la variable objetivo 'Churn' a 0 y 1
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# --- Definición de Variables ---
# Separar las características (X) de la variable objetivo (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print("Columnas numéricas:", numeric_features)
print("Columnas categóricas:", categorical_features)

# --- Creación de Pipelines de Preprocesamiento ---
# Pipeline para transformar variables numéricas (escalado)
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Pipeline para transformar variables categóricas (codificación one-hot)
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combinar ambos pipelines en un único preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --- División de Datos ---
# Dividir los datos en conjuntos de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Aplicar el preprocesamiento
# Se ajusta el preprocesador con los datos de entrenamiento y se transforman ambos conjuntos
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nForma de los datos de entrenamiento procesados:", X_train_processed.shape)
print("Forma de los datos de prueba procesados:", X_test_processed.shape)
