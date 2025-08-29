# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


def load_data(filepath):
    """
    Carga el dataset desde un archivo CSV.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset cargado. Forma: {df.shape}")
    return df


# %%
def clean_data(df):
    """
    Limpieza inicial del dataset.
    """
    # Convertir TotalCharges a numérico y manejar valores vacíos
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)

    # Eliminar customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    print(f"[INFO] Limpieza completada. Valores nulos: {df.isnull().sum().sum()}")
    return df


# %%
def encode_categorical_features(df, target_col="Churn"):
    """
    Aplica codificación a variables categóricas.
    - One-Hot Encoding para variables nominales con >2 categorías.
    - Label Encoding para binarias (Yes/No).
    """
    # Separar variables categóricas y numéricas
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # Variables binarias: Yes/No
    binary_cols = [
        col for col in categorical_cols if set(df[col].unique()) == {"Yes", "No"}
    ]

    # Variables nominales con más de 2 categorías
    nominal_cols = [col for col in categorical_cols if col not in binary_cols]

    # Crear copia para transformar
    df_encoded = df.copy()

    # Label Encoding para binarias
    label_encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])  # Yes=1, No=0
        label_encoders[col] = le

    # One-Hot Encoding para nominales
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, drop_first=True)

    print(f"[INFO] Codificación completada. Nuevas columnas: {df_encoded.shape[1]}")
    return df_encoded, label_encoders


# %%
def split_and_scale_data(df, target_col="Churn", test_size=0.3, val_size=0.5):
    """
    Divide los datos en train, val, test y escala las variables numéricas.
    - train: 70%
    - val: 15% (de 30% restante)
    - test: 15%
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].apply(lambda x: 1 if x == "Yes" else 0)  # Codificar target

    # Dividir: 70% entrenamiento, 30% temporal
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Dividir temporal en validación y prueba (15% cada uno)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
    )

    # Identificar variables numéricas
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Escalar solo las numéricas
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    print(f"[INFO] División realizada:")
    print(f"  Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  Validación: {X_val.shape[0]} muestras")
    print(f"  Prueba: {X_test.shape[0]} muestras")

    # Guardar el scaler para producción
    joblib.dump(scaler, "../models/scaler.pkl")
    print("[INFO] Scaler guardado en '../models/scaler.pkl'")

    return (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)


# %%
def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Guarda los conjuntos procesados en la carpeta data/processed/
    """
    os.makedirs("../data/processed", exist_ok=True)

    X_train.to_csv("../data/processed/X_train.csv", index=False)
    X_val.to_csv("../data/processed/X_val.csv", index=False)
    X_test.to_csv("../data/processed/X_test.csv", index=False)
    y_train.to_csv("../data/processed/y_train.csv", index=False)
    y_val.to_csv("../data/processed/y_val.csv", index=False)
    y_test.to_csv("../data/processed/y_test.csv", index=False)

    print("[INFO] Datos procesados guardados en '../data/processed/'")


# %%
def main():
    """
    Función principal del preprocesamiento.
    """
    # Cargar y limpiar
    df = load_data("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)

    # Codificar
    df_encoded, le_dict = encode_categorical_features(df)

    # Dividir y escalar
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(df_encoded)

    # Guardar
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("[INFO] Preprocesamiento completado con éxito.")


# %%
if __name__ == "__main__":
    main()
