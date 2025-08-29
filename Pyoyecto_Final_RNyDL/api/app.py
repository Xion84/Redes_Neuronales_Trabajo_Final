#
"""
API de Predicción de Churn - Proyecto Final de Redes Neuronales
Maestría en Análisis de Datos e Inteligencia de Negocios
Autores: Hubert Gutiérrez, Danilo Matus, Enllely Roque
Profesor: Dr. Vladimir Gutiérrez
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import sys

# Inicializar la app Flask
app = Flask(__name__, static_folder="web", static_url_path="")

# Rutas a los modelos y scaler
MODEL_PATH = "../models/MLP-2.h5"
SCALER_PATH = ".../models/scaler.pkl"

# Cargar el modelo y el scaler al iniciar la app
print("[INFO] Cargando modelo y scaler...")
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Modelo y scaler cargados correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo o scaler: {e}")
    model = None
    scaler = None

# Obtener las características esperadas (después del one-hot)
#
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "processed", "X_train.csv")

print(f"[INFO] Buscando X_train.csv en: {data_path}")

# Leer el archivo
try:
    EXPECTED_FEATURES = pd.read_csv(data_path).columns.tolist()
    print("[INFO] Archivo X_train.csv cargado correctamente.")
except FileNotFoundError as e:
    print(f"[ERROR] No se encontró el archivo: {e}")
    raise


@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint de salud: Verifica si la API está funcionando.
    """
    if model is not None and scaler is not None:
        return jsonify({"status": "OK", "message": "Modelo listo para predicciones"})
    else:
        return jsonify({"status": "ERROR", "message": "Modelo no cargado"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint de predicción.
    Espera un JSON con las características del cliente.
    Retorna la probabilidad de churn y la predicción.
    """
    if model is None or scaler is None:
        return jsonify({"error": "Modelo no disponible"}), 500

    try:
        # Obtener datos del request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos"}), 400

        # Convertir a DataFrame
        df = pd.DataFrame([data])

        # Validar campos requeridos
        required_fields = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
        ]
        for field in required_fields:
            if field not in df.columns:
                return jsonify({"error": f"Campo faltante: {field}"}), 400

        # Preprocesamiento: One-Hot Encoding
        categorical_cols = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Asegurar que tenga todas las columnas del entrenamiento
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_FEATURES]

        # Convertir SeniorCitizen a int
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        # Manejar TotalCharges (convertir a numérico)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(
            0
        )

        # Escalar variables numéricas
        numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        df[numeric_features] = scaler.transform(df[numeric_features])

        # Convertir a float32 para TensorFlow
        X = df.astype("float32").values

        # Predicción
        prediction = model.predict(X, verbose=0)
        churn_probability = float(prediction[0][0])
        churn_prediction = bool(churn_probability > 0.5)

        # Respuesta
        return jsonify(
            {
                "churn_probability": round(churn_probability, 4),
                "churn_prediction": churn_prediction,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def home():
    """
    Página web principal.
    """
    return app.send_static_file("index.html")


if __name__ == "__main__":
    # Usar el puerto de Heroku o 5000 localmente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
