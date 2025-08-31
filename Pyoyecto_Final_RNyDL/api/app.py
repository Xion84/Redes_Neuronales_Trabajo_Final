"""
API de Predicción de Churn - Proyecto Final de Redes Neuronales
Maestría en Inteligencia de Negocios y Análisis de Datos
Autores: Hubert Gutiérrez, Danilo Matus, Enllely Roque
Profesor: Dr. Vladimir Gutiérrez
"""

# %%
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Inicializar la app Flask
app = Flask(__name__, static_folder="web", static_url_path="/")
CORS(app)  # Habilita CORS para todos los orígenes

# Rutas a los modelos y scaler
MODEL_PATH = "../models/MLP-2.h5"
SCALER_PATH = "../models/scaler.pkl"

# Cargar el modelo y el scaler al iniciar la app
print("[INFO] Cargando modelo y scaler...")
model = None
scaler = None

# Cargar el modelo y el scaler al iniciar la app
print("[INFO] Cargando modelo y scaler...")
model = None
scaler = None

# Cargar el modelo y el scaler al iniciar la app
print("[INFO] Cargando modelo y scaler...")
model = None
scaler = None

# Obtener las características esperadas
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "processed", "X_train.csv")
print(f"[INFO] Buscando X_train.csv en: {data_path}")

# %%

EXPECTED_FEATURES = []
try:
    df_temp = pd.read_csv(data_path)
    EXPECTED_FEATURES = df_temp.columns.tolist()
    print(
        f"[INFO] X_train.csv cargado. Número de características: {len(EXPECTED_FEATURES)}"
    )
    print(f"[INFO] Columnas esperadas: {EXPECTED_FEATURES}")
except Exception as e:
    print(f"[ERROR] No se pudo cargar X_train.csv: {e}")
    EXPECTED_FEATURES = []

try:
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    print(
        f"[INFO] Modelo cargado correctamente. Capas: {[layer.name for layer in model.layers]}"
    )
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    model = None

# %%
try:
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Scaler cargado correctamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el scaler: {e}")
    scaler = None

# %%


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud"""
    if model is not None and scaler is not None and EXPECTED_FEATURES:
        return jsonify({"status": "OK", "message": "Modelo listo para predicciones"})
    else:
        return jsonify(
            {"status": "ERROR", "message": "Modelo o scaler no cargado"}
        ), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de predicción"""
    if model is None or scaler is None or not EXPECTED_FEATURES:
        return jsonify(
            {"error": "Modelo, scaler o características no disponibles"}
        ), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se proporcionaron datos"}), 400

        # Convertir a DataFrame
        df = pd.DataFrame([data])

        # Validación de campos requeridos
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

        # Asegurar que tenga todas las columnas de X_train y en el mismo orden
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_FEATURES]  # Reordenar para que coincida

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

        # Definir churn_prediction
        churn_prediction = bool(churn_probability > 0.5)

        return jsonify(
            {
                "churn_probability": round(churn_probability, 4),
                "churn_prediction": churn_prediction,
            }
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    """Servir la página web"""
    return app.send_static_file("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# %%
