"""
API de Predicción de Churn - Proyecto Final de Redes Neuronales
Maestría en Inteligencia de Negocios y Análisis de Datos
Autores: Hubert Gutiérrez, Danilo Matus, Enllely Roque
Profesor: Dr. Vladimir Gutiérrez
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

# --- INICIO: Configuración de rutas ---
# Directorio actual (donde está app.py)
api_dir = os.path.abspath(os.path.dirname(__file__))

# Directorio raíz del proyecto (un nivel arriba de 'api')
project_root = os.path.dirname(api_dir)

# Rutas a los archivos
MODEL_PATH = os.path.join(project_root, "models", "MLP-2.h5")
SCALER_PATH = os.path.join(project_root, "models", "scaler.pkl")
DATA_PATH = os.path.join(project_root, "data", "processed", "X_train.csv")
WEB_FOLDER = os.path.join(api_dir, "web")
# --- FIN: Configuración de rutas ---

# Inicializar la app Flask
app = Flask(__name__, static_folder=WEB_FOLDER, static_url_path="/")
CORS(app)  # Habilita CORS para todos los orígenes

# Variables globales para modelo, scaler y características
model = None
scaler = None
EXPECTED_FEATURES = []

# --- Carga de recursos al iniciar la app ---
print("[INFO] Cargando modelo, scaler y características esperadas...")

try:
    # Cargar X_train.csv para obtener las características
    print(f"[INFO] Buscando X_train.csv en: {DATA_PATH}")
    if os.path.exists(DATA_PATH):
        df_temp = pd.read_csv(DATA_PATH)
        EXPECTED_FEATURES = df_temp.columns.tolist()
        print(
            f"[INFO] X_train.csv cargado. Número de características: {len(EXPECTED_FEATURES)}"
        )
    else:
        raise FileNotFoundError(f"Archivo no encontrado: {DATA_PATH}")

    # Cargar el modelo
    print(f"[INFO] Buscando modelo en: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)
        print(
            f"[INFO] Modelo cargado correctamente. Capas: {[layer.name for layer in model.layers]}"
        )
    else:
        raise FileNotFoundError(f"Archivo no encontrado: {MODEL_PATH}")

    # Cargar el scaler
    print(f"[INFO] Buscando scaler en: {SCALER_PATH}")
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("[INFO] Scaler cargado correctamente.")
    else:
        raise FileNotFoundError(f"Archivo no encontrado: {SCALER_PATH}")

    print("[INFO] Todos los recursos cargados correctamente.")

except Exception as e:
    print(f"[ERROR] No se pudieron cargar los recursos: {e}")
    model = None
    scaler = None
    EXPECTED_FEATURES = []

# --- ENDPOINTS DE LA API ---


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud: Verifica si la API está funcionando."""
    if model is not None and scaler is not None and EXPECTED_FEATURES:
        return jsonify({"status": "OK", "message": "Modelo listo para predicciones"})
    else:
        return jsonify(
            {
                "status": "ERROR",
                "message": "Modelo, scaler o características no cargados.",
            }
        ), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de predicción: Recibe datos del cliente y devuelve la probabilidad de churn."""
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

        # Asegurar que tenga todas las columnas esperadas y en el mismo orden
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

        # Definir predicción binaria
        churn_prediction = bool(churn_probability > 0.5)

        return jsonify(
            {
                "churn_probability": round(churn_probability, 4),
                "churn_prediction": churn_prediction,
            }
        )

    except Exception as e:
        print(f"[ERROR] Error durante la predicción: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    """Sirve la página web principal (index.html)."""
    return send_from_directory(WEB_FOLDER, "index.html")


# --- Inicio de la aplicación ---
if __name__ == "__main__":
    # Usa el puerto asignado por Railway o 5000 en local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
