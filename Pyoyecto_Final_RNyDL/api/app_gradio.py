# App en gradio

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Cargar modelo y scaler
MODEL_PATH = "models/MLP-2.h5"
SCALER_PATH = "models/scaler.pkl"

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modelo y scaler cargados correctamente.")
except Exception as e:
    print(f"Error al cargar: {e}")
    model = None
    scaler = None

# Características esperadas
EXPECTED_FEATURES = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    if model is None or scaler is None:
        return "Error: Modelo no disponible", 0.0

    # Crear DataFrame
    data = {
        "gender": [gender],
        "SeniorCitizen": [int(senior_citizen)],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [int(tenure)],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [float(monthly_charges)],
        "TotalCharges": [float(total_charges)],
    }

    df = pd.DataFrame(data)

    # One-Hot Encoding
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

    # Asegurar columnas
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[EXPECTED_FEATURES]

    # Preprocesamiento
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_features] = scaler.transform(df[numeric_features])

    # Predicción
    X = df.astype("float32").values
    prob = float(model.predict(X, verbose=0)[0][0])
    prediction = "🔴 ALTO RIESGO" if prob > 0.5 else "🟢 BAJO RIESGO"

    return prediction, round(prob * 100, 2)


# Interfaz Gradio
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Género"),
        gr.Checkbox(label="¿Es adulto mayor?"),
        gr.Dropdown(["Yes", "No"], label="¿Tiene pareja?"),
        gr.Dropdown(["Yes", "No"], label="¿Tiene dependientes?"),
        gr.Slider(0, 72, step=1, label="Meses como cliente (tenure)"),
        gr.Dropdown(["Yes", "No"], label="¿Tiene servicio telefónico?"),
        gr.Dropdown(
            ["No phone service", "No", "Yes"], label="¿Tiene múltiples líneas?"
        ),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Servicio de Internet"),
        gr.Dropdown(
            ["No internet service", "No", "Yes"], label="¿Tiene seguridad en línea?"
        ),
        gr.Dropdown(
            ["No internet service", "No", "Yes"], label="¿Tiene respaldo en línea?"
        ),
        gr.Dropdown(
            ["No internet service", "No", "Yes"],
            label="¿Tiene protección de dispositivo?",
        ),
        gr.Dropdown(
            ["No internet service", "No", "Yes"], label="¿Tiene soporte técnico?"
        ),
        gr.Dropdown(
            ["No internet service", "No", "Yes"], label="¿Ve TV por streaming?"
        ),
        gr.Dropdown(
            ["No internet service", "No", "Yes"], label="¿Ve películas por streaming?"
        ),
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"], label="Tipo de contrato"
        ),
        gr.Dropdown(["Yes", "No"], label="¿Facturación electrónica?"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Método de pago",
        ),
        gr.Number(label="Cargos mensuales ($)"),
        gr.Number(label="Cargos totales ($)"),
    ],
    outputs=[
        gr.Textbox(label="Predicción"),
        gr.Number(label="Probabilidad de churn (%)"),
    ],
    title="🔮 Predicción de Churn - Telco",
    description="Ingrese los datos del cliente para predecir si está en riesgo de abandono.",
)

# Ejecutar localmente
if __name__ == "__main__":
    iface.launch()
