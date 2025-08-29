# %%
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import json
import os

print(f"[INFO] TensorFlow versión: {tf.__version__}")


# %%
# Cargar datos de prueba
def load_test_data():
    """
    Carga los datos de prueba desde la carpeta processed.
    """
    X_test = pd.read_csv("../data/processed/X_test.csv").astype("float32").values
    y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()
    print(
        f"[INFO] Datos de prueba cargados. X_test: {X_test.shape}, y_test: {y_test.shape}"
    )
    return X_test, y_test


X_test, y_test = load_test_data()


# %%
# Cargar modelos entrenados
def load_trained_models():
    """
    Carga los 5 modelos entrenados desde la carpeta models/.
    """
    model_names = ["MLP-1", "MLP-2", "MLP-3", "MLP-4", "MLP-5"]
    models = {}
    for name in model_names:
        try:
            model = load_model(f"../models/{name}.h5")
            models[name] = model
            print(f"[INFO] Modelo {name} cargado correctamente.")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {name}: {e}")
    return models


models = load_trained_models()


# %%
# Cargar historias de entrenamiento (opcional para análisis)
def load_history(model_name):
    """
    Carga la historia de entrenamiento desde JSON.
    """
    try:
        with open(f"../models/{model_name}_history.json", "r") as f:
            return json.load(f)
    except:
        return None


# %%
# Evaluación de modelos en TEST
results = []

for name, model in models.items():
    print(f"\n[INFO] Evaluando {name} en conjunto de prueba...")

    # Predicciones
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    # Métricas en TEST
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    # Guardar resultados
    results.append(
        {
            "Modelo": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        }
    )

    print(
        f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
    )

# Convertir a DataFrame
results_df = pd.DataFrame(results)
print("\n[INFO] Evaluación completada.")
print(results_df)

# Guardar tabla
results_df.to_csv("../results/model_comparison.csv", index=False)

# %%
