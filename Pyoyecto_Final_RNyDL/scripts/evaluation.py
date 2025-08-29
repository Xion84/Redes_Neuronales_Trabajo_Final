# Evaluación
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
# Cargar datos procesados (para evaluación con modelo base)
def load_processed_data():
    """
    Carga los datos procesados desde la carpeta data/processed/.
    """
    X_train = pd.read_csv("../data/processed/X_train.csv").astype("float32").values
    X_test = pd.read_csv("../data/processed/X_test.csv").astype("float32").values
    y_train = (
        pd.read_csv("../data/processed/y_train.csv").values.ravel().astype("float32")
    )
    y_test = (
        pd.read_csv("../data/processed/y_test.csv").values.ravel().astype("float32")
    )

    print(
        f"[INFO] Datos procesados cargados. X_train: {X_train.shape}, X_test: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test


# Llamar a la función
X_train, X_test, y_train, y_test = load_processed_data()


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
# Comparación con modelo base: Regresión Logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

print("\n[INFO] Entrenando modelo base: Regresión Logística...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predicciones
y_pred_lr = lr_model.predict(X_test)
y_pred_lr_prob = lr_model.predict_proba(X_test)[:, 1]

# Métricas
acc_lr = accuracy_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(
    f"Logistic Regression - Accuracy: {acc_lr:.4f}, Recall: {rec_lr:.4f}, F1: {f1_lr:.4f}"
)

# Añadir a resultados
results_df = pd.concat(
    [
        results_df,
        pd.DataFrame(
            [
                {
                    "Modelo": "Logistic Regression",
                    "Accuracy": round(acc_lr, 4),
                    "Precision": round(precision_score(y_test, y_pred_lr), 4),
                    "Recall": round(rec_lr, 4),
                    "F1-Score": round(f1_lr, 4),
                    "ROC-AUC": round(roc_auc_score(y_test, y_pred_lr_prob), 4),
                }
            ]
        ),
    ],
    ignore_index=True,
)

# Guardar tabla actualizada
results_df.to_csv("../results/model_comparison.csv", index=False)
print("\n[INFO] Tabla de comparación con modelo base.")

# %%
# Gráfico de comparación de validación cruzada (usando arrays guardados)

print("\n[INFO] Generando gráfico de validación cruzada...")

try:
    # Verificar si los archivos existen
    if not os.path.exists("../results/scores_lr.npy") or not os.path.exists(
        "../results/scores_mlp2.npy"
    ):
        raise FileNotFoundError(
            "Archivos .npy no encontrados. Ejecute primero 'cross_validation.py'."
        )

    # Cargar los scores guardados
    scores_lr = np.load("../results/scores_lr.npy")
    scores_mlp2 = np.load("../results/scores_mlp2.npy")

    # Graficar
    plt.figure(figsize=(8, 5))
    plt.boxplot([scores_lr, scores_mlp2], labels=["Regresión Logística", "MLP-2"])
    plt.ylabel("F1-Score (5-Fold)")
    plt.title("Comparación de Generalización (Validación Cruzada)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/cv_comparison_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("[INFO] Gráfico de validación cruzada generado y guardado.")

except FileNotFoundError as e:
    print(f"[ERROR] {e}")
except Exception as e:
    print(f"[ERROR] No se pudo generar el gráfico: {e}")

# %%
