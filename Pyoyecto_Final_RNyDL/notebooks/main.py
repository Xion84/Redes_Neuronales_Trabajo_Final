# notebooks/main.py

# %%
# Autores: Hubert Gutiérrez, Danilo Matus, Enllely Roque
# Proyecto: Predicción de Churn con Redes Neuronales
# Maestría en Análisis de Datos e Inteligencia de Negocios

# %%
# Agregar la ruta de los módulos personalizados (libs)
import sys

sys.path.append("../libs")
sys.path.append("../scripts")

# %%
# Cargar librerías
import pandas as pd
import numpy as np
import tensorflow as tf

print(f"[INFO] Python y librerías cargadas. TensorFlow: {tf.__version__}")

# %%
# Ejecutar preprocesamiento
from scripts.preprocessing import main as run_preprocessing

print("Ejecutando preprocesamiento...")
run_preprocessing()

# %%
# Cargar datos procesados para verificar
X_train = pd.read_csv("../data/processed/X_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
print(f"Datos de entrenamiento cargados: {X_train.shape}")

# %%
# Entrenar modelos
from scripts.model_training import models, histories

print("Modelos ya entrenados y disponibles.")

# %%
# Evaluar modelos
from scripts.evaluation import results_df

print("Resultados de evaluación:")
print(results_df)

# %%
# Mostrar mejor modelo
best_model = results_df.loc[results_df["F1-Score"].idxmax()]["Modelo"]
print(f"\nMejor modelo según F1-Score: {best_model}")

# %%
# Fin del script
print("Ejecución del main completada. Puede explorar modelos, datos o gráficos.")
