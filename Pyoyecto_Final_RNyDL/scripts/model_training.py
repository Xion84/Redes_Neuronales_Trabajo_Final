# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Habilitar el uso de memoria dinámica en GPU (si está disponible)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"[INFO] TensorFlow versión: {tf.__version__}")
print(f"[INFO] GPUs disponibles: {len(gpus)}")


# %%
# Cargar datos procesados
def load_data():
    """
    Carga los datos preprocesados desde la carpeta processed.
    Asegura que sean arreglos NumPy de tipo float32.
    """
    X_train = pd.read_csv("../data/processed/X_train.csv")
    X_val = pd.read_csv("../data/processed/X_val.csv")
    X_test = pd.read_csv("../data/processed/X_test.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()
    y_val = pd.read_csv("../data/processed/y_val.csv").values.ravel()
    y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()

    # Convertir a float32 y a numpy arrays
    X_train = X_train.astype("float32").values
    X_val = X_val.astype("float32").values
    X_test = X_test.astype("float32").values
    y_train = y_train.astype("float32")
    y_val = y_val.astype("float32")
    y_test = y_test.astype("float32")

    print(
        f"[INFO] Datos cargados y convertidos a float32. X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---
X_train, X_val, X_test, y_train, y_val, y_test = load_data()


# %%
# Callbacks comunes
def get_callbacks():
    """
    Define callbacks para todos los modelos.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
    )
    return [early_stopping, reduce_lr]


# %%
# Definición de modelos
def create_model_1(input_dim):
    """MLP-1: 1 capa oculta (64 neuronas), ReLU, Adam, sin dropout"""
    model = Sequential(
        [
            Dense(64, activation="relu", input_dim=input_dim),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_model_2(input_dim):
    """MLP-2: 2 capas ocultas (128 → 64), ReLU, Adam, dropout=0.3"""
    model = Sequential(
        [
            Dense(128, activation="relu", input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_model_3(input_dim):
    """MLP-3: 3 capas ocultas (256 → 128 → 64), ReLU, Adam, dropout=0.5, L2"""
    model = Sequential(
        [
            Dense(256, activation="relu", kernel_regularizer="l2", input_dim=input_dim),
            Dropout(0.5),
            Dense(128, activation="relu", kernel_regularizer="l2"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_model_4(input_dim):
    """MLP-4: 2 capas (64 → 32), Tanh, SGD con momentum"""
    model = Sequential(
        [
            Dense(64, activation="tanh", input_dim=input_dim),
            Dense(32, activation="tanh"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_model_5(input_dim):
    """MLP-5: 1 capa (32), ReLU, RMSprop"""
    model = Sequential(
        [
            Dense(32, activation="relu", input_dim=input_dim),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# %%
# Entrenamiento de modelos
def train_model(
    model, model_name, X_train, y_train, X_val, y_val, epochs=100, batch_size=32
):
    """
    Entrena un modelo y guarda historia, modelo y gráficos.
    Convierte el historial a tipos JSON serializables.
    """
    print(f"\n[INFO] Entrenando {model_name}...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Guardar modelo
    os.makedirs("../models", exist_ok=True)
    model.save(f"../models/{model_name}.h5")
    print(f"[INFO] Modelo {model_name} guardado.")

    # --- CORRECCIÓN: Convertir history.history a tipos nativos ---
    history_dict = {}
    for key, value in history.history.items():
        # Convertir cada valor de numpy array a lista de floats nativos
        history_dict[key] = [float(val) for val in value]

    # Guardar historia como JSON
    with open(f"../models/{model_name}_history.json", "w") as f:
        json.dump(history_dict, f, indent=4)

    print(f"[INFO] Historia de {model_name} guardada como JSON.")

    # Graficar curvas de aprendizaje
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Val Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict["accuracy"], label="Train Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Val Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"../results/{model_name}_learning_curves.png")
    plt.show()

    return history


# %%
# Ejecución de entrenamiento
input_dim = X_train.shape[1]

# Crear carpeta de resultados
os.makedirs("../results", exist_ok=True)

# Diccionario de modelos
models = {
    "MLP-1": create_model_1(input_dim),
    "MLP-2": create_model_2(input_dim),
    "MLP-3": create_model_3(input_dim),
    "MLP-4": create_model_4(input_dim),
    "MLP-5": create_model_5(input_dim),
}

# Entrenar todos los modelos
histories = {}
for name, model in models.items():
    history = train_model(model, name, X_train, y_train, X_val, y_val)
    histories[name] = history

print("\n[INFO] Todos los modelos han sido entrenados y guardados.")
# %%
