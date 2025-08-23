import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Asumiendo que ya ejecutaste la Parte 1 ---
# X_train_processed, X_test_processed, y_train, y_test están disponibles en tu entorno.

# --- Construcción del Modelo ---
# Definir la arquitectura de la red neuronal
model = Sequential()

# Capa de entrada: El número de neuronas debe coincidir con el número de características de entrada
model.add(Dense(32, activation="relu", input_shape=(X_train_processed.shape[1],)))

# Capa oculta con Dropout para regularización (previene el sobreajuste)
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))  # Dropout del 20%

# Capa de salida: 1 neurona con activación 'sigmoid' para clasificación binaria
model.add(Dense(1, activation="sigmoid"))

# --- Compilación del Modelo ---
# Configurar el optimizador, la función de pérdida y las métricas
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Mostrar un resumen de la arquitectura del modelo
model.summary()

# --- Entrenamiento del Modelo ---
# Entrenar el modelo con los datos de entrenamiento y validarlo con los de prueba
# El historial de entrenamiento se guarda para poder graficarlo después
history = model.fit(
    X_train_processed,
    y_train,
    epochs=50,  # Número de veces que el modelo verá todo el dataset
    batch_size=32,  # Número de muestras por actualización del gradiente
    validation_data=(X_test_processed, y_test),
    verbose=1,
)  # Muestra una barra de progreso

print("\n¡Entrenamiento completado!")
