# Validación cruzada

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

print("[INFO] Iniciando Validación Cruzada (K-Fold)")

# %%
# Cargar datos procesados
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()

print(f"[INFO] Datos cargados. X_train: {X_train.shape}, X_test: {X_test.shape}")

# %%
# Cargar el scaler
scaler = joblib.load("../models/scaler.pkl")
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
X_train_scaled[numeric_features] = scaler.transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Convertir a numpy
X_train_final = X_train_scaled.astype("float32").values
y_train_final = y_train.astype("float32")

# %%
# Validación cruzada para Regresión Logística
print("\n[INFO] Validación cruzada: Regresión Logística")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_lr = cross_val_score(
    lr_model, X_train_final, y_train_final, cv=cv_lr, scoring="f1"
)
print(
    f"F1-CV (Logistic Regression): {scores_lr.mean():.4f} (+/- {scores_lr.std() * 2:.4f})"
)

# %%
# Validación cruzada para MLP-2
print("\n[INFO] Validación cruzada: MLP-2")
model_mlp2 = load_model("../models/MLP-2.h5")


# Función para convertir predicciones de Keras a formato binario
def keras_cross_val_score(model, X, y, cv):
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Entrenar en el fold (con pocos epochs para no sobreajustar)
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # Evaluar
        y_pred_prob = model.predict(X_val_fold, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).ravel()
        f1 = f1_score(y_val_fold, y_pred)
        scores.append(f1)

    return np.array(scores)


from sklearn.metrics import f1_score

cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_mlp2 = keras_cross_val_score(model_mlp2, X_train_final, y_train_final, cv_skf)

print(f"F1-CV (MLP-2): {scores_mlp2.mean():.4f} (+/- {scores_mlp2.std() * 2:.4f})")

# Guardar los arrays para usarlos después en evaluation.py
np.save("../results/scores_lr.npy", scores_lr)
np.save("../results/scores_mlp2.npy", scores_mlp2)

print("[INFO] Scores de validación cruzada guardados como .npy")
# %%
# Guardar resultados
cv_results = pd.DataFrame(
    {
        "Modelo": ["Logistic Regression", "MLP-2"],
        "F1-CV Mean": [scores_lr.mean(), scores_mlp2.mean()],
        "F1-CV Std": [scores_lr.std(), scores_mlp2.std()],
    }
)

cv_results.to_csv("../results/cross_validation_results.csv", index=False)
print(
    "\n[INFO] Resultados de validación cruzada guardados en '../results/cross_validation_results.csv'"
)

# %%
# Mostrar tabla
print("\n Resultados de Validación Cruzada (5-Fold)")
print(cv_results.to_string(index=False))
