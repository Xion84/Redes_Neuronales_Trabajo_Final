# Proyecto Final: Predicción de Churn con Redes Neuronales Artificiales y Deep Learning

> **Maestría en Inteligencia de Negocios y Análisis de Datos**  
> Asignatura: Redes Neuronales y Deep Learning  
> Fecha de entrega: 16 de agosto de 2025  
> Autores: Hubert Gutiérrez, Danilo Matus, Enllely Roque  
> Profesor: Dr. Vladimir Gutiérrez  

🔗 **Repositorio GitHub**: [https://github.com/Xion84/Redes_Neuronales_Trabajo_Final](https://github.com/Xion84/Redes_Neuronales_Trabajo_Final)

---

## 🎯 Objetivo del Proyecto

Desarrollar un modelo de **Red Neuronal Artificial (ANN)** para predecir el abandono de clientes (**churn**) en una empresa de telecomunicaciones, utilizando el conjunto de datos **Telco Customer Churn**. El proyecto incluye:

- Preprocesamiento de datos.
- Entrenamiento de múltiples arquitecturas de redes densas (MLP).
- Evaluación en conjunto de prueba.
- Validación cruzada (K-Fold).
- Comparación con modelo base (Regresión Logística).
- Puesta en producción simulada mediante una API.

Este proyecto cumple con todos los requisitos del instructivo del curso y sirve como base para un **paper académico** futuro, bajo la asesoría del Dr. Vladimir Gutiérrez.

---

## 📂 Estructura del Proyecto

    Pyoyecto_Final_RNyDL/
    │
    ├── data/
    │   ├── raw/
    │   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
    │   └── processed/
    │       ├── X_train.csv, X_val.csv, X_test.csv
    │       └── y_train.csv, y_val.csv, y_test.csv
    │
    ├── notebooks/
    │   └── 01-drvlado-churn-ann.ipynb
    │
    ├── scripts/
    │   ├── preprocessing.py
    │   ├── model_training.py
    │   ├── evaluation.py
    │   └── cross_validation.py 
    │
    ├── models/
    │   ├── MLP-1.h5, MLP-2.h5, ...
    │   └── MLP-1_history.json, ...
    │
    ├── results/
    │   ├── model_comparison.csv
    │   ├── roc_curves.png
    │   ├── confusion_matrices.png
    │   ├── scatter_tenure_vs_monthly.png
    │   ├── descriptive_statistics.csv
    │   ├── cv_comparison_boxplot.png
    │   └── cross_validation_results.csv
    │
    ├── api/
    │   ├── app.py
    │   ├── wsgi.py
    │   ├── requirements.txt
    │   └── web/
    │       ├── index.html
    │       └── style.css
    │
    ├── .vscode/
    │   └── settings.json
    │
    ├── .gitignore
    ├── requirements.txt
    ├── README.md



---

## ⚙️ Requisitos del Entorno

  - Python 3.11.9
  - TensorFlow 2.16.1
  - Pandas, NumPy, Scikit-learn, Flask, Flask-Cors

### Instalación de dependencias

    ```bash
    # Crear entorno (recomendado con Conda)
    conda create -n telco_churn python=3.11
    conda activate telco_churn

    # Instalar dependencias
    pip install -r requirements.txt

▶️ Ejecución del Proyecto 

    Este proyecto sigue la metodología Top-Down + Baby Steps enseñada por el Dr. Gutiérrez, con celdas #%% en VS Code para desarrollo interactivo. 
    1. Preprocesamiento de datos 
    python scripts/preprocessing.py
    2. Entrenamiento de modelos
    python scripts/model_training.py
    3. Evaluación de modelos
    python scripts/evaluation.py
    4. Validación cruzada
    python scripts/cross_validation.py

Resultados claves 

    Modelo                          Accuracy        Recall          F1-Score        ROC-AUC
    MLP-1                           0.7923          0.4823          0.5541          0.8012
    MLP-2                          	0.8105          0.5431 	        0.6042          0.8267
    MLP-3                           0.8018          0.5102          0.5753          0.8154
    MLP-4                           0.7789          0.4210          0.5012          0.7821
    MLP-5                           0.7856          0.4532          0.5267          0.7903
    Logistic Regression (base)      0.7982          0.4912          0.5763          0.8045

    ✅ Mejor modelo: MLP-2 (2 capas ocultas, dropout 0.3, Adam, ReLU)
    🎯 F1-Score: 0.6042 — superior al modelo base

Validación Cruzada (5-Fold)

    Modelo                          F1-CV MEAN          F1-CV STD
    Regresión Logística             0.5813              0.0330
    MLP-2                           0.6151              0.0369

🌐 API de Producción (Simulada) 

Una API simple con Flask permite exponer el modelo MLP-2 para predicciones en entornos de producción. 
    Ejecutar la API 

    cd api
    python app.py

    Endpoint
    * POST /predict → Recibe datos del cliente y devuelve probabilidad de churn.

    Ejemplo de solicitud (usando curl)

    curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 2148.0
    }'

🌐 API en la Nube (Render) 

    La API está desplegada en Render y es accesible desde cualquier navegador: 

    URL: https://churn-prediction-api-1cqs.onrender.com 

    Endpoints 

    GET /health → Verifica estado del modelo.
    POST /predict → Realiza predicción de churn.
    GET / → Página web interactiva.
     

    ✅ Esta demostración confirma que el modelo puede ser usado en producción real. 
     
🧠 Metodología Aplicada 

    Top-Down + Baby Steps: Diseño incremental desde el main.py hasta la implementación de módulos.
    VS Code como IDE principal, con celdas #%% para ejecución interactiva.
    Control de versiones con Git/GitHub, usando ramas main y development.
    Preprocesamiento: One-Hot Encoding, StandardScaler, división estratificada 70/15/15.
    Modelos: 5 arquitecturas de MLP con variación de hiperparámetros.
    Evaluación: Métricas en conjunto de prueba (al menos 5 estadísticos).
    Comparación con modelo base: Regresión Logística.
    Validación cruzada: 5-Fold para evaluar generalización.
     
🤖 Declaración de Uso de LLM 

    Este proyecto fue desarrollado bajo la supervisión del estudiante. Se utilizó una herramienta de inteligencia artificial generativa (LLM) para asistir en la redacción del informe, diseño de la estructura del código, explicaciones técnicas y generación de ejemplos. Todas las decisiones de modelado, análisis de resultados, entrenamiento y validación fueron realizadas y verificadas por el autor. La herramienta no generó resultados directos sin supervisión ni ejecutó código por sí sola. 
     
📚 Bibliografía 

    Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning.
    Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
    Kaggle. (2018). Telco Customer Churn Dataset. https://www.kaggle.com/blastchar/telco-customer-churn 
    Dr. Vladimir Gutiérrez. (2025). Redes Neuronales Artificiales y Aprendizaje Profundo (Cap. 01 - Cap. 08-2).
     