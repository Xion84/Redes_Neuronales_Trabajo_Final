# Proyecto Final: Predicción de Churn con Redes Neuronales Artificiales y Deep Learning

> **Maestría en Análisis de Datos e Inteligencia de Negocios**  
> Asignatura: Redes Neuronales y Deep Learning  
> Fecha de entrega: 16 de agosto de 2025  
> Autores: [Hubert Gutiérrez, Danilo Matus, Enllely Roque]  
> Profesor: Dr. Vladimir Gutiérrez  

---

## 🎯 Objetivo del Proyecto

    Desarrollar un modelo de **Red Neuronal Artificial (ANN)** para predecir el abandono de clientes (**churn**) en una empresa de telecomunicaciones, utilizando el conjunto de datos **Telco Customer Churn**. El proyecto incluye preprocesamiento, entrenamiento de múltiples arquitecturas de redes densas (MLP), evaluación en conjunto de prueba y comparación con un modelo base (regresión logística).

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
    │   └── evaluation.py
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
    │   └── descriptive_statistics.csv
    │
    ├── .vscode/
    │   └── settings.json
    │
    ├── .gitignore
    ├── requirements.txt
    ├── README.md
 

## ⚙️ Requisitos
    1 - Python 3.10
    2 - TensorFlow 2.13
    3 - Pandas, NumPy, Scikit-learn

    4- Instala dependencias:
    ```bash
    pip install -r requirements.txt

    python main.py
    
▶️ Ejecución del Proyecto 

    Este proyecto sigue la metodología Top-Down y baby steps enseñada por el Dr. Gutiérrez, con celdas #%% en VS Code para desarrollo interactivo. 

    1. Preprocesamiento de datos:
    python scripts/preprocessing.py

    2. Entrenamiento de modelos:
    python scripts/model_training.py

    3. Evaluación de modelos:
    python scripts/evaluation.py
    
    "Todos los resultados se guardan automáticamente en las carpetas models/ y results/."

--- 
 
 
📊 Resultados 
    
   
    ✅ Mejor modelo: MLP-2 (2 capas ocultas, dropout 0.3, Adam, ReLU)

    🎯 F1-Score: 0.6042 — superior al modelo base

🧠 Metodología Aplicada 

    Top-Down + Baby Steps: Diseño incremental desde el main.py hasta la implementación de módulos.
    VS Code como IDE principal, con celdas #%% para ejecución interactiva.
    Control de versiones con Git/GitHub, usando ramas main y development.
    Preprocesamiento: One-Hot Encoding, StandardScaler, división estratificada 70/15/15.
    Modelos: 5 arquitecturas de MLP con variación de hiperparámetros.
    Evaluación: Métricas en conjunto de prueba (al menos 5 estadísticos).
    Comparación con modelo base: Regresión Logística.
     

🤖 Declaración de Uso de LLM 

    Este proyecto fue desarrollado bajo la supervisión del estudiante. Se utilizó una herramienta de inteligencia artificial generativa (LLM) para asistir en la redacción del informe, diseño de la estructura del código, explicaciones técnicas y generación de ejemplos. Todas las decisiones de modelado, análisis de resultados, entrenamiento y validación fueron realizadas y verificadas por el autor. La herramienta no generó resultados directos sin supervisión ni ejecutó código por sí sola. 
     
📚 Bibliografía 

    Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning.
    Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
    Kaggle. (2018). Telco Customer Churn Dataset. https://www.kaggle.com/blastchar/telco-customer-churn 
    Dr. Vladimir Gutiérrez. (2025). Redes Neuronales Artificiales y Aprendizaje Profundo (Cap. 01 - Cap. 08-2).
     
