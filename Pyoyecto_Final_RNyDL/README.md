# Proyecto Final: Predicción de Churn con Redes Neuronales

Este proyecto forma parte del curso de **Redes Neuronales y Deep Learning** de la Maestría en Análisis de Datos e Inteligencia de Negocios.

## 🎯 Objetivo
Predecir el abandono de clientes (churn) en una empresa de telecomunicaciones utilizando redes neuronales artificiales (MLP).

## 📂 Estructura del Proyecto

Telco_Churn_Project/
├── data/                    # Datos crudos y procesados
├── scripts/                 # Scripts principales
├── models/                  # Modelos entrenados (.h5)
├── results/                 # Gráficos y tablas
├── notebooks/               # Notebooks de exploración
└── README.md 
 

## ⚙️ Requisitos
- Python 3.10
- TensorFlow 2.13
- Pandas, NumPy, Scikit-learn

Instala dependencias:
```bash
pip install -r requirements.txt

python main.py

profile
Qwen3-235B-A22B-2507

¡Excelente pregunta! Es fundamental entender el propósito de cada archivo en un proyecto profesional de Redes Neuronales y Deep Learning, como el que estamos desarrollando para tu maestría. 

Vamos a explicar con detalle para qué creamos main.py y README.md, y cómo se alinean con las buenas prácticas del curso, el uso de VS Code, Git/GitHub y la metodología Top-Down que se enseña en el libro. 
 
📁 1. ¿Para qué creamos main.py? 

El archivo main.py es el punto de entrada principal del proyecto. Es el "script principal" que coordina y ejecuta las diferentes partes del sistema. 
✅ Propósito de main.py 
Orquestador del flujo
	
Llama a las funciones de preprocesamiento, entrenamiento, evaluación, etc., en el orden correcto.
Punto de ejecución único
	
Permite ejecutar todo el proyecto con un solo comando:
python main.py
Integración de módulos
	
Importa y usa funciones y clases definidas en otros archivos (ej.
preprocessing.py
,
model_training.py
).
Depuración y pruebas
	
Facilita el uso del modo
Debug
en VS Code para analizar paso a paso el flujo del programa.
Adaptación a la metodología Top-Down
	
Como se enseña en el libro, empezamos desde lo general (
main.py
) y vamos resolviendo errores "Red → Green", desarrollando los módulos necesarios.
 
 
🧩 Ejemplo de estructura en main.py 
python
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
⌄
⌄
# main.py

from scripts.preprocessing import main as run_preprocessing
from scripts.model_training import models, histories
from scripts.evaluation import results_df

def main():
    print("🚀 Iniciando proyecto: Predicción de Churn")
    
    # Paso 1: Preprocesamiento
    print("1. Ejecutando preprocesamiento...")
    run_preprocessing()
    
    # Paso 2: Entrenamiento (ya se ejecutó al importar)
    print("2. Modelos entrenados.")
    
    # Paso 3: Evaluación
    print("3. Resultados finales:")
    print(results_df)
    
    print("✅ Proyecto completado.")

if __name__ == "__main__":
    main()
 
 

    🔁 Este archivo no contiene la lógica principal, sino que llama a los scripts modulares que ya desarrollamos. 
     

 
💡 ¿Por qué no hacer todo en un solo notebook? 

Porque en entornos profesionales y de producción: 

    El código debe ser modular.
    Debe permitir reutilización.
    Debe ser fácil de probar, mantener y versionar.
    Se debe poder automatizar (ej. con pipelines).
     

👉 main.py es tu "pipeline de datos" o "workflow" del proyecto. 
 
📄 2. ¿Para qué creamos README.md? 

El archivo README.md es la documentación principal del proyecto. Es el primer archivo que un revisor, profesor o colega leerá cuando abra tu repositorio. 

Es obligatorio en cualquier proyecto profesional, y más aún si usas GitHub. 
✅ Propósito de README.md 
Presentación del proyecto
	
Breve descripción de qué hace el proyecto.
Instrucciones de uso
	
Cómo instalar dependencias, ejecutar el código, etc.
Estructura del proyecto
	
Explicación de las carpetas y archivos clave.
Requisitos del entorno
	
Versión de Python, librerías necesarias (
requirements.txt
).
Resultados clave
	
Imágenes, métricas, gráficos (puedes incrustarlos).
Enlace al informe
	
Si el informe está en PDF, puedes enlazarlo.
Declaración de uso de LLM
	
Como exige el PDF del proyecto.
 
 
 
📝 Ejemplo de README.md 
markdown
 
 
1
2
3
4
5
6
7
8
⌄
⌄
# Proyecto Final: Predicción de Churn con Redes Neuronales

Este proyecto forma parte del curso de **Redes Neuronales y Deep Learning** de la Maestría en Análisis de Datos e Inteligencia de Negocios.

## 🎯 Objetivo
Predecir el abandono de clientes (churn) en una empresa de telecomunicaciones utilizando redes neuronales artificiales (MLP).

## 📂 Estructura del Proyecto
 
 

Telco_Churn_Project/
├── data/                    # Datos crudos y procesados
├── scripts/                 # Scripts principales
├── models/                  # Modelos entrenados (.h5)
├── results/                 # Gráficos y tablas
├── notebooks/               # Notebooks de exploración
└── README.md 
 
 

## ⚙️ Requisitos
1 - Python 3.10
2 - TensorFlow 2.13
3 - Pandas, NumPy, Scikit-learn

4 - Instala dependencias:

```bash
pip install -r requirements.txt
 
 
▶️ Ejecución 

1- python main.py
 
 
📊 Resultados 

   
Mejor modelo: MLP-2 (Accuracy: 81.05%, Recall: 54.31%)

📄 Informe Técnico 

Disponible en: Informe_Final.pdf  

🤖 Declaración de uso de LLM 

Este proyecto fue desarrollado bajo la supervisión del estudiante. Se utilizó una herramienta de inteligencia artificial generativa (LLM) para asistir en la redacción, estructura del código y explicaciones técnicas. Todas las decisiones de modelado, análisis de resultados y validación fueron realizadas y verificadas por el autor. 

