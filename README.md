# Predicción de eficiencia del hogar inteligente usando Machine Learning

Este proyecto desarrolla un sistema de **predicción de eficiencia en hogares inteligentes** utilizando datos de uso de dispositivos IoT y técnicas avanzadas de **machine learning supervisado**.

El objetivo es clasificar si un dispositivo doméstico inteligente opera de manera **eficiente o ineficiente**, a partir de métricas de consumo, uso, antigüedad y comportamiento del usuario.

---

## 🏠 Contexto del problema

Los hogares inteligentes generan grandes volúmenes de datos a partir de dispositivos como:
- luces inteligentes
- termostatos
- electrodomésticos conectados

Analizar estos datos permite:
- optimizar el consumo energético
- detectar ineficiencias tempranas
- mejorar la experiencia del usuario

Este proyecto aborda el problema como una **clasificación binaria** aplicada a datos de IoT.

---

## 🎯 Objetivo de Machine Learning

- **Tipo de problema:** Clasificación binaria  
- **Variable objetivo:** `SmartHomeEfficiency`  
  - `0` → ineficiente  
  - `1` → eficiente  
- **Meta:** maximizar F1-score y precisión en presencia de posible desbalance de clases

---

## 📊 Dataset

El conjunto de datos incluye métricas de uso de dispositivos domésticos inteligentes:

### Variables principales
- `DeviceType` – tipo de dispositivo
- `UsageHoursPerDay` – horas promedio de uso diario
- `EnergyConsumption` – consumo energético (kWh)
- `UserPreference` – preferencia del usuario (baja / alta)
- `MalfunctionIncidents` – número de fallos reportados
- `DeviceAgeMonths` – antigüedad del dispositivo
- `SmartHomeEfficiency` – estado de eficiencia (target)

> La columna `UserID` fue eliminada por no aportar valor predictivo.

---

## 🧪 Metodología

### 1. Exploración y limpieza de datos
- Análisis de tipos de datos y valores únicos
- Verificación de valores nulos y duplicados
- Estadísticas descriptivas

### 2. Preprocesamiento
- Codificación de variables categóricas (One-Hot Encoding)
- Escalado de características numéricas (`StandardScaler`)
- Detección de desbalance de clases
- Aplicación de **SMOTE** para sobremuestreo

### 3. Modelado y benchmarking
Se entrenaron y compararon múltiples clasificadores para identificar el mejor desempeño general.

---

## 🤖 Modelos evaluados

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Cada modelo fue evaluado utilizando:
- Accuracy
- F1-score (weighted)
- Matriz de confusión
- Classification report

---

## 🧠 Aprendizaje en conjunto (Ensembles)

A partir del ranking por **F1-score**, se seleccionaron los **3 mejores modelos** para construir:

- **Voting Classifier**
- **Stacking Classifier**

Ambos enfoques permitieron mejorar la estabilidad y el rendimiento del modelo final.

---

## 📈 Resultados

- Los modelos basados en **ensembles y boosting** mostraron el mejor desempeño
- El uso de SMOTE mejoró la detección de la clase minoritaria
- El stacking classifier presentó resultados consistentes en precisión y F1-score

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **pandas, numpy**
- **scikit-learn**
- **imbalanced-learn (SMOTE)**
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **matplotlib, seaborn**

---

## 📂 Estructura del repositorio

├── smart_home_device_usage_data.csv
├── Predicción de eficiencia del hogar inteligente.py
├── README.md


---

## 🚀 Próximos pasos

- Ajuste fino de hiperparámetros (Grid / Random Search)
- Feature importance y explainability (SHAP)
- Optimización de métricas orientadas a negocio
- Construcción de un pipeline completo con `sklearn.pipeline`
- Deploy del modelo como servicio para monitoreo en tiempo real

---

## 👤 Autor
**Flavia Hepp**  
Data Scientist en formación  
