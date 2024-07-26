"""
Este conjunto de datos captura métricas de uso de dispositivos domésticos inteligentes y ofrece información sobre el comportamiento del usuario, 
la eficiencia del dispositivo y las preferencias. Incluye datos sobre tipos de dispositivos, patrones de uso, consumo de energía, incidentes de 
mal funcionamiento y métricas de satisfacción del usuario.

Características:
UserID: Identificador único de cada usuario.
DeviceType: tipo de dispositivo doméstico inteligente (p. ej., luces, termostato).
UsageHoursPerDay: Promedio de horas por día que se utiliza el dispositivo.
EnergyConsumption: Consumo energético diario del dispositivo (kWh).
Preferencias de usuario: preferencia del usuario para el uso del dispositivo (0 - Bajo, 1 - Alto).
Incidentes de mal funcionamiento: número de incidentes de mal funcionamiento reportados.
DeviceAgeMonths: Antigüedad del dispositivo en meses.
SmartHomeEfficiency (variable objetivo): estado de eficiencia del dispositivo doméstico inteligente (0: ineficiente, 1: eficiente)."""

import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, StackingClassifier


df = pd.read_csv('smart_home_device_usage_data.csv')

#Explorando el marco de datos
print(df.head())

def get_df_info(df):
    print("\n\033[1mForma del marco de datos:\033[0m ", df.shape)
    print("\n\033[1mColumnas en el marco de datose:\033[0m ", df.columns.to_list())
    print("\n\033[1mTipos de datos de columnas:\033[0m\n", df.dtypes)
    
    print("\n\033[1mInformación sobre el marco de datos:\033[0m")
    df.info()
    
    print("\n\033[1mNúmero de valores únicos en cada columna:\033[0m")
    for col in df.columns:
        print(f"\033[1m{col}\033[0m: {df[col].nunique()}")
        
    print("\n\033[1mNúmero de valores nulos en cada columna:\033[0m\n", df.isnull().sum())
    
    print("\n\033[1mNúmero de filas duplicadas:\033[0m ", df.duplicated().sum())
    
    print("\n\033[1mEstadísticas descriptivas de DataFramee:\033[0m\n", df.describe().transpose())

# Llamar a la función
get_df_info(df)

#Preprocesamiento de datos
#Eliminar la columna ID de usuario:

df = df.drop('UserID', axis = 1)
# Divida el marco de datos en características (X) y destino (y)
X = df.drop('SmartHomeEfficiency', axis=1)
y = df['SmartHomeEfficiency']
# Manejo de variables categóricas en X

X = pd.get_dummies(X)

"""
La función apply_models toma características (X) y etiquetas de destino (y) como entrada y realiza las siguientes tareas:

Preprocesamiento de datos:

Divide los datos en conjuntos de entrenamiento y prueba.
Comprueba el desequilibrio de clases y aplica SMOTE (sobremuestreo) si es necesario.
Escala las funciones usando StandardScaler.
Capacitación y evaluación del modelo:

Define un conjunto de modelos de clasificación de aprendizaje automático.
Entrena cada modelo con los datos de entrenamiento.
Evalúa cada modelo según los datos de prueba utilizando precisión y puntuación F1.
Imprime informes detallados (precisión, matriz de confusión, informe de clasificación) para cada modelo.
Aprendizaje conjunto:

Identifica los 3 modelos con mejor rendimiento según la puntuación F1.
Crea dos modelos de conjunto (Clasificador de votación y Clasificador de apilamiento) utilizando los 3 modelos principales.
Evalúa los modelos de conjunto sobre los datos de prueba utilizando precisión, matriz de confusión e informe de clasificación.
En resumen, esta función está diseñada para explorar varios modelos de clasificación, identificar los de mejor rendimiento y potencialmente 
mejorar interpretación a través de técnicas de aprendizaje en conjunto."""

def apply_models(X, y):
    # Divida los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Comprobar desequilibrio de clases
    class_counts = np.bincount(y_train)
    if len(class_counts) > 2 or np.min(class_counts) / np.max(class_counts) < 0.1:
      print("Desequilibrio de clases detectado. Aplicando SMOTE...")
    
    # Aplicar SMOTE (desequilibrio de clases)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Inicializar el StandardScaler
    scaler = StandardScaler()

    # Ajuste el escalador a los datos de entrenamiento y transforme tanto los datos de entrenamiento como los de prueba
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definir los modelos
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GradientBoost': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    # Inicializar un diccionario para contener el rendimiento de cada modelo
    model_performance = {}

    # Aplicar cada modelo
    for model_name, model in models.items():
        print(f"\n\033[1mClasificación con {model_name}:\033[0m\n{'-' * 30}")
        
        # Ajustar el modelo a los datos de entrenamiento
        model.fit(X_train, y_train)

        # Hacer predicciones sobre los datos de prueba
        y_pred = model.predict(X_test)

        # Calcule la precisión y la puntuación f1
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Almacenar la interpretación en el diccionario
        model_performance[model_name] = (accuracy, f1)

        # Imprimir la puntuación de precisión
        print("\033[1m**Exactitud**:\033[0m\n", accuracy)

        # Imprime la matriz de confusión
        print("\n\033[1m**Matriz de confusión**:\033[0m\n", confusion_matrix(y_test, y_pred))

        # Imprimir el informe de clasificación
        print("\n\033[1m**Informe de clasificación**:\033[0m\n", classification_report(y_test, y_pred))

    # Ordene los modelos según la puntuación f1 y elija los 3 primeros
    top_3_models = sorted(model_performance.items(), key=lambda x: x[1][1], reverse=True)[:3]
    print("\n\033[1m Los 3 mejores modelos según la puntuación F1:\033[0m\n", top_3_models)

    # Extraiga los nombres de los modelos y los clasificadores de los 3 modelos principales
    top_3_model_names = [model[0] for model in top_3_models]
    top_3_classifiers = [models[model_name] for model_name in top_3_model_names]

    # Crea un Clasificador de Votación con los 3 mejores modelos
    print("\n\033[1mInicializando el clasificador de votación con los 3 mejores modelos...\033[0m\n")
    voting_clf = VotingClassifier(estimators=list(zip(top_3_model_names, top_3_classifiers)), voting='hard')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print("\n\033[1m**Evaluación del clasificador de votación**:\033[0m\n")
    print("\033[1m**Exactitud**:\033[0m\n", accuracy_score(y_test, y_pred))
    print("\n\033[1m**Matriz de confusión**:\033[0m\n", confusion_matrix(y_test, y_pred))
    print("\n\033[1m**Informe de clasificación**:\033[0m\n", classification_report(y_test, y_pred))

    # Crea un clasificador de apilamiento con los 3 mejores modelos
    print("\n\033[1mInicializando el clasificador de apilamiento con los 3 mejores modelos...\033[0m\n")
    stacking_clf = StackingClassifier(estimators=list(zip(top_3_model_names, top_3_classifiers)))
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    print("\n\033[1m**Evaluación del clasificador de apilamiento**:\033[0m\n")
    print("\033[1m**Exactitud**:\033[0m\n", accuracy_score(y_test, y_pred))
    print("\n\033[1m**Matriz de confusión**:\033[0m\n", confusion_matrix(y_test, y_pred))
    print("\n\033[1m**Informe de clasificación**:\033[0m\n", classification_report(y_test, y_pred))
# Aplicar la función en X e y
apply_models(X, y)

