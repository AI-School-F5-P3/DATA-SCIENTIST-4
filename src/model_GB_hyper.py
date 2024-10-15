import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from split import split_train_test


def load_and_split_data(file_path):
    """ Carga los datos y los divide en conjunto de entrenamiento y prueba. """
    return split_train_test(file_path)


def apply_smote(X_train, y_train):
    """ Aplica SMOTE para el oversampling de la clase minoritaria. """
    print('Aplicar SMOTE para la clase minoritaria (oversampling)')
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_resampled = X_train_resampled.values
    y_train_resampled = y_train_resampled.values
    
    return X_train_resampled, y_train_resampled


def train_xgboost_model(X_train_resampled, y_train_resampled):
    """ Entrena el modelo XGBoost. """
    print('Crear el modelo XGBoost')
    
    dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
    
    params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
    }
    num_round = 100
    xgb_model = xgb.train(params, dtrain, num_round)
    

    #xgb_model.fit(X_train_resampled, y_train_resampled)
    return xgb_model


def evaluate_model(xgb_model, X_test, y_test):
    """ Evalúa el modelo en el conjunto de prueba. """
    """ Evalúa el modelo en el conjunto de prueba. """
    # Convertir X_test a DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Hacer predicciones
    y_pred_proba = xgb_model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)  # Convertir probabilidades a clases

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print('Métricas:')
    print('Exactitud:', accuracy)
    print('Precisión:', precision)
    print('Sensibilidad (Recall):', recall)
    print('Medida F1:', f1)
    print('Informe de clasificación:')
    print(report)
    print('Matriz de confusión:')
    print(matrix)
    
    return matrix

def model_trainning(file_path):
    """ Función principal que coordina el flujo de entrenamiento del modelo. """
    X_train, X_test, y_train, y_test = load_and_split_data(file_path)

    
    if X_train is not None:
        print('\nIniciando el entrenamiento:\n\n')
        print("Tipos de datos en X_train:")
        print(X_train.dtypes)
        print(X_train.isnull().sum())

        # Verificar que todas las columnas sean numéricas
        non_numeric_columns = X_train.select_dtypes(exclude=['int64', 'float64', 'int32','float32']).columns
        if len(non_numeric_columns) > 0:
            print(f"Advertencia: Las siguientes columnas no son numéricas: {non_numeric_columns}")
            print("Esto puede causar problemas con el modelo XGBoost.")
            return None
        
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
        xgb_model = train_xgboost_model(X_train_resampled, y_train_resampled)
        # Convertir X_test y y_test a numpy arrays
        X_test = X_test.values
        y_test = y_test.values
        
        
        matrix = evaluate_model(xgb_model, X_test, y_test)
        
        return matrix
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        return None


# Ejemplo de uso
if __name__ == "__main__":
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"
    matrix = model_trainning(file_path)
    print("Matriz de confusión final:")