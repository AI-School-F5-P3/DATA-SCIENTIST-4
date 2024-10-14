import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from split import particion #importo la función de carga que está en el archivo data_load.py
from split import split_train_test #importo la función de carga que está en el archivo data_load.py



def model_trainning(file_path):
    
    # Llamar a la función particion para cargar los datos y realizar el split
    X_train, X_test, y_train, y_test = split_train_test(file_path)
      
    if X_train is not None:
        print('\nIniciando el entrenamiento:\n\n')
        
        print("Tipos de datos en X_train:")
        print(X_train.dtypes)

        # Verificar que todas las columnas sean numéricas
        # Justo antes de aplicar SMOTE y entrenar el modelo
        non_numeric_columns = X_train.select_dtypes(exclude=['int64', 'float64', 'int32']).columns
        if len(non_numeric_columns) > 0:
            print(f"Advertencia: Las siguientes columnas no son numéricas: {non_numeric_columns}")
            print("Esto puede causar problemas con el modelo de Regresión Logística.")

        else:
            print("Todas las columnas son numéricas. Procediendo con el entrenamiento del modelo.")

        
        
        #Aplicar SMOTE para la clase minoritaria (oversampling)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Crear el modelo de Random Forest con hiperparámetros ajustados
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Número de árboles en el bosque
            max_depth=10,      # Profundidad máxima de los árboles (puedes ajustar según tus necesidades)
            random_state=42,   # Para reproducibilidad
            class_weight="balanced"  # Para manejar el desbalance de clases
        )        
        
        
        # Entrenar el modelo
        rf_model.fit(X_train_resampled, y_train_resampled)
        
        # Hacer predicciones sobre el conjunto de prueba
        y_pred=rf_model.predict(X_test)
        
        
        # Calcular métricas con el umbral ajustado
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        
        print('Fin del entrenamiento')
        print('Métricas:')
        print('Precisión:', accuracy)
        print('Precisión:', precision)
        print('Sensibilidad (Recall):', recall)
        print('Medida F1:', f1)
        print('Informe de clasificación:')
        print(report)
        print('Matriz de confusión:')
        print(matrix)
        
        print('Terminado el train')
        return matrix
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")


# Ejemplo de uso
if __name__ == "__main__":
    
    ruta="C:/4_F5/017_track_01/DATA-SCIENTIST-4/"
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"     
    matrix = model_trainning(file_path)
    print("Matriz de confusión final:")
    print(matrix)