import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
ruta="C:/4_F5/017_track_01/DATA-SCIENTIST-4/"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import procesado #importo la función de carga que está en el archivo data_load.py
#from catboost import CatBoostClassifier

def particion(file_path):
    # Llamar a la función load_data para cargar los datos
    df = procesado(file_path)
    
    if df is not None:
        print('\n----------Inicio de la PARTICION del dataset:--------\n')
        # Realizar un split de 80% para entrenamiento y 20% para prueba
        X = df.drop('stroke', axis=1)  # Variables independientes
        y = df['stroke']  # Variable dependiente
        """ 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Tamaño del conjunto de entrenamiento de X:", len(X_train))
        print("Tamaño del conjunto de validación de X:", len(X_test))
        print("Tamaño del conjunto de entrenamiento de Y:", len(y_train))
        print("Tamaño del conjunto de validación de Y:", len(y_test))
        print(X_train.head())
        
        #Guardando las particiones
        X_train.to_csv(ruta+'data/processed/X_train.csv')
        X_test.to_csv(ruta+'data/processed/X_test.csv')
        y_train.to_csv(ruta+'data/processed/y_train.csv')
        y_test.to_csv(ruta+'data/processed/y_test.csv')
        return X_train, X_test, y_train, y_test """
        print('Terminado la pariticion')
        print(X.head())
        return X, y
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        
        

"""
# Ejemplo de uso
if __name__ == "__main__":
    ruta="C:/4_F5/017_track_01/DATA-SCIENTIST-4/"
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"  # Reemplazar con la ruta correcta
    
    X, y= particion(file_path)
    X_train, X_test, y_train, y_test = particion(file_path)
    X_train.to_csv(ruta+'data/processed/X_train.csv')
    
    # Utilizar las particiones del dataset
    print("X_train:", X_train.head())
    print("X_test:", X_test.head())
    print("y_train:", y_train.head())
    print("y_test:", y_test.head()) """
    
    
def split_train_test(file_path):
    # Llamar a la función load_data para cargar los datos
    X, y= particion(file_path)
    
    
    if X is not None:
        print('\n----------Inicio deL Split del dataset:--------\n')


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Tamaño del conjunto de entrenamiento de X:", len(X_train))
        print("Tamaño del conjunto de validación de X:", len(X_test))
        print("Tamaño del conjunto de entrenamiento de Y:", len(y_train))
        print("Tamaño del conjunto de validación de Y:", len(y_test))
        print(X_train.head())
        
        #Guardando las particiones
        X_train.to_csv(ruta+'data/processed/X_train.csv')
        X_test.to_csv(ruta+'data/processed/X_test.csv')
        y_train.to_csv(ruta+'data/processed/y_train.csv')
        y_test.to_csv(ruta+'data/processed/y_test.csv')
        print('Terminado el split')
        return X_train, X_test, y_train, y_test
        

    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        
# Ejemplo de uso
if __name__ == "__main__":
    ruta="C:/4_F5/017_track_01/DATA-SCIENTIST-4/"
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"  # Reemplazar con la ruta correcta
    
    X, y= particion(file_path)
    X_train, X_test, y_train, y_test = split_train_test(file_path)