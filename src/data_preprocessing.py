import pandas as pd
import os
import sys
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_load import load_data #importo la función de carga que está en el archivo data_load.py


#Voy a eliminar la columna 'ever_married' porque tiene una alta correlación con 'age'


def remove_id(df):
    """
    Elimina variables redundantes del DataFrame.
    """
    df = df.iloc[:, 1:]
    print(df.head())
    return df

def remove_redundant_variables(df):
    """
    Elimina variables redundantes del DataFrame.
    """
    
    df = df.drop('ever_married', axis=1)
    print("Edad categorizada.")
    print("Variables redundantes eliminadas.")
    return df

def categorize_age(df):
    """
    Categoriza la variable 'age' y elimina la columna original.
    """
    df['age_cat'] = pd.cut(df['age'], 
                           bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                           labels=['1-17','18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    df = df.drop('age', axis=1)
    print("Edad categorizada.")
    return df


# Función para categorizar el BMI
def categorize_bmi(df):
    """
    Categorización de BMI en bajo peso, normal, sobrepeso y obeso.
    """
    conditions = [
        (df['bmi'] < 18.5),  # Bajo peso
        (df['bmi'] >= 18.5) & (df['bmi'] < 24.9),  # Normal
        (df['bmi'] >= 25.0) & (df['bmi'] < 29.9),  # Sobrepeso
        (df['bmi'] >= 30.0)  # Obeso
    ]
    categories = ['Bajo', 'Normal', 'Sobrepeso', 'Obeso']
    
    # Crear una nueva columna 'bmi_category'
    df['bmi_cat'] = np.select(conditions, categories)
    
    df = df.drop('bmi', axis=1)
    print("BMI categorizado.")
    return df


# Función para categorizar el nivel de glucosa promedio
def categorize_glucose(df):
    """
    Categorización de 'avg_glucose_level' en rangos: normal, prediabético, diabético.
    """
    conditions = [
        (df['avg_glucose_level'] < 140),  # Normal
        (df['avg_glucose_level'] >= 140) & (df['avg_glucose_level'] < 200),  # Prediabético
        (df['avg_glucose_level'] >= 200)  # Diabético
    ]
    categories = ['Normal', 'Prediabético', 'Diabético']
    
    # Crear una nueva columna 'glucose_category'
    df['glucose_cat'] = np.select(conditions, categories)
    df = df.drop('avg_glucose_level', axis=1)
    print("BMI categorizado.")
    return df


def encode_categorical_variables(df):
    """
    Codifica las variables categóricas usando one-hot encoding.
    """
    
    categoricas = df.select_dtypes(include=['object']).columns.tolist()
    categoricas.append('age_cat')
    categoricas.append('bmi_cat')
    categoricas.append('glucose_cat')
    df = pd.get_dummies(df, columns=categoricas, drop_first=True, dtype=int)
    print("Variables categóricas codificadas.")
    return df

def procesado(file_path):
    """
    Función principal que coordina el procesamiento de los datos.
    """
    ruta = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/"
    df = load_data(file_path)

    if df is not None:
        print('\n****** Inicio del procesado del dataset: ******\n')
        
        print("Dataset original:")
        print(df.head())
        
        
        df = remove_redundant_variables(df)
        df = remove_id(df)
        df = categorize_age(df)
        df = categorize_bmi(df)
        df = categorize_glucose(df)
        df = encode_categorical_variables(df)
        
        print("\nDataset procesado:")
        print(df.head())
        
        df.to_csv(ruta + 'data/processed/df.csv',index=False)
        print(df.columns)
        
        return df
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        return None


# Ejemplo de uso
if __name__ == "__main__":
    
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"  # Reemplazar con la ruta correcta
    procesado(file_path)
    print(procesado(file_path))
