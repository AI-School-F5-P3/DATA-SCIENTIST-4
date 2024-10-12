import pandas as pd
import os
import sys

 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_load import load_data #importo la función de carga que está en el archivo data_load.py


#Voy a eliminar la columna 'ever_married' porque tiene una alta correlación con 'age'




def remove_redundant_variables(df):
    """
    Elimina variables redundantes del DataFrame.
    """
    # Aquí puedes añadir lógica para eliminar variables redundantes
    # Por ejemplo: df = df.drop(['columna_redundante1', 'columna_redundante2'], axis=1)
    print("Variables redundantes eliminadas.")
    return df

def categorize_age(df):
    """
    Categoriza la variable 'age' y elimina la columna original.
    """
    df['age_cat'] = pd.cut(df['age'], 
                           bins=[18, 25, 35, 45, 55, 65, 100], 
                           labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    df = df.drop('age', axis=1)
    print("Edad categorizada.")
    return df

def encode_categorical_variables(df):
    """
    Codifica las variables categóricas usando one-hot encoding.
    """
    categoricas = df.select_dtypes(include=['object']).columns.tolist()
    categoricas.append('age_cat')
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
        df = categorize_age(df)
        df = encode_categorical_variables(df)
        
        print("\nDataset procesado:")
        print(df.head())
        
        df.to_csv(ruta + 'data/processed/df.csv')
        print(df.columns)
        return df
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        return None

""" 
# Ejemplo de uso
if __name__ == "__main__":
    
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"  # Reemplazar con la ruta correcta
    procesado(file_path)
    
 """