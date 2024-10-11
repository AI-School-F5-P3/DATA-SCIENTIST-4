import pandas as pd
import os
import sys

 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_load import load_data #importo la función de carga que está en el archivo data_load.py


#Voy a eliminar la columna 'ever_married' porque tiene una alta correlación con 'age'






def procesado(file_path):

    # Llamar a la función load_data para cargar los datos
    df = load_data(file_path)
    ruta="C:/4_F5/017_track_01/DATA-SCIENTIST-4/"

    if df is not None:
        print('\n****** Inicio del procesado del dataset: ******\n')
        
        
        #Hago las categorías de 'Edad'
        #df['age_cat'] = pd.cut(df['age'], 
                     #     bins=[18, 25, 35, 45, 55, 65, 100], 
                      #    labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
        
        
        #Elimino las variables que son redundantes
        print(df.head())
        
        #Detecta las variables que con objeto, o categóricas no codificadas:
        categoricas = df.select_dtypes(include=['object']).columns.tolist()
        print(f'Las columnas que son categóricas no codificadas son:\n {categoricas}')

        #Codificar las variables categóricas
        print('\nIniciando la codificación de las variables categóricas:\n')
        df=pd.get_dummies(df, columns=categoricas, drop_first=True, dtype=int)
        
        

        print(df.head())
        df.to_csv(ruta+'data/processed/df.csv')
        print(df.columns)
        return df
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        
        
""" 
# Ejemplo de uso
if __name__ == "__main__":
    
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"  # Reemplazar con la ruta correcta
    procesado(file_path)
    
 """