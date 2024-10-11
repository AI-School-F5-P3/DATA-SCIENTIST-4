import pandas as pd
import os
import sys
#from pandas_profiling import ProfileReport ya no sirve la versión
from ydata_profiling import ProfileReport
from pandas_profiling import ProfileReport

 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_load import load_data #importo la función de carga que está en el archivo data_load.py


def explorar_datos(file_path):
    """
    Explora los datos cargados desde el archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        None
    """
    # Llamar a la función load_data para cargar los datos
    df = load_data(file_path)
    ruta="C:/4_F5/017_track_01/DATA-SCIENTIST-4/"

    if df is not None:
        # Realizar la exploración de datos aquí
        print("Explorando datos...")
        print(df.describe())  # Mostrar estadísticas descriptivas
        print(df.head())  # Mostrar las primeras filas del DataFrame
        
        # Generar un reporte
        profile_stroke = ProfileReport(df, title="Reporte Exploratorio")
        print('\n\nGenerando el archivo de reporte:\n')
        # Genera el html
        profile_stroke.to_file(ruta+"/outputs/graphs/EDA-stroke.html")
        
        print("\nSacando información por terminal:\n")
        print(df.info())
        print(f'\n\nEl DF tiene {df.shape[0]:,} filas')
        print(f'El DF tiene {df.shape[1]} columnas')
        
        
        # Comprobar cuántos valores missing existen en el DataFrame por variable
        nulos={'conteo_nulos':df.isnull().sum(),'proporcion%':round(df.isnull().sum()/df.shape[0]*100, 3)}

        df_nulos=pd.DataFrame(data=nulos)
        print(df_nulos)
        

        # Contar valores únicos por columna con información adicional
        for col in df.columns:
            unique_values = df[col].nunique()
            total_values = len(df[col])
            unique_percentage = (unique_values / total_values) * 100
            print(f"Columna {col}:")
            print(f"  Valores únicos: {unique_values}")
            print(f"  Porcentaje de valores únicos: {unique_percentage:.2f}%")
            print()
    else:
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")

""" # Ejemplo de uso
if __name__ == "__main__":
    
    file_path = "C:/4_F5/017_track_01/DATA-SCIENTIST-4/data/raw/stroke_dataset.csv"  # Reemplazar con la ruta correcta
    explorar_datos(file_path) """