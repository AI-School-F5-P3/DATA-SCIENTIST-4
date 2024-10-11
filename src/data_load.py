import pandas as pd
import os

def load_data(file_path):
    """
    Carga los datos del archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados o None si hay un error.
    """

    # Obtener la ruta absoluta del archivo para evitar problemas con rutas relativas
    abs_file_path = os.path.abspath(file_path)

    # Verificar si el archivo existe en la ruta especificada
    if not os.path.exists(abs_file_path):
        # Imprimir mensajes de error detallados si el archivo no existe
        print(f"Error: El archivo no existe en la ruta: {abs_file_path}")
        print(f"Directorio actual: {os.getcwd()}")  # Mostrar el directorio actual para ayudar a depurar
        print("Contenido del directorio:")
        try:
            # Intentar listar el contenido del directorio para ayudar a identificar el problema
            print(os.listdir(os.path.dirname(abs_file_path)))
        except FileNotFoundError:
            print("El directorio no existe.")  # Informar si el directorio tampoco existe
        return None  # Devolver None para indicar que la carga falló

    try:
        # Intentar cargar los datos utilizando pandas
        df = pd.read_csv(abs_file_path)
        print(f"Datos cargados exitosamente. Shape: {df.shape}")  # Confirmar la carga exitosa
        return df
    except Exception as e:
        # Capturar cualquier excepción que ocurra durante la carga e imprimir un mensaje de error
        print(f"Error al cargar los datos: {e}")
        return None

# Bloque de ejemplo de uso (solo se ejecuta si el script se ejecuta directamente)
if __name__ == "__main__":
    # Obtener la ruta al archivo CSV dentro de la estructura del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "DATA-SCIENTIST-4","data", "raw", "stroke_dataset.csv")

    print(f"Intentando cargar el archivo desde: {data_path}")

    # Llamar a la función load_data para cargar los datos
    df = load_data(data_path)
    if df is not None:
        # Imprimir las primeras filas del DataFrame si la carga fue exitosa
        print(df.head())
    else:
        # Imprimir un mensaje de error si la carga falló
        print("No se pudieron cargar los datos. Verifica la ruta del archivo y su existencia.")
        
        
        