import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  colorama  import  Fore
import os

"""#Normalización de datos"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
df_processed = pd.read_csv('data/processed/stroke_dataset_encoded.csv')

# Crear el scaler
scaler = StandardScaler()

# Aplicar el scaler a las columnas seleccionadas (excluyendo las columnas que no quieras escalar)
# Aquí escalamos todas las columnas. Asegúrate de excluir la columna objetivo si no deseas escalarla.
df_scaled = scaler.fit_transform(df_processed)

# Convertir el resultado a un DataFrame, manteniendo los nombres de las columnas originales
df_scaled = pd.DataFrame(df_scaled, columns=df_processed.columns)

# Mostrar el DataFrame con las columnas escaladas
print("\nDataFrame con columnas escaladas:")
print(df_scaled)

# Guardar el DataFrame preprocesado en un archivo CSV.
df_scaled.to_csv('data/processed/stroke_dataset_processed.csv', index=False)

# from google.colab import drive
# drive.mount('/content/drive')