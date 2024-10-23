#import streamlit as st
import os
import sys
import requests
import sqlite3
import pandas as pd
import datetime
import uuid
from datetime import datetime


import pandas as pd
import numpy as np

# Elimina la columna 'id'
def remove_id(df):
    """
    Elimina la primera columna del DataFrame (por ejemplo, la columna de ID).
    """
    df = df.iloc[:, 1:]
    print(df.head())
    return df

# Elimina variables redundantes
def remove_redundant_variables(df):
    """
    Elimina variables redundantes del DataFrame.
    """
    df = df.drop('ever_married', axis=1)
    print("Variables redundantes eliminadas.")
    return df

# Categoriza la variable 'age'
def categorize_age(df):
    """
    Categoriza la variable 'age' y elimina la columna original.
    """
    df['age_cat'] = pd.cut(df['age'], 
                           bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                           labels=['1-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    df = df.drop('age', axis=1)
    print("Edad categorizada.")
    return df

# Categoriza la variable 'bmi'
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
    
    df['bmi_cat'] = np.select(conditions, categories)
    df = df.drop('bmi', axis=1)
    print("BMI categorizado.")
    return df

# Categoriza el nivel de glucosa promedio
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
    
    df['glucose_cat'] = np.select(conditions, categories)
    df = df.drop('avg_glucose_level', axis=1)
    print("Glucosa categorizada.")
    return df

# Codifica las variables categóricas
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

# Función principal de procesado
def procesado(df):
    """
    Función principal que coordina el procesamiento de los datos.
    Procesa directamente un DataFrame.
    """
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
        
        # Si quieres guardar el resultado procesado en un archivo CSV
        # df.to_csv('ruta/del/archivo/processed_df.csv', index=False)
        print(df.columns)
        print("Variables en STREAM \n:"+df.columns)
        return df
    else:
        print("No se pudieron cargar los datos. El DataFrame está vacío.")
        return None






#--------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
# Inserta el path al script de preprocesamiento



from app import StrokeFeatures
#st.title('Predicción de Accidente Cerebrovascular')

# Función para convertir 'Sí'/'No' a 1/0
def convertir_a_binario(opcion):
    return 1.0 if opcion == 'Yes' else 0.0

# Formulario de entrada
#BMI = st.number_input('Índice de Masa Corporal (BMI)', min_value=0.0, max_value=50.0, value=25.0)
""" estatura=st.number_input('Estatura (m)', min_value=1.0, max_value=210.0)
peso=st.number_input('Peso (kg)', min_value=0.0)
BMI=round((peso/(estatura)**2),2) """

""" gender = st.selectbox('Género', ['Male', 'Female'])
age = st.slider('Edad', min_value=0, max_value=100, value=30)
hypertension = st.selectbox('Hipertensión', ['1', '0'])
heart_disease = st.selectbox('Enfermedad cardíaca', ['0', '1'])
ever_married = st.selectbox('¿Alguna vez ha estado casado?', ['Yes', 'No'])
work_type = st.selectbox('Tipo de trabajo', ['Private', 'Self-employed', 'children', 'Govt_job'])
Residence_type = st.selectbox('Tipo de residencia', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Nivel promedio de glucosa', min_value=0.0, value=100.0)
bmi = st.number_input('Índice de Masa Corporal (BMI)', min_value=0.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox('Estado de fumador', ['never smoked', 'Unknown', 'formerly smoked','smokes'])
 """

#if st.button('Predecir'):
    # Preparar los datos para la API
original_data = pd.DataFrame.from_dict({
    'gender': 'male',
    'age': 30,
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': 1,
    'work_type': 'Private',
    'Residence_type': 'Rutal',
    'avg_glucose_level': 49,
    'bmi': 40,
    'smoking_status': 'smokes'
}, orient='index').T

# Procesar los datos
processed_data = procesado(original_data)
#st.write(processed_data)
print(processed_data)

# Combinar datos originales y procesados
combined_data = original_data.copy()
for col in processed_data.columns:
    if col not in combined_data.columns:
        combined_data[col] = processed_data[col]



# Convertir a diccionario

data_dict = original_data.to_dict(orient='records')[0]

# Hacer la solicitud a la API
#data = stroke_features.model_dump()

response = requests.post("http://localhost:8000/predict", json=data_dict)

if response.status_code == 200:
    result = response.json()
    probability = result["probability"]
    #st.write(f"La probabilidad de tener un accidente cerebrovascular es: {probability:.2%}")
    print('Se ha realizado la predicción')
    print(f'La probabilidad de tener un accidente cerebrovascular es: {probability:.2%}')
else:
    #st.error("Hubo un error al hacer la predicción.")
    print('Hubo un error al hacer la predicción.')
    #st.error(f"Detalles del error: {response.text}")
    

                
# Mostrar predicciones anteriores
#st.subheader("Predicciones Anteriores")

    try:
        conn = sqlite3.connect('stroke.db')
        query = "SELECT * FROM predictions_stroke ORDER BY id DESC LIMIT 10"
        df = pd.read_sql(query, conn)
        #st.dataframe(df)
    except sqlite3.Error as e:
        #st.error(f"Error al conectar con la base de datos: {e}")
        print('Error al conectar con la base de datos:')
    finally:
        if conn:
            conn.close()