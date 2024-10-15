import streamlit as st
import os
import sys
import requests
import sqlite3
import pandas as pd
import datetime
import uuid
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
from app import StrokeFeatures
st.title('Predicción de Accidente Cerebrovascular')

# Función para convertir 'Sí'/'No' a 1/0
def convertir_a_binario(opcion):
    return 1.0 if opcion == 'Yes' else 0.0

# Formulario de entrada
#BMI = st.number_input('Índice de Masa Corporal (BMI)', min_value=0.0, max_value=50.0, value=25.0)
""" estatura=st.number_input('Estatura (m)', min_value=1.0, max_value=210.0)
peso=st.number_input('Peso (kg)', min_value=0.0)
BMI=round((peso/(estatura)**2),2) """
gender = st.selectbox('Género', ['Male', 'Female'])
age = st.slider('Edad', min_value=0, max_value=100, value=30)
hypertension = st.selectbox('Hipertensión', ['1', '0'])
heart_disease = st.selectbox('Enfermedad cardíaca', ['0', '1'])
ever_married = st.selectbox('¿Alguna vez ha estado casado?', ['Yes', 'No'])
work_type = st.selectbox('Tipo de trabajo', ['Private', 'Self-employed', 'children', 'Govt_job'])
Residence_type = st.selectbox('Tipo de residencia', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Nivel promedio de glucosa', min_value=0.0, value=100.0)
bmi = st.number_input('Índice de Masa Corporal (BMI)', min_value=0.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox('Estado de fumador', ['never smoked', 'Unknown', 'formerly smoked','smokes'])

# Crear un objeto StrokeFeatures
stroke_features = StrokeFeatures(
    id=str(uuid.uuid4()),
    gender=gender,
    age=float(age),
    hypertension=hypertension,
    heart_disease=int(heart_disease),
    ever_married=ever_married,
    work_type=work_type,
    Residence_type=Residence_type,
    avg_glucose_level=avg_glucose_level,
    bmi=bmi,
    smoking_status=smoking_status,
    stroke=0,  # Asumimos que es 0 por defecto, ya que estamos prediciendo
    prediction_time=datetime.now().isoformat()
)

if st.button('Predecir'):
    # Preparar los datos para la API
   
    data = {
    'id': str(uuid.uuid4()),
    'gender': gender,
    'age': age,
    'hypertension': convertir_a_binario(hypertension),
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': Residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status,
    'stroke': 0,  # Asumimos que es 0 por defecto, ya que estamos prediciendo
    'prediction_time': datetime.now()
}

    
    # Hacer la solicitud a la API
    data = stroke_features.model_dump()
    response = requests.post("http://localhost:8000/predict", json=data)
    
    if response.status_code == 200:
        result = response.json()
        probability = result["probability"]
        st.write(f"La probabilidad de tener un accidente cerebrovascular es: {probability:.2%}")
    else:
        st.error("Hubo un error al hacer la predicción.")
        st.error(f"Detalles del error: {response.text}")
        

                
# Mostrar predicciones anteriores
st.subheader("Predicciones Anteriores")

try:
    conn = sqlite3.connect('stroke.db')
    query = "SELECT * FROM predictions_stroke ORDER BY id DESC LIMIT 10"
    df = pd.read_sql(query, conn)
    st.dataframe(df)
except sqlite3.Error as e:
    st.error(f"Error al conectar con la base de datos: {e}")
finally:
    if conn:
        conn.close()