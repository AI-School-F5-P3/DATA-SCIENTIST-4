# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Función para obtener la ruta absoluta del directorio del proyecto
def get_project_root():
    current_path = os.path.abspath(__file__)
    while True:
        current_path, tail = os.path.split(current_path)
        if 'models' in os.listdir(current_path):
            return current_path
        if tail == '':
            raise FileNotFoundError("No se pudo encontrar el directorio del proyecto con la carpeta 'models'")

# Ajustar la ruta para acceder a los archivos .pkl
@st.cache_resource
def load_model_and_scaler():
    try:
        project_root = get_project_root()
        model_path = os.path.join(project_root, 'models', 'random_forest', 'model_random_forest.pkl')
        scaler_path = os.path.join(project_root, 'models', 'random_forest', 'scaler_random_forest.pkl')
        
        with open(model_path, 'rb') as file:
            model, best_threshold = joblib.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = joblib.load(file)
        return model, scaler, best_threshold
    except Exception as e:
        st.error(f"Error al cargar el modelo y el scaler: {str(e)}")
        raise

# Cargar el modelo y el scaler
try:
    model, scaler, best_threshold = load_model_and_scaler()
    st.success("Modelo y scaler cargados correctamente.")
except Exception as e:
    st.error(f"No se pudo cargar el modelo o el scaler: {str(e)}")
    st.stop()

st.title('Predictor de Riesgo de Ictus')

st.write('Por favor, ingrese los siguientes datos del paciente:')

# Mapeos para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
ever_married_map = {'No': 0, 'Yes': 1}
work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
residence_type_map = {'Urban': 0, 'Rural': 1}
smoking_status_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}

# Crear inputs para cada característica
gender = st.selectbox('Género', list(gender_map.keys()))
age = st.number_input('Edad', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hipertensión', [0, 1])
heart_disease = st.selectbox('Enfermedad cardíaca', [0, 1])
ever_married = st.selectbox('Alguna vez casado', list(ever_married_map.keys()))
work_type = st.selectbox('Tipo de trabajo', list(work_type_map.keys()))
residence_type = st.selectbox('Tipo de residencia', list(residence_type_map.keys()))
avg_glucose_level = st.number_input('Nivel promedio de glucosa', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('IMC', min_value=10.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox('Estado de fumador', list(smoking_status_map.keys()))

def validate_input(input_data):
    if input_data['age'].values[0] < 0 or input_data['age'].values[0] > 120:
        raise ValueError("La edad debe estar entre 0 y 120 años.")
    if input_data['avg_glucose_level'].values[0] < 0:
        raise ValueError("El nivel de glucosa no puede ser negativo.")
    if input_data['bmi'].values[0] < 10 or input_data['bmi'].values[0] > 50:
        raise ValueError("El IMC debe estar entre 10 y 50.")

if st.button('Predecir'):
    # Crear un DataFrame con los datos ingresados y mapeados
    input_data = pd.DataFrame({
        'gender': [gender_map[gender]],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married_map[ever_married]],
        'work_type': [work_type_map[work_type]],
        'Residence_type': [residence_type_map[residence_type]],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status_map[smoking_status]]
    })

    try:
        validate_input(input_data)
        # Escalar los datos
        input_scaled = scaler.transform(input_data)
        
        st.write("Datos de entrada (escalados):")
        st.write(pd.DataFrame(input_scaled, columns=input_data.columns))
        
        # Hacer la predicción
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Usar el umbral personalizado
        prediction = 1 if probability >= 0.2 else 0

        # Mostrar resultados
        st.subheader('Resultados de la Predicción:')
        if prediction == 1:
            st.warning(f'El paciente tiene un alto riesgo de sufrir un ictus.')
        else:
            st.success(f'El paciente tiene un bajo riesgo de sufrir un ictus.')

        st.write(f'Probabilidad de sufrir un ictus: {probability:.2%}')
        st.write(f'Umbral de decisión: {best_threshold:.2%}')

        # Visualización adicional
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(['No Ictus', 'Ictus'], [1-probability, probability])
        ax.set_ylabel('Probabilidad')
        ax.set_title('Probabilidad de Ictus')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")

st.info('Nota: Esta herramienta es solo para fines informativos y no sustituye el diagnóstico médico profesional.')