from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import sqlite3
import os
from datetime import datetime
import uuid
import pandas as pd

app = FastAPI()

# Load the model
with open('models/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Database configuration
db_path = 'stroke.db'

class StrokeFeatures(BaseModel):
    id: str = str(uuid.uuid4())
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    work_type: str
    ever_married: str #la incluyo?
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    stroke: int = 0
    prediction_time: str = datetime.now().isoformat()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

def categorize_age(age):
    bins = [0, 18, 25, 35, 45, 55, 65, 100]
    labels = ['1-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    return pd.cut([age], bins=bins, labels=labels)[0]

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Bajo'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25.0 <= bmi < 29.9:
        return 'Sobrepeso'
    else:
        return 'Obeso'

def categorize_glucose(glucose):
    if glucose < 140:
        return 'Normal'
    elif 140 <= glucose < 200:
        return 'Prediabético'
    else:
        return 'Diabético'

def preprocess_features(data_input):
    df = pd.DataFrame([data_input])
    
    # Apply categorical transformations
    df['age_cat'] = df['age'].apply(categorize_age)
    df['bmi_cat'] = df['bmi'].apply(categorize_bmi)
    df['glucose_cat'] = df['avg_glucose_level'].apply(categorize_glucose)
    
    # Drop original columns
    df = df.drop(['age', 'bmi', 'avg_glucose_level'], axis=1)
    
    # Encode categorical variables
    categorical_columns = ['gender', 'work_type', 'Residence_type', 'smoking_status', 'age_cat', 'bmi_cat', 'glucose_cat']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    print(df_encoded.columns)
    # Ensure all expected columns are present
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match the model's expected input
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded



data_input = pd.DataFrame.from_dict({
    'gender': 'Male',
    'age': 80,
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': 1,
    'work_type': 'Private',
    'Residence_type': 'Rural',
    'avg_glucose_level': 49,
    'bmi': 40,
    'smoking_status': 'smokes'
}, orient='index').T

# Preprocess the input features
data_procesado = preprocess_features(data_input.values)
print(f'Las procesadas son ;\n\n{data_procesado}')

@app.post("/predict")
async def predict_stroke(data_procesado):
    try:
        
        
        # Make prediction
        prediction = model.predict(data_procesado)[0][1]
        print(f'{prediction}')
        # Save to database
        save_to_database(data_procesado, prediction) ###############################################3
        
        return {"probability": prediction} #Esto es un diccionario
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_to_database(features: StrokeFeatures, prediction: int):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """INSERT INTO predictions_stroke
           (gender, age, hypertension, heart_disease, work_type, 
            Residence_type, avg_glucose_level, bmi, smoking_status, stroke, 
            prediction_time, prediction) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        values = (features.gender, features.age, features.hypertension, features.heart_disease,
            features.work_type, features.Residence_type,
            features.avg_glucose_level, features.bmi, features.smoking_status, features.stroke,
            datetime.now().isoformat(), prediction)
        cursor.execute(query, values)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving to database: {e}")
    finally:
        if conn:
            conn.close()

# Ensure the database exists
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE predictions_stroke
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              gender TEXT, 
              age INTEGER, 
              hypertension INTEGER, 
              heart_disease INTEGER, 
              work_type TEXT, 
              Residence_type TEXT, 
              avg_glucose_level REAL, 
              bmi REAL, 
              smoking_status TEXT, 
              stroke INTEGER, 
              prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              prediction REAL)''')
    conn.commit()
    conn.close()
    
    
    
    


import pandas as pd
import numpy as np



#--------------------------------------------
import os
import sys
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



#if st.button('Predecir'):
    # Preparar los datos para la API

#st.write(processed_data)





# Convertir a diccionario

data_dict = data_procesado.to_dict(orient='records')[0]

# Hacer la solicitud a la API
#data = stroke_features.model_dump()

import requests
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