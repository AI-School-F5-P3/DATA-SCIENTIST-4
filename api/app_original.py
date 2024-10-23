from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import sqlite3
import os
from datetime import datetime
import uuid


app = FastAPI() #Para correr la API en el prompt -->   uvicorn api.app:app --reload  

# Cargar el modelo y el scaler
with open('models/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Asegurarse de que la base de datos existe
db_path = 'stroke.db'
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE predictions_stroke
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              gender TEXT, 
              age INTEGER, 
              hypertension INTEGER, 
              heart_disease INTEGER, 
              ever_married TEXT, 
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

class StrokeFeatures(BaseModel):
    id: str = str(uuid.uuid4())
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
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

@app.post("/predict")
async def predict_stroke(features: StrokeFeatures):
    try:
        gender_encoded = 1 if features.gender == 'Male' else 0
        ever_married_encoded = 1 if features.ever_married == 'Yes' else 0
        
        work_type_encoded = {
            'Private': 0,
            'Self-employed': 1,
            'children': 2,
            'Govt_job': 3
        }.get(features.work_type, 4)  # 4 para cualquier otro valor
        residence_type_encoded = 1 if features.Residence_type == 'Urban' else 0
        smoking_status_encoded = {
            'never smoked': 0,
            'Unknown': 1,
            'formerly smoked': 2,
            'smokes': 3
        }.get(features.smoking_status, 1)  # 1 (Unknown) para cualquier otro valor

        feature_array = np.array([[
            gender_encoded, features.age, features.hypertension, features.heart_disease,
            ever_married_encoded, work_type_encoded, residence_type_encoded,
            features.avg_glucose_level, features.bmi, smoking_status_encoded
        ]])
        
        
        
        
    
        
        #scaled_features = scaler.transform(feature_array)
        prediction = model.predict_proba(feature_array)[0][1]
        
        save_to_database(features, prediction)
        
        return {"probability": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_to_database(features: StrokeFeatures, prediction: float):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """INSERT INTO predictions_stroke
           (gender, age, hypertension, heart_disease, ever_married, work_type, 
            Residence_type, avg_glucose_level, bmi, smoking_status, stroke, 
            prediction_time, prediction) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        values = (features.gender, features.age, features.hypertension, features.heart_disease,
            features.ever_married, features.work_type, features.Residence_type,
            features.avg_glucose_level, features.bmi, features.smoking_status, features.stroke,datetime.now().isoformat(),prediction)
        cursor.execute(query, values)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error al guardar en la base de datos: {e}")
    finally:
        if conn:
            conn.close()