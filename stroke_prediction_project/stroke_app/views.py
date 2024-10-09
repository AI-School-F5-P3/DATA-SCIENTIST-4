import pickle
import pandas as pd
from django.shortcuts import render
from .forms import StrokePredictionForm
import os
from django.conf import settings

def load_model():
    model_path = settings.MODEL_PATH
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

model = load_model()

def predict_stroke(request):
    risk = None # Inicializa 'risk' antes de cualquier lógica
    
    if request.method == 'POST':
        form = StrokePredictionForm(request.POST)
        if form.is_valid():
            input_data = pd.DataFrame({
                'age': [form.cleaned_data['age']],
                'hypertension': [1 if form.cleaned_data['hypertension'] == 'Yes' else 0],
                'heart_disease': [1 if form.cleaned_data['heart_disease'] == 'Yes' else 0],
                'avg_glucose_level': [form.cleaned_data['avg_glucose_level']],
                'bmi': [form.cleaned_data['bmi']],
                'gender': [form.cleaned_data['gender']],
                'ever_married': [form.cleaned_data['ever_married']],
                'work_type': [form.cleaned_data['work_type']],
                'Residence_type': [form.cleaned_data['Residence_type']],
                'smoking_status': [form.cleaned_data['smoking_status']]
            })

            try:
                prediction = model.predict(input_data)
                risk = "at risk" if prediction[0] == 1 else "not at risk"
            except Exception as e:
                risk = f"Error en la predicción: {e}"
            return render(request, 'stroke_app/prediction_form.html', {'form': form, 'risk': risk})
    else:
        form = StrokePredictionForm()
    
    return render(request, 'stroke_app/prediction_form.html', {'form': form, 'risk':risk})