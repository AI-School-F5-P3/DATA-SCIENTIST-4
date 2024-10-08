import pickle
import pandas as pd
from django.shortcuts import render
from .forms import StrokeRiskForm

# Cargar el modelo
with open('Notebooks/finalmodel.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

def stroke_prediction_view(request):
    if request.method == 'POST':
        form = StrokeRiskForm(request.POST)
        if form.is_valid():
            # Obtener datos del formulario
            age = form.cleaned_data['age']
            hypertension = 1 if form.cleaned_data['hypertension'] == 'Yes' else 0
            heart_disease = 1 if form.cleaned_data['heart_disease'] == 'Yes' else 0
            avg_glucose_level = form.cleaned_data['avg_glucose_level']
            bmi = form.cleaned_data['bmi']
            gender = form.cleaned_data['gender']
            ever_married = form.cleaned_data['ever_married']
            work_type = form.cleaned_data['work_type']
            residence_type = form.cleaned_data['residence_type']
            smoking_status = form.cleaned_data['smoking_status']

            # Crear DataFrame
            input_data = pd.DataFrame({
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'gender': [gender],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'smoking_status': [smoking_status]
            })

            # Predecir
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                result = "The individual is at risk of stroke!"
            else:
                result = "The individual is not at risk of stroke."
            return render(request, 'prediction/result.html', {'result': result, 'form': form})

    else:
        form = StrokeRiskForm()

    return render(request, 'prediction/predict.html', {'form': form})

