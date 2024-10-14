import pickle
import pandas as pd
from django.shortcuts import render
from stroke_app.forms import StrokePredictionForm
from stroke_app.models import StrokePrediction
from django.conf import settings
from django.utils import timezone
import joblib

def load_model():
    model_path = settings.MODEL_PATH
    return joblib.load(model_path) 

model = load_model()

def predict_stroke(request):
    if request.method == 'POST':
        form = StrokePredictionForm(request.POST)
        if form.is_valid():
            # Mostrar datos del formulario para depuración
            print("Formulario válido:", form.cleaned_data)
            
            # Convertir los datos para la predicción
            input_data = pd.DataFrame({
                'age': [form.cleaned_data['age']],
                'hypertension': [1 if form.cleaned_data['hypertension'] == 'Yes' else 0],
                'heart_disease': [1 if form.cleaned_data['heart_disease'] == 'Yes' else 0],
                'avg_glucose_level': [form.cleaned_data['avg_glucose_level']],
                'bmi': [form.cleaned_data['bmi']],
                'gender': [form.cleaned_data['gender']],
                'ever_married': [form.cleaned_data['ever_married']],
                'work_type': [form.cleaned_data['work_type']],
                'Residence_type': [form.cleaned_data['residence_type']],
                'smoking_status': [form.cleaned_data['smoking_status']],
            })

            try:
                # Realizar la predicción
                prediction = model.predict(input_data)
                stroke_risk = 'yes' if prediction[0] == 1 else 'no'
                print("Predicción realizada, resultado:", stroke_risk)
            except Exception as e:
                stroke_risk = f"Error en la predicción: {e}"
                print(stroke_risk)  # Mensaje de error

            # Crear la instancia del modelo StrokePrediction
            stroke_prediction = StrokePrediction(
                age=form.cleaned_data['age'],
                hypertension=True if form.cleaned_data['hypertension'] == 'Yes' else False,
                heart_disease=True if form.cleaned_data['heart_disease'] == 'Yes' else False,
                avg_glucose_level=form.cleaned_data['avg_glucose_level'],
                bmi=form.cleaned_data['bmi'],
                gender=form.cleaned_data['gender'],
                ever_married=True if form.cleaned_data['ever_married'] == 'Yes' else False,
                work_type=form.cleaned_data['work_type'],
                residence_type=form.cleaned_data['residence_type'],
                smoking_status=form.cleaned_data['smoking_status'],
                stroke_risk=stroke_risk,
                date_submitted=timezone.now()
            )

            # Guardar los datos en la base de datos
            stroke_prediction.save()
            print("Datos guardados en la base de datos")

            # Renderizar la respuesta con el formulario y el resultado
            return render(request, 'stroke_app/prediction_form.html', {'form': form, 'risk': stroke_risk})
        else:
            print("Errores en el formulario:", form.errors)  # Muestra errores de validación

    else:
        form = StrokePredictionForm()

    return render(request, 'stroke_app/prediction_form.html', {'form': form})
