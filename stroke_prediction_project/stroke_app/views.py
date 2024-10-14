import pickle
import pandas as pd
from django.shortcuts import render
from stroke_app.forms import StrokePredictionForm
from stroke_app.models import StrokePrediction
from django.conf import settings

def load_model():
    model_path = settings.MODEL_PATH
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

model = load_model()

def predict_stroke(request):
    if request.method == 'POST':
        form = StrokePredictionForm(request.POST)
        if form.is_valid():
            # Crear una instancia del modelo StrokePrediction con los datos del formulario
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
                smoking_status=form.cleaned_data['smoking_status']
            )

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
                'smoking_status': [form.cleaned_data['smoking_status']]
            })

            try:
                # Realizar la predicción
                prediction = model.predict(input_data)
                stroke_risk = 'yes' if prediction[0] == 1 else 'no'
            except Exception as e:
                stroke_risk = f"Error en la predicción: {e}"

            # Asignar el resultado de la predicción a stroke_risk
            stroke_prediction.stroke_risk = stroke_risk

            # Guardar los datos en la base de datos, incluyendo stroke_risk
            stroke_prediction.save()

            # Renderizar la respuesta con el formulario y el resultado
            return render(request, 'stroke_app/prediction_form.html', {'form': form, 'risk': stroke_risk})
    
    else:
        form = StrokePredictionForm()
    
    return render(request, 'stroke_app/prediction_form.html', {'form': form})
