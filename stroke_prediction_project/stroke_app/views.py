from django.shortcuts import render
from stroke_app.forms import StrokePredictionForm
from stroke_app.models import StrokePrediction
from django.conf import settings
from django.utils import timezone
import pandas as pd
import joblib
import numpy as np
from django.core.exceptions import ValidationError

def load_model():
    model_path = settings.MODEL_PATH
    try:
        pipeline = joblib.load(model_path)
        return pipeline['model']  # Extraemos solo el modelo del diccionario
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def predict_stroke(request):
    form = StrokePredictionForm()
    context = {'form': form}
    
    if request.method == 'POST':
        print(request.POST)
        form = StrokePredictionForm(request.POST)
        if form.is_valid():
            try:
                # Convertir Yes/No a valores booleanos para la base de datos
                hypertension_bool = form.cleaned_data['hypertension'] == 'Yes'
                heart_disease_bool = form.cleaned_data['heart_disease'] == 'Yes'
                ever_married_bool = form.cleaned_data['ever_married'] == 'Yes'
                
                # Preparar datos para el modelo
                input_data = pd.DataFrame([{
                    'gender': form.cleaned_data['gender'],
                    'age': form.cleaned_data['age'],
                    'hypertension': 1 if hypertension_bool else 0,  # El modelo espera 1/0
                    'heart_disease': 1 if heart_disease_bool else 0,  # El modelo espera 1/0
                    'ever_married': form.cleaned_data['ever_married'],
                    'work_type': form.cleaned_data['work_type'],
                    'Residence_type': form.cleaned_data['Residence_type'],
                    'avg_glucose_level': form.cleaned_data['avg_glucose_level'],
                    'bmi': form.cleaned_data['bmi'],
                    'smoking_status': form.cleaned_data['smoking_status']
                }])

                try:
                    # Realizar predicción
                    prediction_proba = model.predict_proba(input_data)[0]
                    prediction = model.predict(input_data)[0]
                except Exception as e:
                    print(f"Error en la predicción: {e}")
                    context['error'] = str(e)
                
                # Determinar el riesgo y la probabilidad
                stroke_risk = 'High' if prediction == 1 else 'Low'
                risk_probability = f"{prediction_proba[1]:.2%}"
                
                # Guardar en la base de datos
                stroke_prediction = StrokePrediction.objects.create(
                    age=form.cleaned_data['age'],
                    hypertension=hypertension_bool,  # Guardamos el booleano
                    heart_disease=heart_disease_bool,  # Guardamos el booleano
                    avg_glucose_level=form.cleaned_data['avg_glucose_level'],
                    bmi=form.cleaned_data['bmi'],
                    gender=form.cleaned_data['gender'],
                    ever_married=ever_married_bool,  # Guardamos el booleano
                    work_type=form.cleaned_data['work_type'],
                    Residence_type=form.cleaned_data['Residence_type'],
                    smoking_status=form.cleaned_data['smoking_status'],
                    stroke_risk=stroke_risk,
                    date_submitted=timezone.now()
                )
                
                # Actualizar contexto con los resultados
                context.update({
                    'risk': stroke_risk,
                    'probability': risk_probability,
                    'show_result': True
                })
                
                print("Predicción realizada exitosamente")
                print(f"Riesgo: {stroke_risk}")
                print(f"Probabilidad: {risk_probability}")
                print("Datos guardados en la base de datos")
                
            except Exception as e:
                print(f"Error detallado: {e}")
                context['error'] = str(e)
        else:
            print("Errores en el formulario:", form.errors)
            context['error'] = "Por favor, corrija los errores en el formulario."
    
    return render(request, 'stroke_app/prediction_form.html', context)

def upload_csv_and_predict(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render(request, 'stroke_app/upload.html', {'error': f'Error al cargar el archivo CSV: {e}'})

        required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        if all(col in df.columns for col in required_columns):
            # Asegúrate de que las columnas estén en el mismo orden y tipo que el modelo espera
            input_data = df[required_columns]  # Asegúrate de que solo las columnas necesarias estén presentes
            predictions = model.predict(input_data)

            # Agregar predicciones al DataFrame
            df['stroke_risk'] = predictions

            for _, row in df.iterrows():
                StrokePrediction.objects.create(
                    gender=row['gender'],
                    age=row['age'],
                    hypertension=row['hypertension'],
                    heart_disease=row['heart_disease'],
                    ever_married=row['ever_married'],
                    work_type=row['work_type'],
                    Residence_type=row['Residence_type'],
                    avg_glucose_level=row['avg_glucose_level'],
                    bmi=row['bmi'],
                    smoking_status=row['smoking_status'],
                    stroke_risk='High' if row['stroke_risk'] == 1 else 'Low',
                    date_submitted=timezone.now()
                )

            return redirect('success_page')
        else:
            return render(request, 'stroke_app/upload.html', {'error': 'Faltan columnas en el archivo CSV'})

    return render(request, 'stroke_app/upload.html')
