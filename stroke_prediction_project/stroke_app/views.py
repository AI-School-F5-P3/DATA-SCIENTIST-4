import pickle
import pandas as pd
from django.shortcuts import render, redirect
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
                'Residence_type': [form.cleaned_data['Residence_type']],
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
                Residence_type=form.cleaned_data['Residence_type'],
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

# Cargar el modelo y otros archivos de preprocesamiento

def load_scaler():
    scaler_path = settings.SCALER_PATH
    return joblib.load(scaler_path)

scaler = load_scaler()

def load_encoder():
    encoder_path = settings.ENCODER_PATH
    return joblib.load(encoder_path)

encoder = load_encoder()

def preprocess_data(df):
    
    # Escalar las características numéricas
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Codificar las características categóricas
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoded_categorical = encoder.transform(df[categorical_features])

    # Combinar los datos codificados
    df_encoded = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
    df_preprocessed = pd.concat([df[numeric_features], df_encoded], axis=1)
    
    return df_preprocessed

def upload_csv_and_predict(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render(request, 'stroke_app/upload.html', {'error': f'Error al cargar el archivo CSV: {e}'})

        required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        if all(col in df.columns for col in required_columns):
            # Preprocesar los datos
            df_preprocessed = preprocess_data(df)

            # Hacer predicciones
            predictions = model.predict(df_preprocessed)

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