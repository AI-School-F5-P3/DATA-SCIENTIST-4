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


def upload_csv(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        df = pd.read_csv(file)

        # Asumiendo que tienes un modelo StrokePrediction con los campos adecuados
        for _, row in df.iterrows():
            StrokePrediction.objects.create(
                gender=row['gender'],
                age=row['age'],
                hypertension=row['hypertension'],
                heart_disease=row['heart_disease'],
                ever_married=row['ever_married'],
                work_type=row['work_type'],
                residence_type=row['Residence_type'],
                avg_glucose_level=row['avg_glucose_level'],
                bmi=row['bmi'],
                smoking_status=row['smoking_status'],
                stroke=row['stroke']  # Aquí se puede ajustar según lo que necesites
            )
        return redirect('success_page')  # Redirigir a una página de éxito o similar

    return render(request, 'stroke_app/upload.html')

def upload_view(request):
    return render(request, 'stroke_app/upload.html')

def preprocess_data(df):
    # Cargar el scaler y encoder que se guardaron
    scaler = joblib.load(settings.SCALER_PATH)
    encoder = joblib.load(settings.ENCODER_PATH)

    # Definir las características numéricas y categóricas
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']

    # Aplicar el escalado a las características numéricas
    df[numeric_features] = scaler.transform(df[numeric_features])

    # Codificar las variables categóricas
    encoded_categorical = encoder.transform(df[categorical_features])

    # Combinar el resultado
    df_encoded = pd.concat([df.drop(categorical_features, axis=1), pd.DataFrame(encoded_categorical)], axis=1)

    return df_encoded

def upload_csv_and_predict(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        df = pd.read_csv(file)

        # Asegurarse de que las columnas del CSV coincidan con lo esperado
        required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        if all(col in df.columns for col in required_columns):
            
            # Preprocesar los datos antes de hacer la predicción
            df_preprocessed = preprocess_data(df)
            
            # Hacer predicciones con el modelo preentrenado
            predictions = model.predict(df_preprocessed)

            # Agregar las predicciones al DataFrame
            df['prediction'] = predictions
            
            # Guardar los resultados en la base de datos
            for _, row in df.iterrows():
                StrokePrediction.objects.create(
                    gender=row['gender'],
                    age=row['age'],
                    hypertension=row['hypertension'],
                    heart_disease=row['heart_disease'],
                    ever_married=row['ever_married'],
                    work_type=row['work_type'],
                    residence_type=row['residence_type'],
                    avg_glucose_level=row['avg_glucose_level'],
                    bmi=row['bmi'],
                    smoking_status=row['smoking_status'],
                    stroke=row['prediction'],  # Guardar la predicción
                    date_submitted=timezone.now()
                )

            return redirect('success_page')  # Redirigir a una página de éxito o similar
        else:
            # Manejar el error si faltan columnas
            print("Faltan columnas en el CSV")
            return render(request, 'stroke_app/upload.html', {'error': 'Faltan columnas en el archivo CSV'})

    return render(request, 'stroke_app/upload.html')
