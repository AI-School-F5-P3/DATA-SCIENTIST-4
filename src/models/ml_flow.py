import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Cargar el dataset
data = pd.read_csv('data/processed/stroke_dataset_encoded.csv')  # Cambia esto por tu archivo CSV

# Separar características y variable objetivo
X = data.drop(columns=['stroke'])  # Asegúrate de que 'stroke' es la variable objetivo
y = data['stroke']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el DataFrame escalado en CSV
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
scaled_df['stroke'] = y.values  # Agregar la variable objetivo
scaled_df.to_csv('data/processed/scaled_data.csv', index=False)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Listado de modelos a evaluar
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# Crear un DataFrame para almacenar las métricas
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Iniciar MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')  # Asegúrate de tener un servidor MLflow corriendo
mlflow.start_run()

for model_name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Guardar métricas en el DataFrame
    metrics_df = metrics_df.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }, ignore_index=True)

    # Guardar el modelo con MLflow
    mlflow.sklearn.log_model(model, model_name)

    # Crear y guardar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)  # Cambia el rango si hay más clases
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'reports/figures/confusion_matrix_{model_name}.png')
    plt.close()

# Guardar las métricas en un archivo de Excel
metrics_df.to_excel('reports/model_metrics.xlsx', index=False)

# Finalizar el seguimiento de MLflow
mlflow.end_run()

print("El proceso ha finalizado. Los resultados se han guardado.")
