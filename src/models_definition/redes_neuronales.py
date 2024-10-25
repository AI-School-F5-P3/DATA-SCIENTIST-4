import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
import joblib
import os
import mlflow
import mlflow.keras
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import time
from sklearn.utils.class_weight import compute_class_weight

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables
model_name = "Neural Network"
mlflow.set_tracking_uri("file:///C:/Users/avkav/Documents/BootcampAI/ProyectoDataScientist-Grupo4/DATA-SCIENTIST-4/mlruns")

def load_and_preprocess_data(file_path):
    try:
        logging.info(f"Cargando y preprocesando el dataset desde {file_path}")
        data = pd.read_csv(file_path)
        X = data.drop(columns=['stroke'])
        y = data['stroke']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        logging.info(f"Distribución de clases: {dict(y.value_counts())}")
        logging.info(f"Pesos calculados para las clases: {class_weight_dict}")

        return X_scaled, y, X.columns, scaler, class_weight_dict
    except Exception as e:
        logging.error(f"Error en la carga y preprocesamiento de datos: {str(e)}")
        raise

def apply_smote(X, y):
    try:
        start_time = time.time()
        oversample = SMOTE(sampling_strategy=0.5)
        undersample = RandomUnderSampler(sampling_strategy=0.8)
        
        pipeline = Pipeline([
            ('SMOTE', oversample),
            ('RandomUnderSampler', undersample)
        ])
        
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        end_time = time.time()
        
        logging.info(f"Distribución de clases después de SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de balanceo: {execution_time:.2f} segundos")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error al aplicar el balanceo de datos: {str(e)}")
        raise

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test, class_weight_dict):
    try:
        model = create_model(X_train.shape[1])
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            class_weight=class_weight_dict,
            validation_split=0.2,
            verbose=2
        )
        
        y_pred_proba = model.predict(X_test)
        best_threshold = find_best_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return model, metrics, y_pred, y_pred_proba, history.history, best_threshold
    except Exception as e:
        logging.error(f"Error en el entrenamiento y evaluación del modelo: {str(e)}")
        raise

def find_best_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0, 1, 0.01)
    f1_scores = [f1_score(y_true, (y_pred_proba >= threshold).astype(int)) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    logging.info(f"Mejor umbral encontrado: {best_threshold}")
    return best_threshold

def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reports/figures/training_history.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name):
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        output_path = f'reports/figures/confusion_matrix_{model_name}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Gráfico de matriz de confusión guardado para el modelo {model_name}.")
    except Exception as e:
        logging.error(f"Error al generar la matriz de confusión: {str(e)}")
        raise

def plot_roc_curve(y_test, y_pred_proba):
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.savefig('reports/figures/roc_curve.png')
        plt.close()
        logging.info("Gráfico de curva ROC guardado.")
    except Exception as e:
        logging.error(f"Error al generar la curva ROC: {str(e)}")
        raise

def log_mlflow(model, metrics, model_name, history):
    try:
        with mlflow.start_run():
            # Registrar parámetros del modelo
            mlflow.log_param("layers", [32, 16, 1])
            mlflow.log_param("activation", "relu, relu, sigmoid")
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("learning_rate", 0.001)
            
            # Registrar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Registrar métricas de entrenamiento final
            mlflow.log_metric("final_training_accuracy", history['accuracy'][-1])
            mlflow.log_metric("final_validation_accuracy", history['val_accuracy'][-1])
            mlflow.log_metric("final_training_loss", history['loss'][-1])
            mlflow.log_metric("final_validation_loss", history['val_loss'][-1])
            
            # Guardar el modelo en MLflow
            mlflow.keras.log_model(model, model_name)
            
        logging.info(f"Métricas del modelo {model_name} registradas en MLflow.")
    except Exception as e:
        logging.error(f"Error al registrar en MLflow: {str(e)}")
        raise

def generate_report(model, metrics, history, best_threshold):
    try:
        report = f"Informe Detallado del Modelo {model_name} para la Predicción de Ictus\n"
        report += "=" * 50 + "\n\n"

        report += "1. Arquitectura del Modelo\n"
        model.summary(print_fn=lambda x: report + x + '\n')
        report += "\n"

        report += f"2. Mejor Umbral Encontrado: {best_threshold:.4f}\n\n"

        report += "3. Métricas Finales del Modelo\n"
        for metric_name, metric_value in metrics.items():
            report += f"- {metric_name.capitalize()}: {metric_value:.4f}\n"

        report += "\n4. Métricas de Entrenamiento Final\n"
        report += f"- Training Accuracy: {history['accuracy'][-1]:.4f}\n"
        report += f"- Validation Accuracy: {history['val_accuracy'][-1]:.4f}\n"
        report += f"- Training Loss: {history['loss'][-1]:.4f}\n"
        report += f"- Validation Loss: {history['val_loss'][-1]:.4f}\n"

        output_dir = 'reports/'
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/report_{model_name.lower().replace(' ', '_')}.txt", "w") as f:
            f.write(report)

        logging.info(f"Informe guardado como 'report_{model_name.lower().replace(' ', '_')}.txt'.")
    except Exception as e:
        logging.error(f"Error al generar el informe: {str(e)}")
        raise

if __name__ == "__main__":
    # Cargar y preprocesar los datos
    X_scaled, y, feature_names, scaler, class_weight_dict = load_and_preprocess_data('data/processed/dataset_escalado_copiaeda.csv')

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Aplicar balanceo de datos
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Entrenar y evaluar el modelo
    model, metrics, y_pred, y_pred_proba, history, best_threshold = train_and_evaluate_model(
        X_train_balanced, X_test, y_train_balanced, y_test, class_weight_dict
    )

    # Generar visualizaciones
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_pred_proba)

    # Registrar en MLflow
    log_mlflow(model, metrics, model_name, history)

    # Generar informe
    generate_report(model, metrics, history, best_threshold)

    # Guardar modelo y escalador
    model_dir = f'models/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Guardar el modelo en formato Keras
    model.save(f'{model_dir}/model_{model_name.lower().replace(" ", "_")}.keras')
    
    # Guardar el escalador y el mejor umbral
    joblib.dump(scaler, f'{model_dir}/scaler_{model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(best_threshold, f'{model_dir}/threshold_{model_name.lower().replace(" ", "_")}.pkl')
    
    print(f"Modelo, umbral óptimo y escalador guardados en '{model_dir}'.")