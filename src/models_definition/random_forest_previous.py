import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import os
import mlflow
import mlflow.sklearn
import joblib
import logging
import time

# Variables
file_path = 'data/processed/stroke_dataset_encoded.csv'
model_name = "Random Forest"


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configura MLflow para usar un directorio local para el seguimiento
# Aquí es donde ajustamos la URI para que sea absoluta en lugar de relativa
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

def save_scaled_data(X_scaled, y, columns, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scaled_df = pd.DataFrame(X_scaled, columns=columns)
        scaled_df['stroke'] = y
        scaled_df.to_csv(output_path, index=False)
        logging.info(f"Datos escalados guardados en '{output_path}'.")
    except Exception as e:
        logging.error(f"Error al guardar los datos escalados: {str(e)}")
        raise

def apply_smote(X, y):
    try:
        # Medir el tiempo de ejecución
        start_time = time.time()
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        end_time = time.time()
        logging.info(f"Distribución de clases después de SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
        return X_resampled, y_resampled
        # Calcular y mostrar el tiempo de ejecución
        
    except Exception as e:
        logging.error(f"Error al aplicar SMOTE: {str(e)}")
        raise

def cross_validation_evaluate_model(X_train, y_train, model, cv=5):
    try:
        logging.info("Realizando validación cruzada...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        logging.info(f"Scores de validación cruzada: {cv_scores}")
        logging.info(f"Precisión promedio de validación cruzada: {cv_scores.mean():.4f}")
        return cv_scores
    except Exception as e:
        logging.error(f"Error en la validación cruzada: {str(e)}")
        raise

def detect_overfitting(model, X_train, y_train, X_test, y_test):
    """Calcula las métricas en los sets de entrenamiento y prueba para detectar overfitting."""
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        logging.info(f"Precisión en el conjunto de entrenamiento: {train_accuracy:.4f}")
        logging.info(f"Precisión en el conjunto de prueba: {test_accuracy:.4f}")
        
        overfitting_score = train_accuracy - test_accuracy
        if overfitting_score > 0.1:
            logging.warning(f"Posible overfitting detectado: diferencia en precisión de {overfitting_score:.4f} entre entrenamiento y prueba.")
        else:
            logging.info("No se ha detectado un overfitting significativo.")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_score': overfitting_score
        }
    except Exception as e:
        logging.error(f"Error al detectar overfitting: {str(e)}")
        raise

def train_and_evaluate_model(X_train, X_test, y_train, y_test, class_weight_dict):
    try:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [class_weight_dict, 'balanced', 'balanced_subsample']
        }

        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        best_threshold = find_best_threshold(y_test, y_pred_proba)
        y_pred_adjusted = (y_pred_proba >= best_threshold).astype(int)

        metrics = calculate_metrics(y_test, y_pred_adjusted, y_pred_proba)

        return best_model, metrics, y_pred_adjusted, y_pred_proba, grid_search.best_params_, best_threshold
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

def log_mlflow(model, metrics, model_name, best_params):
    try:
        # Inicia un nuevo run de MLflow
        with mlflow.start_run():
            mlflow.log_params(best_params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.sklearn.log_model(model, model_name)
        logging.info(f"Métricas del modelo {model_name} registradas en MLflow.")
    except Exception as e:
        logging.error(f"Error al registrar en MLflow: {str(e)}")
        raise

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

def plot_feature_importance(model, feature_names):
    try:
        importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title('Top 10 Características Más Importantes')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png')
        plt.close()
        logging.info("Gráfico de importancia de características guardado.")
    except Exception as e:
        logging.error(f"Error al generar el gráfico de importancia de características: {str(e)}")
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

# Modificar el informe para incluir la validación cruzada y el reporte de overfitting
def generate_report(model, X_train, X_test, y_train, y_test, X, y, feature_names, metrics, best_params, class_weight_dict, best_threshold, cv_scores, overfitting_metrics):
    try:
        report = f"Informe Detallado del Modelo {model_name} para la Predicción de Ictus\n"
        report += "=" * 50 + "\n\n"

        report += f"1. Resumen del Conjunto de Datos\n"
        report += f"Total de muestras: {len(X)}\n"
        report += f"Características: {X.shape[1]}\n"
        report += f"Distribución de clases: {dict(pd.Series(y).value_counts())}\n\n"

        report += f"2. Mejor Umbral Encontrado: {best_threshold:.2f}\n\n"

        report += f"3. Hiperparámetros del Mejor Modelo\n"
        for param, value in best_params.items():
            report += f"- {param}: {value}\n"

        report += "\n4. Métricas del Modelo\n"
        for metric_name, metric_value in metrics.items():
            report += f"- {metric_name.capitalize()}: {metric_value:.4f}\n"

        report += "\n5. Validación Cruzada\n"
        report += f"Scores de validación cruzada: {cv_scores}\n"
        report += f"Precisión promedio: {cv_scores.mean():.4f}\n"

        report += "\n6. Detección de Overfitting\n"
        report += f"Precisión en entrenamiento: {overfitting_metrics['train_accuracy']:.4f}\n"
        report += f"Precisión en prueba: {overfitting_metrics['test_accuracy']:.4f}\n"
        report += f"Diferencia (Overfitting): {overfitting_metrics['overfitting_score']:.4f}\n"

        report += "\n7. Importancia de las Características\n"
        importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=False).head(10)
        for idx, row in importances.iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"

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
    X_scaled, y, feature_names, scaler, class_weight_dict = load_and_preprocess_data(file_path)

    # Guardar los datos escalados (por si lo necesitas en el futuro)
    save_scaled_data(X_scaled, y, feature_names, 'data/processed/stroke_dataset_scaled.csv')

    # Aplicar SMOTE
    X_resampled, y_resampled = apply_smote(X_scaled, y)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Entrenar y evaluar el modelo
    best_model, metrics, y_pred_adjusted, y_pred_proba, best_params, best_threshold = train_and_evaluate_model(X_train, X_test, y_train, y_test, class_weight_dict)

    # Realizar la validación cruzada
    cv_scores = cross_validation_evaluate_model(X_train, y_train, best_model)

    # Detectar overfitting
    overfitting_metrics = detect_overfitting(best_model, X_train, y_train, X_test, y_test)

    # Guardar modelo y escalador
    model_dir = f'models/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    # joblib.dump(best_model, f'{model_dir}/model_{model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler_{model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump((best_model, best_threshold), f'{model_dir}/model_{model_name.lower().replace(" ", "_")}.pkl')
    print(f"Modelo, umbral óptimo y escalador guardados en '{model_dir}'.")

    # Registrar los resultados en MLflow
    log_mlflow(best_model, metrics, model_name, best_params)

    # Graficar la matriz de confusión
    plot_confusion_matrix(y_test, y_pred_adjusted, model_name)

    # Graficar la importancia de las características
    plot_feature_importance(best_model, feature_names)

    # Graficar la curva ROC
    plot_roc_curve(y_test, y_pred_proba)

    # Generar un informe detallado con los nuevos parámetros
    generate_report(
        best_model, 
        X_train, X_test, y_train, y_test, 
        X_scaled, y, feature_names, 
        metrics, best_params, class_weight_dict, 
        best_threshold, cv_scores, overfitting_metrics
    )