import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, RocCurveDisplay
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import os
import mlflow
import mlflow.sklearn
import joblib
import logging
import time

# Variables
file_path = 'data/processed/stroke_dataset_encoded.csv'
model_name = "Logistic Regression"

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configura MLflow para usar un directorio local para el seguimiento
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
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        end_time = time.time()
        logging.info(f"Distribución de clases después de SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de SMOTE: {execution_time:.2f} segundos")
        return X_resampled, y_resampled
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

def evaluate_thresholds(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append((threshold, precision, recall, f1))
        print(f"Threshold: {threshold:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    return results

def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('reports/figures/precision_recall_curve.png')
    plt.close()
    logging.info("Gráfico de curva Precision-Recall guardado.")

def cross_validate_roc_auc(X, y, model, cv=5):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        # Cambia a RocCurveDisplay.from_estimator
        viz = RocCurveDisplay.from_estimator(model, X[test], y[test], ax=ax, name=f"ROC fold {i}")
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.savefig('reports/figures/cross_validated_roc_curve.png')
    plt.close()
    logging.info("Gráfico de curva ROC con validación cruzada guardado.")
    return mean_auc, std_auc


def train_and_evaluate_model(X_train, X_test, y_train, y_test, class_weight_dict):
    try:
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs'],
            'class_weight': [class_weight_dict, 'balanced']
        }

        base_model = LogisticRegression(random_state=42)
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

def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


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
        # Verificar si el modelo tiene el atributo 'feature_importances_' (como Random Forest)
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        # Verificar si el modelo tiene el atributo 'coef_' (como Regresión Logística)
        elif hasattr(model, 'coef_'):
            # Tomamos el valor absoluto de los coeficientes para evaluar la importancia
            importances = pd.DataFrame({'feature': feature_names, 'importance': np.abs(model.coef_[0])})
        else:
            logging.warning(f"El modelo {type(model).__name__} no soporta la obtención de importancia de características.")
            return
        
        # Ordenar las importancias y seleccionar las 10 características más importantes
        importances = importances.sort_values('importance', ascending=False).head(10)

        # Crear el gráfico
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title('Top 10 Características Más Importantes')
        plt.tight_layout()

        # Guardar el gráfico
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

def generate_report(model, feature_names, X_test, y_test, report_path):
    try:
        # Predecir valores
        y_pred = model.predict(X_test)
        
        # Generar el reporte de clasificación
        classification_rep = classification_report(y_test, y_pred)
        logging.info("Reporte de clasificación generado.")
        
        # Guardar el reporte de clasificación en un archivo
        with open(report_path + '/classification_report.txt', 'w') as f:
            f.write(classification_rep)
        
        logging.info("Reporte de clasificación guardado en archivo.")

        # Importancia de características según el tipo de modelo
        if hasattr(model, 'feature_importances_'):
            # Modelos como Random Forest
            importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        elif hasattr(model, 'coef_'):
            # Modelos como Regresión Logística
            importances = pd.DataFrame({'feature': feature_names, 'importance': np.abs(model.coef_[0])})
        else:
            logging.warning(f"El modelo {type(model).__name__} no soporta la obtención de importancia de características.")
            return

        # Ordenar las importancias y guardar en un archivo CSV
        importances = importances.sort_values('importance', ascending=False)
        importances.to_csv(report_path + '/feature_importances.csv', index=False)
        
        logging.info("Importancia de características guardada en archivo CSV.")
        
    except Exception as e:
        logging.error(f"Error al generar el reporte: {str(e)}")
        raise


if __name__ == "__main__":
    # Cargar y preprocesar los datos
    X_scaled, y, feature_names, scaler, class_weight_dict = load_and_preprocess_data(file_path)

    # Dividir los datos en entrenamiento y prueba antes de aplicar SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Aplicar SMOTE solo a los datos de entrenamiento
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Entrenar y evaluar el modelo de Regresión Logística
    best_model, metrics, y_pred_adjusted, y_pred_proba, best_params, best_threshold = train_and_evaluate_model(X_train_resampled, X_test, y_train_resampled, y_test, class_weight_dict)

    # Evaluar umbrales
    threshold_results = evaluate_thresholds(y_test, y_pred_proba)

    # Graficar la curva Precision-Recall
    plot_precision_recall_curve(y_test, y_pred_proba)

    # Realizar validación cruzada para ROC y AUC
    mean_auc, std_auc = cross_validate_roc_auc(X_scaled, y, best_model)

    # Realizar la validación cruzada
    cv_scores = cross_validation_evaluate_model(X_train_resampled, y_train_resampled, best_model)

    # Detectar overfitting
    overfitting_metrics = detect_overfitting(best_model, X_train_resampled, y_train_resampled, X_test, y_test)

    # Guardar modelo y escalador
    model_dir = f'models/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, f'{model_dir}/scaler_{model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump((best_model, best_threshold), f'{model_dir}/model_{model_name.lower().replace(" ", "_")}.pkl')
    print(f"Modelo, umbral óptimo y escalador guardados en '{model_dir}'.")

    # Registrar los resultados en MLflow
    log_mlflow(best_model, metrics, model_name, best_params)

    # Graficar la matriz de confusión
    plot_confusion_matrix(y_test, y_pred_adjusted, model_name)

    # Graficar la curva ROC
    plot_roc_curve(y_test, y_pred_proba)

    # Generar un informe detallado
    generate_report(
    best_model, 
    feature_names, 
    X_test, 
    y_test, 
    'reports/'
)
