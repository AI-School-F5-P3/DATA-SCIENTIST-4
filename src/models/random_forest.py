import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Cargar los datos
data = pd.read_csv('data/raw/stroke_dataset.csv')

# Análisis Exploratorio de Datos (EDA)
def eda(data):
    print(data.info())
    print("\nEstadísticas descriptivas:")
    print(data.describe())
    
    # Visualizaciones
    plt.figure(figsize=(12, 6))
    sns.countplot(x='stroke', data=data)
    plt.title('Distribución de Ictus')
    plt.show()
    
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='stroke', y=col, data=data)
        plt.title(f'Distribución de {col} por Ictus')
        plt.show()

# Preprocesamiento de datos
def preprocess_data(data):
    # Convertir variables categóricas a numéricas
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    data['ever_married'] = data['ever_married'].map({'No': 0, 'Yes': 1})
    data['work_type'] = pd.get_dummies(data['work_type'], drop_first=True)
    data['Residence_type'] = data['Residence_type'].map({'Rural': 0, 'Urban': 1})
    data['smoking_status'] = pd.get_dummies(data['smoking_status'], drop_first=True)
    
    # Separar características y variable objetivo
    X = data.drop(['stroke', 'id'], axis=1)
    y = data['stroke']
    
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Escalar las características
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# Entrenamiento del modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

# Evaluación del modelo
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Accuracy en entrenamiento: {train_accuracy:.4f}")
    print(f"Accuracy en prueba: {test_accuracy:.4f}")
    print(f"Overfitting: {abs(train_accuracy - test_accuracy):.4f}")
    
    print("\nMétricas en conjunto de prueba:")
    print(f"Precisión: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_test_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

# Importancia de características
def feature_importance(model, X):
    importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances.head(10))
    plt.title('Top 10 Características Más Importantes')
    plt.show()

# Nueva función para generar el informe detallado
def generate_detailed_report(model, X_train, X_test, y_train, y_test, X, y):
    report = "Informe Detallado del Modelo de Predicción de Ictus\n"
    report += "=" * 50 + "\n\n"

    # 1. Resumen del conjunto de datos
    report += "1. Resumen del Conjunto de Datos\n"
    report += "-" * 30 + "\n"
    report += f"Total de muestras: {len(X)}\n"
    report += f"Características: {X.shape[1]}\n"
    report += f"Distribución de clases: {dict(y.value_counts())}\n\n"

    # 2. Rendimiento del modelo
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    report += "2. Rendimiento del Modelo\n"
    report += "-" * 30 + "\n"
    report += "Métricas en conjunto de entrenamiento:\n"
    report += f"  Accuracy: {accuracy_score(y_train, y_train_pred):.4f}\n"
    report += f"  Precisión: {precision_score(y_train, y_train_pred):.4f}\n"
    report += f"  Recall: {recall_score(y_train, y_train_pred):.4f}\n"
    report += f"  F1-score: {f1_score(y_train, y_train_pred):.4f}\n"
    
    report += "\nMétricas en conjunto de prueba:\n"
    report += f"  Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n"
    report += f"  Precisión: {precision_score(y_test, y_test_pred):.4f}\n"
    report += f"  Recall: {recall_score(y_test, y_test_pred):.4f}\n"
    report += f"  F1-score: {f1_score(y_test, y_test_pred):.4f}\n"
    report += f"  AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}\n\n"

    # 3. Análisis de overfitting
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    overfitting = abs(train_accuracy - test_accuracy)
    
    report += "3. Análisis de Overfitting\n"
    report += "-" * 30 + "\n"
    report += f"Diferencia entre accuracy de entrenamiento y prueba: {overfitting:.4f}\n"
    if overfitting < 0.05:
        report += "El modelo no muestra signos significativos de overfitting.\n\n"
    else:
        report += "El modelo muestra signos de overfitting. Se recomienda ajustar los hiperparámetros o usar técnicas de regularización.\n\n"

    # 4. Validación cruzada
    cv_scores = cross_val_score(model, X, y, cv=5)
    report += "4. Validación Cruzada\n"
    report += "-" * 30 + "\n"
    report += f"Scores de validación cruzada: {cv_scores}\n"
    report += f"Media de accuracy en validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n"

    # 5. Importancia de características
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    report += "5. Importancia de Características\n"
    report += "-" * 30 + "\n"
    report += "Top 10 características más importantes:\n"
    for i, row in feature_importance.head(10).iterrows():
        report += f"  {i+1}. {row['feature']}: {row['importance']:.4f}\n"
    report += "\n"

    # 6. Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    report += "6. Matriz de Confusión\n"
    report += "-" * 30 + "\n"
    report += str(cm) + "\n\n"

    # 7. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    
    # Visualizaciones
    plt.figure(figsize=(15, 5))
    
    # Importancia de características
    plt.subplot(1, 2, 1)
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Características Más Importantes')
    plt.tight_layout()
    
    # Curva ROC
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('model_performance_visualizations.png')
    plt.close()

    report += "7. Visualizaciones\n"
    report += "-" * 30 + "\n"
    report += "Se han generado visualizaciones de la importancia de características y la curva ROC. "
    report += "Consulte el archivo 'model_performance_visualizations.png'.\n\n"

    # 8. Conclusiones y recomendaciones
    report += "8. Conclusiones y Recomendaciones\n"
    report += "-" * 30 + "\n"
    report += "- El modelo muestra un rendimiento [bueno/moderado/bajo] en la predicción de riesgo de ictus.\n"
    report += f"- La precisión en el conjunto de prueba es de {accuracy_score(y_test, y_test_pred):.2f}, "
    report += f"lo que indica que el modelo acierta en el {accuracy_score(y_test, y_test_pred)*100:.1f}% de los casos.\n"
    report += f"- El recall de {recall_score(y_test, y_test_pred):.2f} sugiere que el modelo identifica correctamente "
    report += f"el {recall_score(y_test, y_test_pred)*100:.1f}% de los casos positivos de ictus.\n"
    report += "- Las características más importantes para la predicción son [liste las top 3].\n"
    report += "- Recomendaciones para mejorar el modelo:\n"
    report += "  1. Recopilar más datos, especialmente para la clase minoritaria (si existe desbalance).\n"
    report += "  2. Experimentar con diferentes algoritmos de clasificación.\n"
    report += "  3. Realizar una selección de características más exhaustiva.\n"
    report += "  4. Ajustar los hiperparámetros del modelo mediante una búsqueda en cuadrícula o aleatoria.\n"

    return report

# Función main para incluir la generación del informe
def main():
    data = pd.read_csv('stroke_dataset.csv')
    eda(data)
    X, y = preprocess_data(data)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    evaluate_model(model, X_train, X_test, y_train, y_test)
    feature_importance(model, X)
    
    # Generar y guardar el informe detallado
    detailed_report = generate_detailed_report(model, X_train, X_test, y_train, y_test, X, y)
    with open('reports/informe_detallado_modelo_ictus.txt', 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    print("Se ha generado el informe detallado. Consulte el archivo 'informe_detallado_modelo_ictus.txt'.")

if __name__ == "__main__":
    main()