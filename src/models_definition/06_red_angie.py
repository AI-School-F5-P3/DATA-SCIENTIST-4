import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline



# Crear directorio para guardar los modelos si no existe
model_directory = 'saved_model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cargar el dataset
data = pd.read_csv("dataset_escalado_copiaeda.csv")

# Guardar los nombres de las columnas para uso futuro
feature_columns = list(data.drop('stroke', axis=1).columns)
joblib.dump(feature_columns, os.path.join(model_directory, 'feature_columns.joblib'))

# Separar características (X) y la variable objetivo (y)
X = data.drop('stroke', axis=1)
y = data['stroke']

# Primer split: separar el conjunto de "nuevos pacientes" (20%)
X_temp, X_nuevos, y_temp, y_nuevos = train_test_split(
    X, y, 
    test_size=0.2,  # 20% para nuevos pacientes
    random_state=42,
    stratify=y  # Mantener la proporción de clases
)

# Segundo split: dividir los datos restantes en train (87.5%) y test (12.5%)
# Esto resulta en la división final aproximada de 70% train, 10% test, 20% nuevos
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.125,  # 0.125 * 0.8 = 0.1 (10% del total)
    random_state=42,
    stratify=y_temp
)

# Verificar los tamaños de los conjuntos
print("\nTamaño de los conjuntos de datos:")
print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Nuevos: {len(X_nuevos)} ({len(X_nuevos)/len(X)*100:.1f}%)")

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_nuevos_scaled = scaler.transform(X_nuevos)

# Guardar el scaler
joblib.dump(scaler, os.path.join(model_directory, 'scaler.joblib'))


# Crear y entrenar el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 1. Usando class_weights
# Calcula los pesos de clase
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convierte los pesos a un diccionario
class_weights_dict = dict(zip(np.unique(y_train), class_weights))


# Asegurarse de que y_train sea un array de numpy, en lugar de una Series
y_train_array = np.array(y_train)

# Verifica que las clases en class_weights_dict coincidan con y_train
print(f"Clases en y_train: {np.unique(y_train_array)}")
print(f"Clases en class_weights_dict: {class_weights_dict.keys()}")


# 2. Usando SMOTE (Synthetic Minority Over-sampling Technique)
# Primero el oversampling con SMOTE, luego un poco de undersampling
oversample = SMOTE(sampling_strategy=0.5)
undersample = RandomUnderSampler(sampling_strategy=0.8)

pipeline = Pipeline([
    ('SMOTE', oversample),
    ('RandomUnderSampler', undersample)
])

# Aplicar el balanceo
X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train_scaled, y_train_array)

# Entrenar con datos balanceados


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Ahora usa el diccionario en el model.fit
history = model.fit(
    X_train_balanced, y_train_balanced,
    epochs=50,
    batch_size=32,
    class_weight=class_weights_dict,
    validation_split=0.2,
    verbose=2
)




""" 
# 2. Usando SMOTE (Synthetic Minority Over-sampling Technique)
# Primero el oversampling con SMOTE, luego un poco de undersampling
oversample = SMOTE(sampling_strategy=0.5)
undersample = RandomUnderSampler(sampling_strategy=0.8)

pipeline = Pipeline([
    ('SMOTE', oversample),
    ('RandomUnderSampler', undersample)
])

# Aplicar el balanceo
X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train_scaled, y_train)

# Entrenar con datos balanceados
model.fit(
    X_train_balanced, y_train_balanced,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
"""
y_test_array = np.array(y_test)


# Evaluar el modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_array)
print(f"\nPrecisión en el conjunto de test: {test_accuracy:.4f}")

# Guardar el modelo
model_path=os.path.join(model_directory, 'stroke_model.keras')
model.save(model_path)
print(f"\nModelo guardado en: {model_path}")



# Función para evaluar el modelo en diferentes conjuntos de datos
def evaluar_conjunto(X_scaled, y_true, nombre_conjunto):
    """
    Evalúa el modelo en un conjunto de datos específico.
    """
    predicciones = model.predict(X_scaled)
    predicciones_binarias = (predicciones > 0.5).astype(int)
    
    print(f"\nResultados para el conjunto {nombre_conjunto}:")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_true, predicciones_binarias))
    print("\nReporte de clasificación:")
    print(classification_report(y_true, predicciones_binarias))


# Evaluar en todos los conjuntos
evaluar_conjunto(X_train_scaled, y_train, "TRAIN")
evaluar_conjunto(X_test_scaled, y_test, "TEST")
evaluar_conjunto(X_nuevos_scaled, y_nuevos, "NUEVOS PACIENTES")




# Función para probar un paciente nuevo
def probar_paciente_nuevo(indice):
    """
    Prueba el modelo con un paciente del conjunto de nuevos pacientes.
    """
    # Obtener los datos del paciente
    datos_paciente = X_nuevos.iloc[indice]
    datos_escalados = scaler.transform(datos_paciente.values.reshape(1, -1))
    
    # Realizar predicción
    probabilidad = model.predict(datos_escalados, verbose=0)[0][0]
    prediccion = 1 if probabilidad > 0.5 else 0
    valor_real = y_nuevos.iloc[indice]
    
    print("\nDatos del paciente nuevo:")
    for columna, valor in datos_paciente.items():
        print(f"{columna}: {valor}")
    
    print(f"\nPredicción (probabilidad): {probabilidad:.4f}")
    print(f"Predicción (binaria): {prediccion}")
    print(f"Valor real: {valor_real}")
    print(f"¿Predicción correcta?: {'Sí' if prediccion == valor_real else 'No'}")

# Guardar algunos ejemplos del conjunto de nuevos pacientes para uso futuro
ejemplos_nuevos = pd.DataFrame({
    'datos': [X_nuevos.iloc[i].to_dict() for i in range(min(5, len(X_nuevos)))],
    'stroke_real': y_nuevos.iloc[:5].tolist()
})
joblib.dump(ejemplos_nuevos, os.path.join(model_directory, 'ejemplos_nuevos.joblib'))

# Ejemplo de uso
print("\nProbando con algunos pacientes nuevos:")
for i in range(3):  # Probar con 3 pacientes nuevos
    print(f"\nPaciente nuevo #{i+1}")
    probar_paciente_nuevo(i)

 
 