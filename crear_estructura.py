import os
import venv  # Módulo para crear entornos virtuales

# Función para crear directorios si no existen
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Función para crear un archivo vacío si no existe
def create_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as file:
            pass

def create_project_structure(base_path):
    # Crear estructura de carpetas
    create_directory(os.path.join(base_path, 'data/raw'))
    create_directory(os.path.join(base_path, 'data/processed'))
    create_directory(os.path.join(base_path, 'data/particion'))
    
    create_directory(os.path.join(base_path, 'api'))
    
    create_directory(os.path.join(base_path, 'notebooks'))
    
    
    create_directory(os.path.join(base_path, 'outputs/csv'))
    create_directory(os.path.join(base_path, 'outputs/graphs/otros_graficos'))
    
    create_directory(os.path.join(base_path, 'models'))
    
    create_directory(os.path.join(base_path, 'src'))
    
    create_directory(os.path.join(base_path, 'tests'))
    
    create_directory(os.path.join(base_path, 'config'))
    
    # Crear archivos dentro de las carpetas correspondientes
    create_file(os.path.join(base_path, 'data/raw/train.csv'))
    create_file(os.path.join(base_path, 'data/raw/test.csv'))
    create_file(os.path.join(base_path, 'data/processed/train_processed.csv'))
    create_file(os.path.join(base_path, 'data/processed/test_processed.csv'))
    
    
    create_file(os.path.join(base_path, 'api/app.py'))  # Archivo principal de la API
    create_file(os.path.join(base_path, 'api/model.py'))  # Lógica del modelo
    create_file(os.path.join(base_path, 'api/requirements.txt'))  # Dependencias de la API
    
    create_file(os.path.join(base_path, 'notebooks/eda.ipynb'))
    create_file(os.path.join(base_path, 'notebooks/modelado.ipynb'))
    
    create_file(os.path.join(base_path, 'outputs/csv/predictions.csv'))
    create_file(os.path.join(base_path, 'outputs/csv/metrics.csv'))
    create_file(os.path.join(base_path, 'outputs/graphs/corr_matrix.png'))
    create_file(os.path.join(base_path, 'outputs/graphs/feature_importance.png'))
    
    create_file(os.path.join(base_path, 'src/data_preprocessing.py')) # Preprocesamiento de datos (limpieza, imputación, etc.)
    create_file(os.path.join(base_path, 'src/model_training.py')) # Entrenamiento del modelo
    create_file(os.path.join(base_path, 'src/evaluate.py')) # Evaluación del modelo (métricas, curvas ROC, etc.)
    create_file(os.path.join(base_path, 'src/predict.py')) # Código para predecir con el modelo entrenado
    
    create_file(os.path.join(base_path, 'models/modelo_entrenado.pkl')) # Modelo entrenado
    
    create_file(os.path.join(base_path, 'config/config.yaml'))
    
    # Crear archivos README.md, requirements.txt y .gitignore en la raíz del proyecto
    create_file(os.path.join(base_path, 'README.md'))
    create_file(os.path.join(base_path, 'requirements.txt'))
    create_file(os.path.join(base_path, '.gitignore'))
    
    # Crear el entorno virtual
    #venv.create(os.path.join(base_path, 'venv'), with_pip=True)

    print("Estructura de proyecto creada con éxito en:", base_path)

# Ejecutar la función para crear la estructura
if __name__ == "__main__":
    # Define la ruta base del proyecto
    project_base_path = './DATA-SCIENTIST-4/'  # Cambia esto si quieres una ruta diferente
    create_project_structure(project_base_path)
#C:\4_F5\017_track_01\DATA-SCIENTIST-4