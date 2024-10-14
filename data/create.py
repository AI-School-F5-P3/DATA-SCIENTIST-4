import os

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def create_file(path):
    with open(path, 'w') as f:
        pass  # Create an empty file

def create_project_structure():
    # Create main project directory
    project_name = "ONYVA"
    create_directory(project_name)

    # Create subdirectories
    directories = [
        "data/raw",
        "data/processed",
        "models/xgboost",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "reports/figures",       # Gráficos y visualizaciones
        "tests"      
    ]

    for directory in directories:
        create_directory(os.path.join(project_name, directory))

    # Create Python files
    python_files = [
        "src/data/__init__.py",
        "src/data/load_nasa_data.py",
        "src/data/preprocess_climate_data.py",
        "src/features/__init__.py",
        "src/features/build_climate_features.py",
        "src/models/__init__.py",
        "src/models/train_xgboost_model.py",
        "src/models/predict_climate_conditions.py",
        "src/visualization/__init__.py",
        "src/visualization/visualize_climate_data.py",
        "tests/test_data_loading.py",
        "tests/test_xgboost_model.py"
    ]

    for file in python_files:
        create_file(os.path.join(project_name, file))

    # Create notebook files
    notebook_files = [
        "notebooks/exploratory_climate_analysis.ipynb",
        "notebooks/xgboost_model_development.ipynb"
    ]

    for file in notebook_files:
        create_file(os.path.join(project_name, file))

    # Create other files
    other_files = [
        ".gitignore",
        "requirements.txt",
        "setup.py",
        "README.md",
        "config.yaml"  # For storing model parameters and data paths
    ]

    for file in other_files:
        create_file(os.path.join(project_name, file))

    print(f"Project structure for '{project_name}' has been created.")

if __name__ == "__main__":
    create_project_structure()