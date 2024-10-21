import logging
import time
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, precision_recall_curve, auc, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Variables
file_path = 'data/processed/stroke_dataset_encoded.csv'
model_name = "Ensemble Model"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure MLflow to use a local directory for tracking
mlflow.set_tracking_uri("file:///C:/Users/avkav/Documents/BootcampAI/ProyectoDataScientist-Grupo4/DATA-SCIENTIST-4/mlruns")

def load_and_preprocess_data(file_path):
    try:
        logging.info(f"Loading and preprocessing the dataset from {file_path}")
        data = pd.read_csv(file_path)
        X = data.drop(columns=['stroke'])
        y = data['stroke']
        
        # Apply standard scaling to features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        logging.info("Dataset loaded, processed, and scaled")
        return X_scaled, y, scaler
    except Exception as e:
        logging.error(f"Error in data loading and preprocessing: {str(e)}")
        raise

def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }

def apply_smote(X, y):
    try:
        start_time = time.time()
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        end_time = time.time()
        logging.info(f"Class distribution after SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
        execution_time = end_time - start_time
        logging.info(f"SMOTE execution time: {execution_time:.2f} seconds")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error applying SMOTE: {str(e)}")
        raise

def train_ensemble_model(X_train_resampled, X_test, y_train_resampled, y_test, class_weight_dict):
    try:
        # Define base classifiers
        rf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
        xgb = XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[1])
        gb = GradientBoostingClassifier(random_state=42)
        lr = LogisticRegression(random_state=42, class_weight=class_weight_dict)

        # Assemble classifiers using VotingClassifier
        ensemble_model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('gb', gb), ('lr', lr)],
            voting='soft'
        )

        # Define hyperparameter search
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [10, 20, None],
            'xgb__learning_rate': [0.01, 0.1, 0.2],
            'xgb__max_depth': [3, 6, 10],
            'gb__n_estimators': [100, 200],
            'gb__learning_rate': [0.01, 0.1],
            'lr__C': [0.1, 1, 10]
        }

        grid_search = GridSearchCV(ensemble_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)

        best_model = grid_search.best_estimator_

        # Predictions on test set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        metrics = calculate_metrics(y_test, y_pred_adjusted, y_pred_proba)
        best_params = grid_search.best_params_

        return best_model, metrics, y_pred_adjusted, y_pred_proba, best_params
    except Exception as e:
        logging.error(f"Error in training and evaluating the ensemble model: {str(e)}")
        raise

def detect_overfitting(model, X_train, y_train, X_test, y_test):
    """Calculate metrics on training and test sets to detect overfitting."""
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        logging.info(f"Training set accuracy: {train_accuracy:.4f}")
        logging.info(f"Test set accuracy: {test_accuracy:.4f}")
        
        overfitting_score = train_accuracy - test_accuracy
        if overfitting_score > 0.1:
            logging.warning(f"Possible overfitting detected: accuracy difference of {overfitting_score:.4f} between training and test sets.")
        else:
            logging.info("No significant overfitting detected.")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_score': overfitting_score
        }
    except Exception as e:
        logging.error(f"Error detecting overfitting: {str(e)}")
        raise

def log_mlflow(model, metrics, model_name, params, input_example=None):
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

def plot_confusion_matrix(y_test, y_pred, model_name):
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        output_path = f'reports/figures/confusion_matrix_{model_name}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Confusion matrix plot saved for model {model_name}.")
    except Exception as e:
        logging.error(f"Error generating confusion matrix: {str(e)}")
        raise   

def plot_roc_curve(y_test, y_pred_proba, model_name):
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(f'reports/figures/roc_curve_{model_name}.png')
        plt.close()
        logging.info(f"ROC curve plot saved for model {model_name}.")
    except Exception as e:
        logging.error(f"Error generating ROC curve: {str(e)}")
        raise

def save_scaled_data(X, y, feature_names, output_file):
    df = pd.DataFrame(X, columns=feature_names)
    df['stroke'] = y
    df.to_csv(output_file, index=False)
    logging.info(f"Scaled data saved to {output_file}")

def evaluate_thresholds(y_true, y_pred_proba):
    thresholds = np.arange(0, 1.01, 0.01)
    scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        scores.append({
            'threshold': threshold,
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        })
    return pd.DataFrame(scores)

def plot_precision_recall_curve(y_true, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(f'reports/figures/precision_recall_curve_{model_name}.png')
    plt.close()
    logging.info(f"Precision-Recall curve plot saved for model {model_name}.")

def cross_validate_roc_auc(X, y, model):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return np.mean(cv_scores), np.std(cv_scores)

def generate_report(
    model, X_train_resampled, X_test, y_train_resampled, y_test,
    X_scaled, y, feature_names, metrics, best_params, class_weight_dict,
    threshold, cv_scores, overfitting_metrics,
    threshold_results, mean_auc, std_auc
):
    report = f"""
    Model Performance Report
    ========================
    
    Model: {type(model).__name__}
    
    Best Parameters:
    {best_params}
    
    Class Weights:
    {class_weight_dict}
    
    Metrics:
    {metrics}
    
    Cross-validation ROC AUC:
    Mean: {mean_auc:.3f} (+/- {std_auc * 2:.3f})
    
    Threshold Analysis:
    Best F1 Score: {threshold_results['f1'].max():.3f} at threshold {threshold_results.loc[threshold_results['f1'].idxmax(), 'threshold']:.2f}
    
    Feature Importance:
    """

    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        report += str(feature_importance)
    elif hasattr(model, 'estimators_'):
        # For VotingClassifier, calculate average importance of estimators that have feature_importances_
        importances = []
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances.append(pd.DataFrame({'feature': feature_names, 'importance': estimator.feature_importances_, 'estimator': name}))
        
        if importances:
            feature_importance = pd.concat(importances)
            feature_importance = feature_importance.groupby('feature')['importance'].mean().reset_index()
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            report += str(feature_importance)
        else:
            report += "Feature importance not available for this model."
    else:
        report += "Feature importance not available for this model."

    report += f"""
    
    Classification Report:
    {classification_report(y_test, (model.predict_proba(X_test)[:, 1] >= threshold).astype(int))}
    """
    
    with open('reports/vc_model_performance_report.txt', 'w') as f:
        f.write(report)
    logging.info("Model performance report generated and saved.")

if __name__ == "__main__":
    try:
        # Load and preprocess data
        X, y, scaler = load_and_preprocess_data(file_path)

        # Split data into training and test sets before applying SMOTE
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply SMOTE
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

        # Calculate class weight
        class_weight_dict = {0: (len(y_train) - sum(y_train)) / len(y_train), 1: sum(y_train) / len(y_train)}

        # Train the model
        best_ensemble_model, metrics, y_pred_adjusted, y_pred_proba, best_params = train_ensemble_model(
            X_train_resampled, X_test, y_train_resampled, y_test, class_weight_dict
        )

        # Get an input example
        input_example = X_test.iloc[0].to_frame().T

        # Save scaled data
        save_scaled_data(X_train_resampled, y_train_resampled, X.columns, 'data/processed/stroke_dataset_scaled.csv')

        # Log metrics in MLflow
        log_mlflow(best_ensemble_model, metrics, model_name, best_params, input_example)

        # Detect overfitting
        overfitting_metrics = detect_overfitting(best_ensemble_model, X_train_resampled, y_train_resampled, X_test, y_test)

        # Save model and scaler
        model_dir = f'models/{model_name.lower().replace(" ", "_")}'
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, f'{model_dir}/scaler_{model_name.lower().replace(" ", "_")}.pkl')
        joblib.dump(best_ensemble_model, f'{model_dir}/model_{model_name.lower().replace(" ", "_")}.pkl')
        logging.info(f"Model and scaler saved in '{model_dir}'.")

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred_adjusted, model_name)

        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, model_name)

        # Evaluate thresholds and plot curves
        threshold_results = evaluate_thresholds(y_test, y_pred_proba)
        plot_precision_recall_curve(y_test, y_pred_proba, model_name)
        mean_auc, std_auc = cross_validate_roc_auc(X, y, best_ensemble_model)

        # Generate final report
        generate_report(
            best_ensemble_model,
            X_train_resampled, X_test, y_train_resampled, y_test,
            X, y, X.columns,
            metrics, best_params, class_weight_dict,
            0.5, cv_scores=None, overfitting_metrics=overfitting_metrics,
            threshold_results=threshold_results, mean_auc=mean_auc, std_auc=std_auc
        )

        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the execution: {str(e)}")
        raise