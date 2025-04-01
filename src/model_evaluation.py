import os
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from dvclive import Live
from components.logger_code import setup_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = setup_logger('model_evaluation', 'model_evaluation.log')

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug(f'Loaded parameters from: {params_path}')
#         return params
#     except FileNotFoundError:
#         logger.error(f'Parameters file not found: {params_path}')
#         raise
#     except yaml.YAMLError as e:
#         logger.error(f'Error parsing YAML file: {params_path}, Error: {e}')
#         raise
#     except Exception as e:
#         logger.error(f'Unexpected error loading parameters: {e}')
#         raise

def load_model(file_path: str):
    """Load a model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f'Loaded model from: {file_path}')
        return model
    except FileNotFoundError:
        logger.error(f'Model file not found: {file_path}')
        raise   
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Loaded data from: {file_path}')
        return df
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise   
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a model on a test set."""
    try:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] # probability of positive class for binary classification

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logger.debug(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}')
        return metrics_dict
    except Exception as e:
        logger.error(f'Error evaluating model: {e}')
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f'Metrics saved to: {file_path}')
    except Exception as e:
        logger.error(f'Error saving metrics: {e}')
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        model = load_model('./models/model.pkl')
        test_data = load_data('./data/feature_engineered/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values 
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        # Experiment tracking with dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            # live.log_params(params)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error(f'Error in main function: {e}')
        raise   

if __name__ == '__main__':
    main()