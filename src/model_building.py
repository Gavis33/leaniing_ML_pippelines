import os
import yaml
import pickle
import numpy as np
import pandas as pd
from components.logger_code import setup_logger
from sklearn.ensemble import RandomForestClassifier

logger = setup_logger('model_building', 'model_building.log')

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug(f'Parameters loaded from {params_path}')
#         return params
#     except FileNotFoundError:
#         logger.error(f'Parameters file not found: {params_path}')
#         raise
#     except yaml.YAMLError as e:
#         logger.error(f'Error parsing YAML file {params_path}: {e}')
#         raise
#     except Exception as e:
#         logger.error(f'Error loading parameters from {params_path}: {e}')
#         raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Error parsing CSV file {file_path}: {e}')
        raise
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise

# def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        params (dict): Dictionary of hyperparameters.
    Returns:
        RandomForestClassifier: Trained Random Forest classifier.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('Number of samples in X_train and y_train must match.')
        # logger.debug(f'Initializing RandomForestClassifier with params: {params}')
        logger.debug(f'Initializing RandomForestClassifier with params:')
        # model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        model = RandomForestClassifier()
        logger.debug(f'Model training started with {X_train.shape[0]} samples.')
        model.fit(X_train, y_train)
        logger.debug('Model training completed.')
        return model
    except ValueError as e:
        logger.error(f'ValueError during model training: {e}')
        raise
    except Exception as e:
        logger.error(f'Error during model training: {e}')
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    Args:
        model (object): Trained model object.
        file_path (str): Path to save the model file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f'Model saved to {file_path}')
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}') 
        raise   
    except Exception as e:
        logger.error(f'Error saving model: {e}')
        raise

def main():
    try:
        # params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/feature_engineered/train_tfidf.csv')
        test_data = load_data('./data/feature_engineered/test_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values # Extract all columns except the last (features) last column is the label column for train data
        y_train = train_data.iloc[:, -1].values

        # model = train_model(X_train, y_train, params)
        model = train_model(X_train, y_train)
        model_save_path = './models/model.pkl'
        save_model(model, model_save_path)

    except Exception as e:
        logger.error(f'Error in main function: {e}')
        raise

if __name__ == '__main__':
    main()