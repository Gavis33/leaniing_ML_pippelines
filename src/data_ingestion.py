import os
import yaml
import pandas as pd
from components.logger_code import setup_logger
from sklearn.model_selection import train_test_split

# Setup logger
logger = setup_logger('data_ingestion', 'data_ingestion.log')

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters loaded successfully from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'Parameters file not found: {params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'Error parsing YAML file: {params_path}, Error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading parameters: {e}')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded successfully from {data_url}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Error parsing CSV file: {data_url}, Error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by renaming columns and dropping unnecessary ones"""
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug(f'Data preprocessing completed successfully for {df.shape[0]} rows')
        return df
    except KeyError as e:
        logger.error(f'Missing column in dataframe: {e}')    
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test datasets to CSV files"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f'Train and test datasets saved successfully to {data_path}')
    except Exception as e:
        logger.error(f'Error saving datasets: {e}')
        raise   

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.20
        data_path = 'experiments/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(f'Error in main function: {e}')
        raise

if __name__ == '__main__':
    main()

# This code is designed to be run as a script. It will not execute if imported as a module.