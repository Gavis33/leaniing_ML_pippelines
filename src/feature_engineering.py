import os
import yaml
import pandas as pd
from components.logger_code import setup_logger
from sklearn.feature_extraction.text import TfidfVectorizer

logger = setup_logger('feature_engineering', 'feature_engineering.log')

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

def load_data(file_path: str) ->pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df= pd.read_csv(file_path)
        df.fillna('', inplace=True) # replace missing values with empty strings
        logger.debug(f'Loaded data and NaNs filled from: {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.debug(f'Error parsing CSV file: {file_path}, Error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading data: {e}')
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features:int) -> tuple:
    """Apply TF-IDF vectorization to the text data in the train and test datasets."""
    try:
        verctorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values

        X_train_tfidf = verctorizer.fit_transform(X_train)
        X_test_tfidf = verctorizer.fit_transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('TF-IDF vectorization applied to train and test data.')
        return train_df, test_df
    except Exception as e:
        logger.error(f'Error applying TF-IDF: {e}')
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f'Data saved to: {file_path}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # max_features = params['feature_engineering']['max_features']
        max_features = 50    

        train_data = load_data('./data/preprocessed/train_processed.csv')
        test_data = load_data('./data/preprocessed/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join('./data', 'feature_engineered', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('./data', 'feature_engineered', 'test_tfidf.csv'))
    except Exception as e:
        logger.error(f'Error in main function: {e}')
        raise

if __name__ == '__main__':
    main()