import os
import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from components.logger_code import setup_logger

import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

logger = setup_logger('data_preprocessing', 'data_preprocessing.log')

def transform_text(text):
    """
    Transform the text by removing punctuation, converting to lowercase, and stemming
    """
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

def preprocess_df(df, text_col='text', target_col='target'):
    """
    Preprocess the DataFrame by encoding the target column, removing duplicates, and transforming the text by applying the transform_text function to the text column and encoding the target column
    """
    try:
        logger.debug('Starting preprocessing of DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        logger.debug('Target column encoded successfully')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first') # keep the first occurrence of duplicate rows
        logger.debug('Duplicate rows removed successfully')

        # Apply the transform_text function to the specified text column
        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logger.debug('Text column transformed successfully')
        return df
    except KeyError as e:
        logger.error(f'Missing column in DataFrame: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def main(text_col='text', target_col='target'):
    """
    Main function to load, preprocess, and save the data
    """
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        processed_train_data = preprocess_df(train_data, text_col, target_col)
        processed_test_data = preprocess_df(test_data, text_col, target_col)
        
        # Save the preprocessed data
        pp_data_path = os.path.join('./data', "preprocessed")
        os.makedirs(pp_data_path, exist_ok=True)

        processed_train_data.to_csv(os.path.join(pp_data_path, 'train_processed.csv'), index=False)
        processed_test_data.to_csv(os.path.join(pp_data_path, 'test_processed.csv'), index=False)
        logger.debug(f'Preprocessed data saved successfully to {pp_data_path}')
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except pd.errors.EmptyDataError as e:
        logger.error(f'No data: {e}')
    except Exception as e:
        logger.error(f'Error in main function: {e}')

if __name__ == '__main__':
    main()