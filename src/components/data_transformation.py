import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation."""

    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """Handles text vectorization using TF-IDF."""

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates and returns the TfidfVectorizer object.
        """
        try:
            tfidf = TfidfVectorizer(
                stop_words='english', 
                max_features=5000,
                ngram_range=(1, 2),  # Include unigrams and bigrams
                min_df=2,            # Ignore terms that appear in less than 2 documents
                max_df=0.95          # Ignore terms that appear in more than 95% of documents
            )
            return tfidf
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Applies TF-IDF transformation to the dataset."""

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "label"
            
            # Use the cleaned full_text column (created during cleaning)
            # This combines title and text that were already cleaned
            text_column = 'full_text' if 'full_text' in train_df.columns else 'text_cleaned'
            
            if text_column not in train_df.columns:
                raise Exception(f"Required text column '{text_column}' not found. Ensure data cleaning was performed.")
            
            # Extract features and targets
            input_feature_train_df = train_df[text_column].fillna('')
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df[text_column].fillna('')
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying TfidfVectorizer on {text_column} column")
            
            # Fit vectorizer on training data only (prevents data leakage)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Transform test data using fitted vectorizer (no data leakage)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"TF-IDF transformation completed. Feature dimensions: {input_feature_train_arr.shape[1]}")
            
            # Combine features with targets
            train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]
            
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)

