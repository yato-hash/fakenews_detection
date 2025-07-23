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
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
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
            
            # Select the 'text' column for input features and handle potential missing values
            input_feature_train_df = train_df['text'].fillna('')
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df['text'].fillna('')
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying TfidfVectorizer on text data.")
            
            # Apply the vectorizer directly to the text column (pandas Series)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Convert the sparse matrix output from TF-IDF to a dense array
            # before combining it with the target column.
            train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")
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