import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from .data_cleaning import FakeNewsDataCleaner

@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion paths."""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    cleaned_data_path: str = os.path.join('artifacts', 'cleaned.csv')

class DataIngestion:
    """Handles reading, cleaning, and splitting the dataset."""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.cleaner = FakeNewsDataCleaner()

    def initiate_data_ingestion(self, input_file_path: str):
        logging.info('Entered the data ingestion method or component')
        try:
            # Step 1: Read the raw data
            logging.info('Reading the raw dataset from input file')
            df_raw = self.cleaner.load_data("notebook\model_data.csv")

            # Step 2: Clean the entire dataset
            # This is the crucial step to prevent data leakage.
            # All cleaning operations (handling NaNs, text cleaning,
            # removing duplicates, etc.) are applied to the full dataset.
            logging.info('Initiating data cleaning process on the raw dataset')
            df_cleaned = self.cleaner.clean_dataset(df_raw)
            
            # Save the fully cleaned dataset
            os.makedirs(os.path.dirname(self.ingestion_config.cleaned_data_path), exist_ok=True)
            df_cleaned.to_csv(self.ingestion_config.cleaned_data_path, index=False, header=True)
            logging.info(f'Cleaned data saved to {self.ingestion_config.cleaned_data_path}')

            # Step 3: Split the cleaned data into training and testing sets
            logging.info('Initiating train-test split on the cleaned data')
            train_set, test_set = train_test_split(df_cleaned, test_size=0.2, random_state=42)

            # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Ingestion of the data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Example of how to use this new class
if __name__ == "__main__":
    # Assuming 'notebook\model_data.csv' is the path to your raw data
    input_file = 'notebook/model_data.csv' 
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' was not found.")
        sys.exit(1)
        
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion(input_file)
    print(f"Train data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")