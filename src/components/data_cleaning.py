import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from typing import Optional

class FakeNewsDataCleaner:
    
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Dataset loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        text = re.sub(r'[.!?]{2,}', lambda m: m.group()[0], text)
        
        return text.strip()
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        if pd.isna(date_str) or date_str == "" or str(date_str) == "########":
            return None
        
        date_formats = ['%d-%b-%y', '%d-%B-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
        date_str = str(date_str).strip()
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'label' not in df.columns:
            return df
        
        valid_labels = [0, 1]
        invalid_mask = ~df['label'].isin(valid_labels + [np.nan])
        
        if invalid_mask.any():
            df = df[~invalid_mask]
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset_columns: list = None) -> pd.DataFrame:
        if subset_columns is None:
            subset_columns = ['title', 'text']
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset_columns, keep='first')
        
        if len(df) < initial_count:
            self.logger.info(f"Removed {initial_count - len(df)} duplicates")
        
        return df
    
    def filter_by_length(self, df: pd.DataFrame, min_text_length: int = 50) -> pd.DataFrame:
        initial_count = len(df)
        
        if 'title' in df.columns:
            df = df[df['title'].str.len() >= 10]
        
        if 'text' in df.columns:
            df = df[df['text'].str.len() >= min_text_length]
        
        if len(df) < initial_count:
            self.logger.info(f"Filtered {initial_count - len(df)} articles by length")
        
        return df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Handle missing values
        if 'title' in df_clean.columns:
            df_clean['title'] = df_clean['title'].fillna('')
        if 'text' in df_clean.columns:
            df_clean['text'] = df_clean['text'].fillna('')
        if 'subject' in df_clean.columns:
            df_clean['subject'] = df_clean['subject'].fillna('unknown')
        
        # Clean text
        if 'title' in df_clean.columns:
            df_clean['title_cleaned'] = df_clean['title'].apply(self.clean_text)
        if 'text' in df_clean.columns:
            df_clean['text_cleaned'] = df_clean['text'].apply(self.clean_text)
        
        # Parse dates
        if 'date' in df_clean.columns:
            df_clean['date_parsed'] = df_clean['date'].apply(self.parse_date)
        
        # Validate labels and remove duplicates
        df_clean = self.validate_labels(df_clean)
        df_clean = self.remove_duplicates(df_clean, ['title_cleaned', 'text_cleaned'])
        df_clean = self.filter_by_length(df_clean)
        
        # Create full text column
        if 'title_cleaned' in df_clean.columns and 'text_cleaned' in df_clean.columns:
            df_clean['full_text'] = (df_clean['title_cleaned'] + ' ' + 
                                   df_clean['text_cleaned']).str.strip()
        
        # Final cleanup
        if 'full_text' in df_clean.columns:
            df_clean = df_clean[df_clean['full_text'].str.len() > 0]
        if 'label' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['label'])
        
        return df_clean.reset_index(drop=True)
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved to: {output_path}")

def clean_raw_data_pipeline(input_path: str, output_path: str = None) -> pd.DataFrame:
    cleaner = FakeNewsDataCleaner()
    
    try:
        df = cleaner.load_data(input_path)
        df_cleaned = cleaner.clean_dataset(df)
        
        if output_path:
            cleaner.save_cleaned_data(df_cleaned, output_path)
        else:
            output_path = input_path.replace('.csv', '_cleaned.csv')
            cleaner.save_cleaned_data(df_cleaned, output_path)
        
        return df_cleaned
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_cleaning.py <input_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    clean_raw_data_pipeline(input_file, output_file)