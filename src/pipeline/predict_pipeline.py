import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    Pipeline for making predictions on new data.
    """
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Helper class to format new input data into a DataFrame.
    """
    def __init__(self, text: str):
        self.text = text

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {"text": [self.text]}
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Example usage
    sample_text = "Breaking News: Scientists discover water on Mars, raising hopes for future colonization efforts."
    
    custom_data = CustomData(text=sample_text)
    pred_df = custom_data.get_data_as_frame()
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    if results[0] == 0:
        print("Prediction: The news is likely REAL.")
    else:
        print("Prediction: The news is likely FAKE.")