import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class ModelEvaluation:
    """Handles evaluation of the trained model."""
    def __init__(self):
        pass

    def initiate_model_evaluation(self, test_arr):
        """Calculates and prints evaluation metrics."""
        try:
            logging.info("Splitting test array into features and labels")
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("Loading the trained model from artifacts/model.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)

            logging.info("Making predictions on the test set")
            predictions = model.predict(X_test)
            
            # --- Calculate all the required metrics ---
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, pos_label=1)
            recall = recall_score(y_test, predictions, pos_label=1)
            f1 = f1_score(y_test, predictions, pos_label=1)
            class_report = classification_report(y_test, predictions)
            conf_matrix = confusion_matrix(y_test, predictions)
            
            # ROC-AUC score requires prediction probabilities
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except AttributeError:
                roc_auc = "N/A (model does not support predict_proba)"

            logging.info("Model evaluation metrics calculated successfully")
            
            
            print("="*30)
            print("--- MODEL EVALUATION METRICS ---")
            print("="*30)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (for 'fake' class): {precision:.4f}")
            print(f"Recall (for 'fake' class): {recall:.4f}")
            print(f"F1-Score (for 'fake' class): {f1:.4f}")
            print(f"ROC-AUC Score: {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
            print("\nClassification Report:\n", class_report)
            print("\nConfusion Matrix:\n", conf_matrix)
            print("="*30)

           
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)