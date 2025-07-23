import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    """Configuration class for model training."""
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    """Handles model training, tuning, and selection."""

    def __init__(self):
        self.model_trainer_config =  ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):

        """Trains, tunes, and saves the best model."""
        
        try:
            logging.info('Split training and test input data')
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Passive Aggressive Classifier': PassiveAggressiveClassifier(),
                'Multinomial Naive Bayes': MultinomialNB()
            }
            params = {
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],  # Regularization parameter
                    'solver': ['liblinear'] # A good solver for this kind of problem
                },
                "Passive Aggressive Classifier": {
                    'C': [0.1, 0.5, 1.0] # Maximum step size (regularization)
                },
                "Multinomial Naive Bayes": {
                    'alpha': [0.1, 0.5, 1.0] # Additive smoothing parameter
                }
            }

            # This will now perform hyperparameter tuning before giving a score.
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best found model on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy,best_model_name
        except Exception as e:
            raise CustomException(e,sys)