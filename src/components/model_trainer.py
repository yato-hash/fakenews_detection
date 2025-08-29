import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    """Configuration class for model training."""
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    """Handles model training, tuning, and selection."""
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Trains, tunes, and saves the best model using cross-validation.
        No data leakage - test set only used for final evaluation.
        """
        
        try:
            logging.info('Split training and test input data')
            
            # Debug: Check input data
            print(f"DEBUG: Train array shape: {train_arr.shape}")
            print(f"DEBUG: Test array shape: {test_arr.shape}")
            print(f"DEBUG: Train array dtype: {train_arr.dtype}")
            print(f"DEBUG: Test array dtype: {test_arr.dtype}")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            # Debug: Check split data
            print(f"DEBUG: X_train shape: {X_train.shape}")
            print(f"DEBUG: y_train shape: {y_train.shape}")
            print(f"DEBUG: X_test shape: {X_test.shape}")
            print(f"DEBUG: y_test shape: {y_test.shape}")
            print(f"DEBUG: Unique values in y_train: {np.unique(y_train)}")
            print(f"DEBUG: Unique values in y_test: {np.unique(y_test)}")
            print(f"DEBUG: Any NaN in X_train: {np.isnan(X_train).any()}")
            print(f"DEBUG: Any NaN in y_train: {np.isnan(y_train).any()}")
            print(f"DEBUG: X_train min/max: {X_train.min()}, {X_train.max()}")
            
            # Check for common issues
            if X_train.shape[0] == 0:
                raise CustomException('Training data is empty', sys)
            
            if len(np.unique(y_train)) < 2:
                raise CustomException(f'Training data has insufficient classes: {np.unique(y_train)}', sys)
            
            # Define models with their hyperparameter grids
            models_with_params = {
                'Logistic Regression': {
                    'model': LogisticRegression(max_iter=1000, random_state=42),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['liblinear']
                    }
                },
                'Passive Aggressive Classifier': {
                    'model': PassiveAggressiveClassifier(random_state=42),
                    'params': {
                        'C': [0.1, 0.5, 1.0]
                    }
                },
                'Multinomial Naive Bayes': {
                    'model': MultinomialNB(),
                    'params': {
                        'alpha': [0.1, 0.5, 1.0]
                    }
                }
            }
            
            logging.info('Starting model selection using cross-validation on training data only')
            
            best_model = None
            best_model_name = ""
            best_cv_score = 0
            model_cv_scores = {}
            successful_models = 0
            
            # First, let's try a simple baseline model to check if data is valid
            try:
                print("DEBUG: Testing simple baseline model...")
                from sklearn.dummy import DummyClassifier
                dummy = DummyClassifier(strategy='most_frequent')
                dummy.fit(X_train, y_train)
                dummy_score = dummy.score(X_train, y_train)
                print(f"DEBUG: Dummy classifier baseline accuracy: {dummy_score:.4f}")
                logging.info(f"Baseline dummy classifier accuracy: {dummy_score:.4f}")
            except Exception as dummy_error:
                print(f"DEBUG: Even dummy classifier failed: {dummy_error}")
                raise CustomException(f'Data validation failed. Even baseline model cannot be trained: {dummy_error}', sys)
            
            # Perform hyperparameter tuning and model selection using ONLY training data
            for name, model_config in models_with_params.items():
                try:
                    logging.info(f'Training and tuning {name}...')
                    print(f"DEBUG: Starting {name}")
                    
                    # First try fitting the base model without GridSearch
                    base_model = model_config['model']
                    print(f"DEBUG: Trying base {name} model...")
                    base_model.fit(X_train, y_train)
                    base_score = base_model.score(X_train, y_train)
                    print(f"DEBUG: Base {name} training score: {base_score:.4f}")
                    
                    # Use GridSearchCV with cross-validation on training data only
                    grid_search = GridSearchCV(
                        estimator=model_config['model'],
                        param_grid=model_config['params'],
                        cv=5,  # 5-fold cross-validation
                        scoring='accuracy',
                        n_jobs=1,  # Changed from -1 to 1 for debugging
                        
                        error_score='raise'  # This will help identify specific errors
                    )
                    
                    # Fit on training data only
                    print(f"DEBUG: Starting GridSearchCV for {name}...")
                    grid_search.fit(X_train, y_train)
                    
                    # Get the best cross-validation score
                    cv_score = grid_search.best_score_
                    model_cv_scores[name] = cv_score
                    successful_models += 1
                    
                    logging.info(f'{name} - Best CV Score: {cv_score:.4f}, Best Params: {grid_search.best_params_}')
                    print(f"DEBUG: {name} completed successfully with CV score: {cv_score:.4f}")
                    
                    # Track the best model
                    if cv_score > best_cv_score:
                        best_cv_score = cv_score
                        best_model = grid_search.best_estimator_
                        best_model_name = name
                        
                except Exception as model_error:
                    print(f"DEBUG: Error training {name}: {str(model_error)}")
                    print(f"DEBUG: Error type: {type(model_error)}")
                    logging.warning(f'Error training {name}: {str(model_error)}')
                    continue
            
            print(f"DEBUG: Successfully trained {successful_models} models")
            print(f"DEBUG: Best model: {best_model_name} with score: {best_cv_score}")
            
            if best_model is None:
                raise CustomException('No model could be trained successfully', sys)
            
            if best_cv_score < 0.6:
                raise CustomException(f'No best model found. Best CV score: {best_cv_score:.4f} is below threshold of 0.6', sys)
            
            logging.info(f'Best model selected: {best_model_name} with CV score: {best_cv_score:.4f}')
            
            # Save the best model (already trained with best hyperparameters)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f'Best model saved to {self.model_trainer_config.trained_model_file_path}')
            
            # Final evaluation on test set (used only once, for final performance assessment)
            logging.info('Evaluating best model on test set...')
            test_predictions = best_model.predict(X_test)
            final_test_accuracy = accuracy_score(y_test, test_predictions)
            
            logging.info(f'Final test accuracy: {final_test_accuracy:.4f}')
            logging.info('Model training completed successfully')
            
            return final_test_accuracy, best_model_name, model_cv_scores
            
        except Exception as e:
            print(f"DEBUG: Final exception in model trainer: {str(e)}")
            print(f"DEBUG: Exception type: {type(e)}")
            raise CustomException(str(e), sys)