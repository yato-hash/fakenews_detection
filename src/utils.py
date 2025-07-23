import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

from sklearn.metrics import accuracy_score

def save_object(file_path,obj):

    """Saves a Python object to a file using dill."""

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
                                   
def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    """
    Evaluates models using GridSearchCV to find best parameters and returns a report
    of their performance based on accuracy.
    """

    try:
        report = {}

        # Iterate through each model provided
        for model_name, model in models.items():
            # Get the hyperparameter grid for the current model
            param_grid = params[model_name]

            # This will search for the best parameters for the model from the 'param_grid'.
            # cv=3 means it will use 3-fold cross-validation, which is fast.
            # n_jobs=-1 uses all available CPU cores to speed up the search.
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            
            # Fit the grid search to the data. This performs the tuning.
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            
            # Train the model with the best parameters on the full training data
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_test_pred = model.predict(X_test)

            # Calculate the accuracy score
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):

    """Loads a Python object from a file using dill."""

    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)