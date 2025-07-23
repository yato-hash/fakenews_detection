# Fake News Detection Project

## Overview

This project is a machine learning pipeline designed to classify news articles as either "Real" or "Fake". It uses Natural Language Processing (NLP) techniques to process the text and trains a classification model to make predictions. The entire workflow is modularized into a reproducible pipeline.

---

## Project Structure

-   `artifacts/`: Stores all outputs, including processed data, trained models (`model.pkl`), and preprocessors (`preprocessor.pkl`).
-   `notebook/`: Contains Jupyter notebooks for exploratory data analysis (EDA).
-   `src/`: Contains all the source code for the project.
    -   `components/`: Holds the main ML pipeline components (data ingestion, transformation, training, evaluation).
    -   `pipeline/`: Contains scripts to orchestrate the pipeline (training and prediction).
    -   `exception.py`, `logger.py`, `utils.py`: Utility scripts for error handling, logging, and common functions.
-   `requirements.txt`: A list of all Python dependencies required to run the project.

---

## How to Run

### Step 1: Set up the Environment

1.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Run the Training Pipeline

To train the model from scratch, run the main training pipeline script. This will perform data ingestion, data transformation, model training, and evaluation.

 ```bash
    python src/pipeline/train_pipeline.py
    ```


### Step 3: Test with the Prediction Pipeline (Optional)

After training, we can test the model with a single prediction directly from our terminal. We can change the sample text inside the src/pipeline/predict_pipeline.py file to test different inputs.

```bash
    python src/pipeline/predict_pipeline.py
    ```

### Step 4: Deploy as an Interactive Web Application
We can launch a user-friendly web interface built with Streamlit,by running the following command from the project's root directory:

```bash
    streamlit run app.py
    ```
This will open a new tab in our web browser where we can paste any news article text for real-time classification.
