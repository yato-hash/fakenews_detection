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




