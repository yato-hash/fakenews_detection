from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

if __name__ == '__main__':
    """
    Main execution script to run the full training pipeline.
    """

    # 1. Data Ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # 2. Data Transformation
    data_transformation = DataTransformation()
    # This will return the transformed numpy arrays
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # 3. Model Training
    model_trainer = ModelTrainer()
    print("Starting Model Training...")
    print(model_trainer.initiate_model_trainer(train_arr=train_arr, test_arr=test_arr))
    print("Model Training Completed.")

    # 4. Model Evaluation
    print("\nInitiating Model Evaluation...")
    evaluation = ModelEvaluation()
    evaluation.initiate_model_evaluation(test_arr=test_arr)
    print("Model Evaluation Completed.")