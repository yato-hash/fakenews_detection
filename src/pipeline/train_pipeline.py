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
    train_data_path, test_data_path = obj.initiate_data_ingestion(input_file_path="notebook\model_data.csv")

    # 2. Data Transformation
    data_transformation = DataTransformation()
    # This will return the transformed numpy arrays
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print(train_arr.shape, test_arr.shape)

    # Add this to your pipeline to check for data leakage
    train_df = pd.read_csv('artifacts/train.csv')
    test_df = pd.read_csv('artifacts/test.csv')

    # Check for exact duplicates
    train_texts = set(train_df['full_text'].fillna(''))
    test_texts = set(test_df['full_text'].fillna(''))
    overlap = train_texts.intersection(test_texts)
    print(f"Exact overlaps between train/test: {len(overlap)}")

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
