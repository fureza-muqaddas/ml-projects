from src.mlproject.components import data_transformation
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.utils import read_sql_data
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.utils import save_object
import sys
from src.mlproject.utils import evaluate_models
from src.mlproject.components.model_trainer import ModelTrainerConfig, ModelTrainer

if __name__ == "__main__":
    logging.info("the execution has started")

    try:
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()
        
        train_path = r'artifacts\train.csv'
        test_path = r'artifacts\test.csv'

        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)