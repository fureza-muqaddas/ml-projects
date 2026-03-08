from src.mlproject.components import data_transformation
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.utils import read_sql_data
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.utils import save_object
import sys


if __name__ == "__main__":
    logging.info("the execution has started")

    try:
        #obj= DataIngestion()
        #obj.initiate_data_ingestion()
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()
        
        train_path = r'artifacts\train.csv'
        test_path = r'artifacts\test.csv'
        data_transformation.initiate_data_transformation(train_path, test_path)



    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)